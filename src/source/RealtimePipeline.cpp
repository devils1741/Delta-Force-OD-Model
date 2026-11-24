#include "RealtimePipeline.h"
#include "MouseController.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
void checkCuda(cudaError_t status, char const* operation) {
    if(status!=cudaSuccess) throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
using Clock=std::chrono::steady_clock;
}
struct AsyncMousePipeline::Impl {
    struct Slot {
        void* device{};
        void* host{};
        cudaEvent_t ready{}, copied{};
        bool pending{};
        uint64_t sequence{};
        LetterboxInfo letterbox{};
        Clock::time_point submitted{};
    };
    std::array<Slot,3> slots;
    cudaStream_t transfer{};
    int capacity{}, deviceIndex{};
    float keypointThreshold{0.3f};
    std::unique_ptr<MouseController> mouse;
#ifdef DELTA_GPU_TESTING
    TestConsumer testConsumer;
#endif
    std::mutex mutex;
    std::exception_ptr error;
    std::atomic_bool stopping{false};
    std::thread worker;
    ~Impl() {
        stopping.store(true);
        if(worker.joinable()) worker.join();
        // A producer may still be filling a snapshot. Only shutdown waits here.
        for(auto& slot:slots) if(slot.ready) cudaEventSynchronize(slot.ready);
        if(transfer) cudaStreamSynchronize(transfer);
        for(auto& slot:slots) {
            if(slot.ready) cudaEventDestroy(slot.ready);
            if(slot.copied) cudaEventDestroy(slot.copied);
            if(slot.device) cudaFree(slot.device);
            if(slot.host) cudaFreeHost(slot.host);
        }
        if(transfer) cudaStreamDestroy(transfer);
    }
    void initialize(int capacity) {
        auto& s=*this; s.capacity=capacity;
        checkCuda(cudaGetDevice(&s.deviceIndex),"mouse cudaGetDevice");
        checkCuda(cudaStreamCreateWithFlags(&s.transfer,cudaStreamNonBlocking),"mouse transfer stream");
        size_t bytes=16+static_cast<size_t>(capacity)*sizeof(Box);
        for(auto& slot:s.slots) {
            checkCuda(cudaMalloc(&slot.device,bytes),"mouse snapshot allocation");
            checkCuda(cudaMallocHost(&slot.host,bytes),"mouse pinned allocation");
            checkCuda(cudaEventCreateWithFlags(&slot.ready,cudaEventDisableTiming),"mouse ready event");
            checkCuda(cudaEventCreateWithFlags(&slot.copied,cudaEventDisableTiming),"mouse copied event");
        }
    }
    void run() {
        try {
            checkCuda(cudaSetDevice(deviceIndex),"mouse cudaSetDevice");
            uint64_t lastSequence=0;
            while(!stopping.load()) {
                std::vector<Box> boxes;
                LetterboxInfo letterbox{};
                uint64_t sequence=0;
                {
                    std::lock_guard lock(mutex);
                    Slot* latest=nullptr;
                    for(auto& slot:slots) {
                        if(!slot.pending) continue;
                        auto status=cudaEventQuery(slot.copied);
                        if(status==cudaErrorNotReady) continue;
                        checkCuda(status,"mouse readback event");
                        slot.pending=false;
                        if(slot.sequence>lastSequence && (!latest || slot.sequence>latest->sequence)) latest=&slot;
                    }
                    // Reject old control results after a stall instead of moving toward stale positions.
                    if(latest && Clock::now()-latest->submitted<std::chrono::milliseconds(100)) {
                        int count=std::clamp(*static_cast<int const*>(latest->host),0,capacity);
                        auto data=reinterpret_cast<Box const*>(static_cast<char const*>(latest->host)+16);
                        boxes.assign(data,data+count);
                        letterbox=latest->letterbox; sequence=latest->sequence;
                        lastSequence=sequence;
                    }
                }
                if(sequence) {
#ifdef DELTA_GPU_TESTING
                    if(testConsumer) testConsumer(boxes,letterbox,sequence);
                    else
#endif
                    mouse->update(boxes,letterbox,sequence,keypointThreshold);
                }
                else std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        } catch(...) {
            std::lock_guard lock(mutex);
            error=std::current_exception();
        }
    }
};
AsyncMousePipeline::AsyncMousePipeline(MouseConfig const& config, int capacity, float keypointThreshold)
    : impl_(std::make_unique<Impl>()) {
    if(!config.enabled) return;
    auto& s=*impl_;
    s.keypointThreshold=keypointThreshold;
    s.initialize(capacity);
    s.mouse=std::make_unique<MouseController>(config);
    s.worker=std::thread([&s]{s.run();});
}
AsyncMousePipeline::~AsyncMousePipeline()=default;
void AsyncMousePipeline::checkError() {
    std::lock_guard lock(impl_->mutex);
    if(impl_->error) std::rethrow_exception(impl_->error);
}
void AsyncMousePipeline::submit(Box const* boxes, int const* count,
    LetterboxInfo const& letterbox, uint64_t sequence, cudaStream_t producer) {
    auto& s=*impl_;
    if(!s.transfer) return;
    std::unique_lock lock(s.mutex,std::try_to_lock);
    if(!lock.owns_lock()) return;
    if(s.error) std::rethrow_exception(s.error);
    for(auto& slot:s.slots) {
        if(slot.pending) continue;
        // Snapshot on the producer stream before next frame overwrites its boxes.
        checkCuda(cudaMemcpyAsync(slot.device,count,sizeof(int),cudaMemcpyDeviceToDevice,producer),"snapshot count");
        checkCuda(cudaMemcpyAsync(static_cast<char*>(slot.device)+16,boxes,
            static_cast<size_t>(s.capacity)*sizeof(Box),cudaMemcpyDeviceToDevice,producer),"snapshot boxes");
        checkCuda(cudaEventRecord(slot.ready,producer),"record snapshot ready");
        checkCuda(cudaStreamWaitEvent(s.transfer,slot.ready,0),"wait snapshot on transfer stream");
        checkCuda(cudaMemcpyAsync(slot.host,slot.device,sizeof(int),cudaMemcpyDeviceToHost,s.transfer),"async mouse count");
        checkCuda(cudaMemcpyAsync(static_cast<char*>(slot.host)+16,static_cast<char*>(slot.device)+16,
            static_cast<size_t>(s.capacity)*sizeof(Box),cudaMemcpyDeviceToHost,s.transfer),"async mouse boxes");
        checkCuda(cudaEventRecord(slot.copied,s.transfer),"record mouse copy complete");
        slot.letterbox=letterbox; slot.sequence=sequence;
        slot.submitted=Clock::now(); slot.pending=true;
        return;
    }
    // All slots busy: skip mouse readback; rendering continues with the current GPU result.
}

#ifdef DELTA_GPU_TESTING
AsyncMousePipeline::AsyncMousePipeline(int capacity, TestConsumer consumer)
    : impl_(std::make_unique<Impl>()) {
    auto& s=*impl_;
    s.initialize(capacity);
    s.testConsumer=std::move(consumer);
    s.worker=std::thread([&s]{s.run();});
}
#endif
