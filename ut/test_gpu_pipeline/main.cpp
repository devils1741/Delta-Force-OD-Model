#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <cuda_d3d11_interop.h>
#include <wrl/client.h>
#include "CudaPostprocess.h"
#include "OverlayWindow.h"
#include "RealtimePipeline.h"
#include <atomic>
#include <thread>
#include <chrono>
#include <future>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
using Microsoft::WRL::ComPtr;
void require(bool condition,char const* text) { if(!condition) throw std::runtime_error(text); }
void cudaCheck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
// A different UI thread is essential: HTTRANSPARENT alone only works within one thread.
struct InputTarget {
    HWND hwnd{};
    std::thread thread;
    InputTarget() {
        std::promise<HWND> ready;
        auto future=ready.get_future();
        thread=std::thread([ready=std::move(ready)]() mutable {
            WNDCLASSW wc{}; wc.hInstance=GetModuleHandleW(nullptr); wc.lpszClassName=L"DeltaInputTest";
            wc.lpfnWndProc=[](HWND h,UINT m,WPARAM w,LPARAM l)->LRESULT {
                if(m==WM_DESTROY) { PostQuitMessage(0); return 0; }
                return DefWindowProcW(h,m,w,l);
            };
            RegisterClassW(&wc);
            HWND h=CreateWindowExW(WS_EX_TOPMOST|WS_EX_NOACTIVATE,wc.lpszClassName,
                L"GPU overlay input test",WS_POPUP,0,0,320,240,nullptr,nullptr,wc.hInstance,nullptr);
            ShowWindow(h,SW_SHOWNOACTIVATE);
            ready.set_value(h);
            MSG msg{};
            while(GetMessageW(&msg,nullptr,0,0)>0) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        });
        hwnd=future.get();
    }
    ~InputTarget() { if(hwnd) PostMessageW(hwnd,WM_CLOSE,0,0); if(thread.joinable()) thread.join(); }
};
int main() {
    try {
        ComPtr<IDXGIFactory1> factory;
        require(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))),"factory");
        ComPtr<IDXGIAdapter1> adapter;
        int cudaDevice=-1;
        for(UINT index=0;;++index) {
            adapter.Reset();
            if(factory->EnumAdapters1(index,&adapter)==DXGI_ERROR_NOT_FOUND) break;
            if(cudaD3D11GetDevice(&cudaDevice,adapter.Get())==cudaSuccess) break;
        }
        require(adapter && cudaDevice>=0,"No CUDA D3D11 adapter");
        cudaCheck(cudaSetDevice(cudaDevice));
        ComPtr<ID3D11Device> device;
        ComPtr<ID3D11DeviceContext> context;
        require(SUCCEEDED(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,nullptr,&context)),"D3D device");
        cudaStream_t stream{}; cudaCheck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        float* output{}; cudaCheck(cudaMalloc(reinterpret_cast<void**>(&output),30*sizeof(float)));
        float host[30]={100,100,100,250,200, 100,100,100,100,200,
                       80,80,80,4,40, 100,100,100,4,40,
                       .9f,.8f,.1f,.95f,.1f, .1f,.1f,.85f,.1f,.1f};
        cudaCheck(cudaMemcpyAsync(output,host,sizeof(host),cudaMemcpyHostToDevice,stream));
        LetterboxInfo info{}; info.inputW=info.screenW=320; info.inputH=info.screenH=240;
        info.captureX=20; info.captureY=20; info.captureW=280; info.captureH=200;
        // Decode coordinates are offset by ROI origin; use an identity mapping for test boxes.
        auto decodeInfo=info; decodeInfo.captureX=decodeInfo.captureY=0;
        CudaPostprocessor post(5);
        InputTarget inputTarget;
        GpuOverlay overlay(device.Get(),context.Get(),info,5);
        for(int frame=0;frame<12;++frame) {
            float threshold=frame%2==0 ? .5f : 1.f;
            post.enqueue(output,5,decodeInfo,threshold,10,10,stream);
            overlay.draw(post.deviceBoxes(),post.deviceCount(),info,true,stream);
            auto pixels=overlay.readPixelsForTest();
            require(WindowFromPoint(POINT{100,100})==inputTarget.hwnd,
                "Overlay blocks hit testing of a window on another UI thread");
            int count=-1; cudaCheck(cudaMemcpy(&count,post.deviceCount(),sizeof(int),cudaMemcpyDeviceToHost));
            require(count==(frame%2==0 ? 2 : 0),"NMS, class isolation, size filter or empty frame failed");
            require(pixels[20*320+20]==0xffff0000,"ROI border is not opaque red");
            require(pixels[100*320+100]==0,"Overlay interior is not transparent");
            require((frame%2==0 ? pixels[50*320+60]!=0 : pixels[50*320+60]==0),"Detection border or stale-frame clearing failed");
            if(frame%2==0) {
                bool label=false;
                for(int y=36;y<46;++y) for(int x=60;x<100;++x)
                    label |= pixels[y*320+x]!=0;
                require(label,"GPU confidence label missing");
            }
        }
        std::atomic<int> received{0};
        std::atomic<bool> valid{true};
        {
            AsyncMousePipeline readback(5,[&](std::vector<Box> const& boxes,LetterboxInfo const& lb,uint64_t sequence) {
                int expected=sequence%2==0 ? 2 : 0;
                if(static_cast<int>(boxes.size())!=expected || lb.captureX!=20) valid.store(false);
                for(auto const& box:boxes) if(box.x1!=60 || box.y1!=50) valid.store(false);
                received.fetch_add(1);
                // Slow CPU consumer deliberately: producer must skip busy slots, not wait.
                std::this_thread::sleep_for(std::chrono::milliseconds(3));
            });
            for(uint64_t sequence=1;sequence<=100;++sequence) {
                post.enqueue(output,5,decodeInfo,sequence%2==0 ? .5f : 1.f,10,10,stream);
                readback.submit(post.deviceBoxes(),post.deviceCount(),info,sequence,stream);
                if(sequence%10==0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            cudaCheck(cudaStreamSynchronize(stream));
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(1);
            while(received.load()==0 && std::chrono::steady_clock::now()<deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            readback.checkError();
        }
        require(received.load()>0 && valid.load(),"Async readback snapshot corruption or no completed results");
        cudaCheck(cudaStreamSynchronize(stream));
        // End-to-end pose rows: overlapping detections must survive without NMS.
        float poseRows[24]={40,40,80,100,.9f,2,50,50,.8f,60,80,.9f,
                            40,40,80,100,.8f,2,50,50,.8f,60,80,.9f};
        float* poseOutput{};
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&poseOutput),sizeof(poseRows)));
        cudaCheck(cudaMemcpyAsync(poseOutput,poseRows,sizeof(poseRows),cudaMemcpyHostToDevice,stream));
        post.enqueue(poseOutput,2,decodeInfo,.5f,0,0,stream,true);
        int poseCount=0; Box poseBoxes[2]{};
        cudaCheck(cudaMemcpyAsync(&poseCount,post.deviceCount(),sizeof(int),cudaMemcpyDeviceToHost,stream));
        cudaCheck(cudaMemcpyAsync(poseBoxes,post.deviceBoxes(),sizeof(poseBoxes),cudaMemcpyDeviceToHost,stream));
        cudaCheck(cudaStreamSynchronize(stream));
        require(poseCount==2,"Pose output unexpectedly suppressed by NMS");
        require(poseBoxes[0].classId==2 && poseBoxes[0].keypoints[2]>.79f,"Pose class/keypoint decoding failed");
        overlay.draw(post.deviceBoxes(),post.deviceCount(),info,false,stream,true,.3f,4.f,true,true);
        auto posePixels=overlay.readPixelsForTest();
        require(posePixels[50*320+50]!=0,"Keypoint not rendered");
        require(posePixels[65*320+55]!=0,"Keypoint connecting line missing");
        overlay.draw(post.deviceBoxes(),post.deviceCount(),info,false,stream,false,.3f,4.f,true,true);
        posePixels=overlay.readPixelsForTest();
        require(posePixels[50*320+50]==0,"Keypoint show flag ignored");
        poseRows[5]=3; poseRows[4]=.4f;
        poseRows[2]=45; poseRows[3]=45; poseRows[16]=0;
        cudaCheck(cudaMemcpyAsync(poseOutput,poseRows,sizeof(poseRows),cudaMemcpyHostToDevice,stream));
        post.enqueue(poseOutput,2,decodeInfo,.8f,24,48,stream,true,.3f,0,0);
        cudaCheck(cudaMemcpyAsync(&poseCount,post.deviceCount(),sizeof(int),cudaMemcpyDeviceToHost,stream));
        cudaCheck(cudaStreamSynchronize(stream));
        require(poseCount==1,"Logo independent confidence/size filters failed");
        overlay.draw(post.deviceBoxes(),post.deviceCount(),info,false,stream,true,.3f,4.f,true,false);
        posePixels=overlay.readPixelsForTest();
        require(posePixels[40*320+40]==0,"Logo display flag ignored");
        // A centered 1280x1280 crop on a 4K monitor maps model coordinates back
        // to monitor-local pixels, regardless of the monitor's desktop origin.
        LetterboxInfo fourK{};
        fourK.inputW=fourK.inputH=fourK.resizedW=fourK.resizedH=640;
        fourK.screenW=3840; fourK.screenH=2160;
        fourK.captureW=fourK.captureH=1280;
        fourK.captureX=1280; fourK.captureY=440;
        fourK.desktopX=-3840; fourK.desktopY=2160;
        fourK.scale=.5f;
        float fourKRow[12]={300,300,400,400,.9f,0,320,320,.9f,370,345,.8f};
        cudaCheck(cudaMemcpyAsync(output,fourKRow,sizeof(fourKRow),cudaMemcpyHostToDevice,stream));
        post.enqueue(output,1,fourK,.5f,0,0,stream,true);
        int fourKCount=0; Box fourKBox{};
        cudaCheck(cudaMemcpyAsync(&fourKCount,post.deviceCount(),sizeof(int),cudaMemcpyDeviceToHost,stream));
        cudaCheck(cudaMemcpyAsync(&fourKBox,post.deviceBoxes(),sizeof(Box),cudaMemcpyDeviceToHost,stream));
        cudaCheck(cudaStreamSynchronize(stream));
        require(fourKCount==1 && fourKBox.classId==0,"4K pose count/class decoding failed");
        require(fourKBox.x1==1880 && fourKBox.y1==1040 &&
                fourKBox.x2==2080 && fourKBox.y2==1240,
                "4K pose box ROI scaling/origin mapping failed");
        require(fourKBox.keypoints[0]==1920 && fourKBox.keypoints[1]==1080 &&
                fourKBox.keypoints[3]==2020 && fourKBox.keypoints[4]==1130,
                "4K pose keypoints must be monitor-local without a desktop offset");
        cudaFree(poseOutput);
        cudaFree(output); cudaStreamDestroy(stream);
        std::cout << "PASS: cross-thread mouse hit-test passthrough, GPU decode/NMS/class isolation/size filter, CUDA-D3D sharing, transparent boxes, confidence labels, ROI, 4K monitor-local pose mapping, empty frames, buffer reuse (12 frames), asynchronous mouse snapshots under slow-consumer load (100 submissions).\n";
        return 0;
    } catch(std::exception const& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}

