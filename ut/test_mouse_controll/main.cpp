#include "MouseRectangleTest.h"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        return MouseRectangleTest(argc > 1 ? argv[1] : "absolute").run();
    } catch (std::exception const& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
