#ifndef RAYTRACINGDEMO_TIMER_HPP
#define RAYTRACINGDEMO_TIMER_HPP

#include <chrono>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
public:
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}
    void reset() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    double elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end_time - start_time;
        return diff.count();
    }
};

#endif //RAYTRACINGDEMO_TIMER_HPP
