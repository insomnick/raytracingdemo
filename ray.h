#ifndef RAYTRACINGDEMO_RAY_H
#define RAYTRACINGDEMO_RAY_H

#include <vector>

struct Ray {
    std::vector<double> origin;
    std::vector<double> direction;
    Ray(): origin({0.0,0.0,0.0}), direction({0.0,0.0,0.0}) {}
    Ray(double ox, double oy, double oz, double dx, double dy, double dz)
        : origin({ox, oy, oz}), direction({dx, dy, dz}) {}
};

#endif //RAYTRACINGDEMO_RAY_H
