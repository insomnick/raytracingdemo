#ifndef RAYTRACINGDEMO_RAY_H
#define RAYTRACINGDEMO_RAY_H

#include "vector3.h"

struct Ray {
    Vector3 origin;
    Vector3 direction;
    Ray(): origin({0.0,0.0,0.0}), direction({0.0,0.0,0.0}) {}
    Ray(Vector3 origin, Vector3 direction) : origin(origin), direction(direction) {}
};

#endif //RAYTRACINGDEMO_RAY_H
