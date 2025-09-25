#ifndef RAYTRACINGDEMO_RAY_HPP
#define RAYTRACINGDEMO_RAY_HPP

#include "vector3.hpp"

struct Ray {
private:
    Vector3 origin;
    Vector3 direction;
public:
    Ray(): origin({0.0,0.0,0.0}), direction({0.0,0.0,0.0}) {}
    Ray(Vector3 origin, Vector3 direction) : origin(origin), direction(direction) {}
    Vector3 getOrigin() const { return origin; }
    Vector3 getDirection() const { return direction; }
};

#endif //RAYTRACINGDEMO_RAY_HPP
