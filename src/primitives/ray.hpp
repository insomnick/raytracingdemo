#ifndef RAYTRACINGDEMO_RAY_HPP
#define RAYTRACINGDEMO_RAY_HPP

#include <limits>
#include "vector3.hpp"

struct Ray {
    Vector3 origin;
    Vector3 direction; // should be normalized
    Vector3 invDir;    // component-wise inverse
    Ray(const Vector3& o, const Vector3& d): origin(o), direction(d){
        invDir = { d.getX()!=0.f? 1.f/d.getX(): std::numeric_limits<float>::infinity(),
                   d.getY()!=0.f? 1.f/d.getY(): std::numeric_limits<float>::infinity(),
                   d.getZ()!=0.f? 1.f/d.getZ(): std::numeric_limits<float>::infinity()};
    }
};

#endif //RAYTRACINGDEMO_RAY_HPP
