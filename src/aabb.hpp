#ifndef RAYTRACINGDEMO_AABB_HPP
#define RAYTRACINGDEMO_AABB_HPP

#include <vector>
#include <algorithm>
#include "primitives/vector3.hpp"
#include "primitives/ray.hpp"
#include "primitives/primitive.hpp"

struct AABB {
    Vector3 minB{0,0,0};
    Vector3 maxB{0,0,0};
    std::vector<Primitive*> prims; // leaf primitives (if leaf)
    bool intersect(const Ray& r, float& tminOut, float& tmaxOut) const {
        float tmin = ( (r.invDir.getX()>=0? minB.getX(): maxB.getX()) - r.origin.getX()) * r.invDir.getX();
        float tmax = ( (r.invDir.getX()>=0? maxB.getX(): minB.getX()) - r.origin.getX()) * r.invDir.getX();
        float tymin = ( (r.invDir.getY()>=0? minB.getY(): maxB.getY()) - r.origin.getY()) * r.invDir.getY();
        float tymax = ( (r.invDir.getY()>=0? maxB.getY(): minB.getY()) - r.origin.getY()) * r.invDir.getY();
        if ( (tmin > tymax) || (tymin > tmax)) return false;
        if (tymin > tmin) tmin = tymin; if (tymax < tmax) tmax = tymax;
        float tzmin = ( (r.invDir.getZ()>=0? minB.getZ(): maxB.getZ()) - r.origin.getZ()) * r.invDir.getZ();
        float tzmax = ( (r.invDir.getZ()>=0? maxB.getZ(): minB.getZ()) - r.origin.getZ()) * r.invDir.getZ();
        if ( (tmin > tzmax) || (tzmin > tmax)) return false;
        if (tzmin > tmin) tmin = tzmin; if (tzmax < tmax) tmax = tzmax;
        if (tmax < 0) return false;
        tminOut = tmin; tmaxOut = tmax; return true;
    }
};

#endif //RAYTRACINGDEMO_AABB_HPP
