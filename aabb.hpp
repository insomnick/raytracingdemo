#ifndef RAYTRACINGDEMO_AABB_HPP
#define RAYTRACINGDEMO_AABB_HPP


#include <memory>
#include <vector>
#include "vector3.hpp"
#include "ray.hpp"
#include "sphere.hpp"

class AABB {    //Axis-Aligned Bounding Box

    Vector3 min; //Minimum corner
    Vector3 max; //Maximum corner
    std::vector<Sphere> primitives;

public:
    AABB(const Vector3& min, const Vector3& max, const std::vector<Sphere> objects) : min(min), max(max), primitives(objects) {}
    Vector3 getMin() const { return min; }
    Vector3 getMax() const { return max; }
    bool containsPrimitives() const { return !primitives.empty(); }
    std::vector<Sphere> getPrimitives() const { return primitives; }

    bool hit(const Ray& ray) const {
        // Using the "slab"
        double tmin = (min.getX() - ray.getOrigin().getX()) / ray.getDirection().getX();
        double tmax = (max.getX() - ray.getOrigin().getX()) / ray.getDirection().getX();
        if (tmin > tmax) std::swap(tmin, tmax);
        double tymin = (min.getY() - ray.getOrigin().getY()) / ray.getDirection().getY();
        double tymax = (max.getY() - ray.getOrigin().getY()) / ray.getDirection().getY();
        if (tymin > tymax) std::swap(tymin, tymax);
        if ((tmin > tymax) || (tymin > tmax))
            return false;
        if (tymin > tmin)
            tmin = tymin;
        if (tymax < tmax)
            tmax = tymax;
        double tzmin = (min.getZ() - ray.getOrigin().getZ()) / ray.getDirection().getZ();
        double tzmax = (max.getZ() - ray.getOrigin().getZ()) / ray.getDirection().getZ();
        if (tzmin > tzmax) std::swap(tzmin, tzmax);
        if ((tmin > tzmax) || (tzmin > tmax))
            return false;
        if (tzmin > tmin)
            tmin = tzmin;
        if (tzmax < tmax)
            tmax = tzmax;
        return true;
    }
};


#endif //RAYTRACINGDEMO_AABB_HPP
