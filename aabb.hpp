#ifndef RAYTRACINGDEMO_AABB_HPP
#define RAYTRACINGDEMO_AABB_HPP


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
    std::vector<Sphere> getPrimitives() const { return primitives; }

    //TODO: Optimize hit function because of eyes on the back of the ray direction
    bool hit(const Ray& ray) const {
        // Using "slab" method with 3 planes
        double tx1 = (min.getX() - ray.getOrigin().getX())*ray.getInvDirection().getX();
        double tx2 = (max.getX() - ray.getOrigin().getX())*ray.getInvDirection().getX();
        double tmin = std::min(tx1, tx2);
        double tmax = std::max(tx1, tx2);

        double ty1 = (min.getY() - ray.getOrigin().getY())*ray.getInvDirection().getY();
        double ty2 = (max.getY() - ray.getOrigin().getY())*ray.getInvDirection().getY();
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));

        double tz1 = (min.getZ() - ray.getOrigin().getZ())*ray.getInvDirection().getZ();
        double tz2 = (max.getZ() - ray.getOrigin().getZ())*ray.getInvDirection().getZ();
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));

        return tmax >= tmin;
    }
};


#endif //RAYTRACINGDEMO_AABB_HPP
