#ifndef RAYTRACINGDEMO_AABB_HPP
#define RAYTRACINGDEMO_AABB_HPP


#include <vector>
#include <memory>
#include <utility>
#include <algorithm>

#include "primitives/vector3.hpp"
#include "primitives/ray.hpp"
#include "primitives/primitive.hpp"

class AABB {    //Axis-Aligned Bounding Box

    Vector3 min; //Minimum corner
    Vector3 max; //Maximum corner
    std::vector<std::unique_ptr<Primitive>> primitives;

public:
    AABB(Vector3 min_, Vector3 max_, std::vector<std::unique_ptr<Primitive>>&& objs)
        : min(std::move(min_)), max(std::move(max_)), primitives(std::move(objs)) {}

    //copy
    AABB(const AABB& other) : min(other.min), max(other.max) {
        primitives.reserve(other.primitives.size());
        for (const auto &p : other.primitives) {
            primitives.push_back(p->clone());
        }
    }
    AABB& operator=(const AABB& other) {
        if (this == &other) return *this;
        min = other.min;
        max = other.max;
        primitives.clear();
        primitives.reserve(other.primitives.size());
        for (const auto &p : other.primitives) {
            primitives.push_back(p->clone());
        }
        return *this;
    }
    // moves
    AABB(AABB&&) noexcept = default;
    AABB& operator=(AABB&&) noexcept = default;

    const Vector3& getMin() const { return min; }
    const Vector3& getMax() const { return max; }

    const std::vector<std::unique_ptr<Primitive>>& getPrimitives() const { return primitives; }
    std::vector<std::unique_ptr<Primitive>>& getPrimitives() { return primitives; } // optional non-const overload

    // TODO: Optimize hit function (e.g., precompute inverse dir or use branchless slabs)
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
