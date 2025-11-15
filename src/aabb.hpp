#ifndef RAYTRACINGDEMO_AABB_HPP
#define RAYTRACINGDEMO_AABB_HPP


#include <vector>
#include <algorithm>

#include "primitives/vector3.hpp"
#include "primitives/ray.hpp"
#include "primitives/primitive.hpp"

class AABB {
    Vector3 min; // minimum corner
    Vector3 max; // maximum corner
    std::vector<Primitive*> primitives;
public:
    AABB(const Vector3& min_, const Vector3& max_, const std::vector<Primitive*>& objs)
        : min(min_), max(max_), primitives(objs) {}
    AABB(const Vector3& min_, const Vector3& max_) : min(min_), max(max_) {}

    AABB(const AABB&) = default;
    AABB& operator=(const AABB&) = default;
    AABB(AABB&&) noexcept = default;
    AABB& operator=(AABB&&) noexcept = default;

    const Vector3& getMin() const { return min; }
    const Vector3& getMax() const { return max; }

    const std::vector<Primitive*>& getPrimitives() const { return primitives; }
    std::vector<Primitive*>& getPrimitives() { return primitives; }

    bool hit(const Ray& ray) const {
        double tx1 = (min.getX() - ray.getOrigin().getX()) * ray.getInvDirection().getX();
        double tx2 = (max.getX() - ray.getOrigin().getX()) * ray.getInvDirection().getX();
        double tmin = std::min(tx1, tx2);
        double tmax = std::max(tx1, tx2);

        double ty1 = (min.getY() - ray.getOrigin().getY()) * ray.getInvDirection().getY();
        double ty2 = (max.getY() - ray.getOrigin().getY()) * ray.getInvDirection().getY();
        tmin = std::max(tmin, std::min(ty1, ty2));
        tmax = std::min(tmax, std::max(ty1, ty2));

        double tz1 = (min.getZ() - ray.getOrigin().getZ()) * ray.getInvDirection().getZ();
        double tz2 = (max.getZ() - ray.getOrigin().getZ()) * ray.getInvDirection().getZ();
        tmin = std::max(tmin, std::min(tz1, tz2));
        tmax = std::min(tmax, std::max(tz1, tz2));

        return tmax >= tmin;
    }
};


#endif //RAYTRACINGDEMO_AABB_HPP
