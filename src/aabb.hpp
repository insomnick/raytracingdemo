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

public:
    const Vector3& getMin() const { return min; }
    const Vector3& getMax() const { return max; }

    int getLongestAxis() const {
        auto lengths =  max - min;
        if (lengths.getX() >= lengths.getY() && lengths.getX() >= lengths.getZ()) {
            return 0;
        } else if (lengths.getY() >= lengths.getX() && lengths.getY() >= lengths.getZ()) {
            return 1;
        } else {
            return 2;
        }
    }

    AABB(const std::vector<Primitive *> &primitives, size_t i, size_t i1) {
        if (primitives.empty()) {
            min = Vector3{0.0, 0.0, 0.0};
            max = Vector3{0.0, 0.0, 0.0};
            return;
        }
        min = Vector3{
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()
        };
        max = Vector3{
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity(),
                -std::numeric_limits<double>::infinity()
        };

        for (const auto &p : primitives) {
            Vector3 p_min = p->getMin();
            Vector3 p_max = p->getMax();
            min = Vector3{
                    std::min(min.getX(), p_min.getX()),
                    std::min(min.getY(), p_min.getY()),
                    std::min(min.getZ(), p_min.getZ())
            };
            max = Vector3{
                    std::max(max.getX(), p_max.getX()),
                    std::max(max.getY(), p_max.getY()),
                    std::max(max.getZ(), p_max.getZ())
            };
        }
    }
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
