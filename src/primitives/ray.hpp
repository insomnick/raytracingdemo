#ifndef RAYTRACINGDEMO_RAY_HPP
#define RAYTRACINGDEMO_RAY_HPP

#include <limits>
#include "vector3.hpp"

struct Ray {
private:
    Vector3 origin;
    Vector3 direction;
    Vector3 inv_direction; // Precomputed inverse direction for AABB intersection
public:
    Ray(Vector3 origin, Vector3 direction) : origin(origin), direction(direction) {
        inv_direction = Vector3{
            direction.getX() != 0.0 ? 1.0 / direction.getX() : std::numeric_limits<double>::infinity(),
            direction.getY() != 0.0 ? 1.0 / direction.getY() : std::numeric_limits<double>::infinity(),
            direction.getZ() != 0.0 ? 1.0 / direction.getZ() : std::numeric_limits<double>::infinity()
        };
    }
    Vector3 getOrigin() const { return origin; }
    Vector3 getDirection() const { return direction; }
    Vector3 getInvDirection() const { return inv_direction; }
};

#endif //RAYTRACINGDEMO_RAY_HPP
