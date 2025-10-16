#ifndef RAYTRACINGDEMO_PRIMITIVE_HPP
#define RAYTRACINGDEMO_PRIMITIVE_HPP

#include <memory>
#include "vector3.hpp"
#include "ray.hpp"

class Primitive {
public:
    virtual ~Primitive() = default;

    virtual Vector3 getCenter() const = 0;

    virtual Vector3 getMin() const = 0;
    virtual Vector3 getMax() const = 0;

    // returns true if hit; outputs t (distance along ray) and outward normal
    virtual bool intersect(const Ray& ray, float& tOut, Vector3& normalOut) const = 0;

    virtual std::unique_ptr<Primitive> clone() const = 0;
};

#endif //RAYTRACINGDEMO_PRIMITIVE_HPP
