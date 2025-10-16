#ifndef RAYTRACINGDEMO_PRIMITIVE_HPP
#define RAYTRACINGDEMO_PRIMITIVE_HPP

#include <memory>
#include "vector3.hpp"
#include "ray.hpp"

class Primitive {
public:
    virtual Vector3 getCenter() const = 0;

    virtual Vector3 getMin() const = 0;
    virtual Vector3 getMax() const = 0;


    virtual ~Primitive() = default;

    virtual bool intersect(const Ray &ray) const = 0;

    virtual std::unique_ptr<Ray> getIntersectionNormalAndDirection(const Ray &ray) const = 0;

    virtual std::unique_ptr<Primitive> clone() const = 0;
};

#endif //RAYTRACINGDEMO_PRIMITIVE_HPP
