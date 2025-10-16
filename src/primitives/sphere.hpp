#ifndef RAYTRACINGDEMO_SPHERE_HPP
#define RAYTRACINGDEMO_SPHERE_HPP

#include <memory>
#include "vector3.hpp"
#include "ray.hpp"
#include "primitive.hpp"

class Sphere : public Primitive {
    Vector3 center; float radius;
public:
    Sphere(): center(0,0,0), radius(1.f) {}
    Sphere(float cx,float cy,float cz,float r): center(cx,cy,cz), radius(r) {}
    Vector3 getCenter() const override { return center; }
    Vector3 getMin() const override { return center - Vector3(radius,radius,radius); }
    Vector3 getMax() const override { return center + Vector3(radius,radius,radius); }
    bool intersect(const Ray& ray, float& tOut, Vector3& normalOut) const override {
        Vector3 oc = ray.origin - center;
        float a = Vector3::dot(ray.direction, ray.direction); // ~1 if normalized
        float b = 2.f * Vector3::dot(oc, ray.direction);
        float c = Vector3::dot(oc, oc) - radius*radius;
        float disc = b*b - 4*a*c;
        if (disc < 0.f) return false;
        float s = std::sqrt(disc);
        float t = (-b - s) / (2.f*a);
        if (t <= 0.f) { t = (-b + s)/(2.f*a); if (t <= 0.f) return false; }
        tOut = t;
        Vector3 hit = ray.origin + ray.direction * t;
        normalOut = (hit - center).normalize();
        if (Vector3::dot(normalOut, ray.direction) > 0) normalOut = normalOut * -1.f;
        return true;
    }
    std::unique_ptr<Primitive> clone() const override { return std::make_unique<Sphere>(*this); }
};

#endif //RAYTRACINGDEMO_SPHERE_HPP
