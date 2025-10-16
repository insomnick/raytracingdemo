#ifndef RAYTRACINGDEMO_SPHERE_HPP
#define RAYTRACINGDEMO_SPHERE_HPP

#include <memory>
#include "vector3.hpp"
#include "ray.hpp"
#include "primitive.hpp"

class Sphere : public Primitive {
private:
    Vector3 center;
    double radius;
public:
    Sphere() : center({0.0,0.0,0.0}), radius(1.0) {}
    Sphere(double cx, double cy, double cz, double r) : center({cx,cy,cz}), radius(r) {}
    Vector3 getCenter() const override { return center; }
    //double getRadius() const { return radius; }

    Vector3 getMin() const override{ return center - Vector3(1,1,1) * radius; }
    Vector3 getMax() const override{ return center + Vector3(1,1,1) * radius; }

    bool intersect(const Ray& ray) const override {
        Vector3 oc = (ray.getOrigin() - center);
        double a = Vector3::dot(ray.getDirection(), ray.getDirection());
        double b = 2.0 * Vector3::dot(oc, ray.getDirection());
        double c = Vector3::dot(oc, oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0.0) {
            return false;
        }
        double t = (-b - std::sqrt(discriminant)) / (2.0 * a);
        if(t <= 0.0){
            return false;
        }
        return true;
    }

    std::unique_ptr<Ray> getIntersectionNormalAndDirection(const Ray& ray) const override {
        Vector3 oc = (ray.getOrigin() - center);
        double a = Vector3::dot(ray.getDirection(), ray.getDirection());
        double b = 2.0 * Vector3::dot(oc, ray.getDirection());
        double c = Vector3::dot(oc, oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0.0) {
            return nullptr;
        }
        double t = (-b - std::sqrt(discriminant)) / (2.0 * a);
        if(t <= 0.0){
            return nullptr;
        }
        const auto intersection = (ray.getDirection() * t);
        return std::make_unique<Ray>(intersection, ((ray.getOrigin() + intersection) - center).normalize());
    }

    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Sphere>(*this);
    }
    ~Sphere() override = default;
};

#endif //RAYTRACINGDEMO_SPHERE_HPP
