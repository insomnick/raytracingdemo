#ifndef RAYTRACINGDEMO_SPHERE_HPP
#define RAYTRACINGDEMO_SPHERE_HPP

#include <memory>
#include "vector3.hpp"
#include "ray.hpp"

class Sphere {
private:
    Vector3 center;
    double radius;
public:
    Sphere() : center({0.0,0.0,0.0}), radius(1.0) {}
    Sphere(double cx, double cy, double cz, double r) : center({cx,cy,cz}), radius(r) {}
    Vector3 getCenter() const { return center; }
    double getRadius() const { return radius; }

    bool intersect(const Ray& ray) const {
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

    std::unique_ptr<Vector3> getIntersectionPoint(const Ray& ray) const {
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
        return std::make_unique<Vector3>(ray.getOrigin() + (ray.getDirection() * t));    }
};

#endif //RAYTRACINGDEMO_SPHERE_HPP
