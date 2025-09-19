#ifndef RAYTRACINGDEMO_SPHERE_H
#define RAYTRACINGDEMO_SPHERE_H

#include "vector3.h"

class Sphere {
private:
    Vector3 center;
    double radius;
public:
    Sphere() : center({0.0,0.0,0.0}), radius(1.0) {}
    Sphere(double cx, double cy, double cz, double r) : center({cx,cy,cz}), radius(r) {}
    Vector3 getCenter() { return center; }
    double getRadius() { return radius; }

    Vector3 intersect(const Vector3& ray_origin, const Vector3& ray_direction) {
        Vector3 oc = Vector3::subtract(ray_origin, center);
        double a = Vector3::dot(ray_direction, ray_direction);
        double b = 2.0 * Vector3::dot(oc, ray_direction);
        double c = Vector3::dot(oc, oc) - radius * radius;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0) {
            return {0.0, 0.0, 0.0}; // No intersection
        } else {
            double t = (-b - std::sqrt(discriminant)) / (2.0 * a);
            return Vector3::add(ray_origin, Vector3::multiply(ray_direction, t));
        }
    }

};

#endif //RAYTRACINGDEMO_SPHERE_H
