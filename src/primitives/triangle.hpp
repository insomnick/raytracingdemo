#ifndef RAYTRACINGDEMO_TRIANGLE_H
#define RAYTRACINGDEMO_TRIANGLE_H

#include "vector3.hpp"
#include "ray.hpp"

class Triangle : public Primitive {
private:
    Vector3 v0, v1, v2; // Triangle vertices
    Vector3 normal; // Precomputed normal for the triangle
    Vector3 center;
public:
    Triangle(const Vector3& v0, const Vector3& v1, const Vector3& v2)
        : v0(v0), v1(v1), v2(v2) {
        // Precompute the normal and center
        normal = Vector3::cross(v1 - v0, v2 - v0).normalize();
        center = (v0 + v1 + v2) * (1.0/3);
        printf("Triangle vertices: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", v0.getX(), v0.getY(), v0.getZ(), v1.getX(), v1.getY(), v1.getZ(), v2.getX(), v2.getY(), v2.getZ());
    }

    Vector3 getCenter() const override{
        return center;
    }

    Vector3 getMin() const override {
        return Vector3{
            std::min({v0.getX(), v1.getX(), v2.getX()}),
            std::min({v0.getY(), v1.getY(), v2.getY()}),
            std::min({v0.getZ(), v1.getZ(), v2.getZ()})
        };
    }
    Vector3 getMax() const override {
        return Vector3{
            std::max({v0.getX(), v1.getX(), v2.getX()}),
            std::max({v0.getY(), v1.getY(), v2.getY()}),
            std::max({v0.getZ(), v1.getZ(), v2.getZ()})
        };
    }

    bool intersect(const Ray& ray) const override{
        const double EPSILON = 1e-8;
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        Vector3 h = Vector3::cross(ray.getDirection(), edge2);
        double a = Vector3::dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return false; // Ray is parallel to triangle
        double f = 1.0 / a;
        Vector3 s = ray.getOrigin() - v0;
        double u = f * Vector3::dot(s, h);
        if (u < 0.0 || u > 1.0)
            return false;
        Vector3 q = Vector3::cross(s, edge1);
        double v = f * Vector3::dot(ray.getDirection(), q);
        if (v < 0.0 || u + v > 1.0)
            return false;
        double t = f * Vector3::dot(edge2, q);
        if (t > EPSILON) // Intersection
            return true;
        else // Line intersection but not a ray intersection
            return false;
    }

    std::unique_ptr<Ray> getIntersectionNormalAndDirection(const Ray& ray) const override{
        const double EPSILON = 1e-8;
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        Vector3 h = Vector3::cross(ray.getDirection(), edge2);
        double a = Vector3::dot(edge1, h);
        if (a > -EPSILON && a < EPSILON)
            return nullptr; // Ray is parallel to triangle
        double f = 1.0 / a;
        Vector3 s = ray.getOrigin() - v0;
        double u = f * Vector3::dot(s, h);
        if (u < 0.0 || u > 1.0)
            return nullptr;
        Vector3 q = Vector3::cross(s, edge1);
        double v = f * Vector3::dot(ray.getDirection(), q);
        if (v < 0.0 || u + v > 1.0)
            return nullptr;
        double t = f * Vector3::dot(edge2, q);
        if (t <= EPSILON)
            return nullptr;

        const auto intersection = (ray.getDirection() * t);

        return std::make_unique<Ray>(intersection, normal);
    }
    std::unique_ptr<Primitive> clone() const override {
        return std::make_unique<Triangle>(*this);
    }
    ~Triangle() override = default;
};


#endif //RAYTRACINGDEMO_TRIANGLE_H
