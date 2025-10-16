#ifndef RAYTRACINGDEMO_TRIANGLE_H
#define RAYTRACINGDEMO_TRIANGLE_H

#include "vector3.hpp"
#include "ray.hpp"
#include "primitive.hpp"

class Triangle : public Primitive {
    Vector3 v0,v1,v2; Vector3 e1,e2; Vector3 normal; Vector3 center;
public:
    Triangle(const Vector3& a,const Vector3& b,const Vector3& c): v0(a),v1(b),v2(c){
        e1 = v1 - v0; e2 = v2 - v0; normal = Vector3::cross(e1, e2).normalize(); center = (v0+v1+v2)*(1.f/3.f);
    }
    Vector3 getCenter() const override { return center; }
    Vector3 getMin() const override { return { std::min({v0.getX(),v1.getX(),v2.getX()}), std::min({v0.getY(),v1.getY(),v2.getY()}), std::min({v0.getZ(),v1.getZ(),v2.getZ()})}; }
    Vector3 getMax() const override { return { std::max({v0.getX(),v1.getX(),v2.getX()}), std::max({v0.getY(),v1.getY(),v2.getY()}), std::max({v0.getZ(),v1.getZ(),v2.getZ()})}; }
    bool intersect(const Ray& ray, float& tOut, Vector3& normalOut) const override {
        const float EPS = 1e-6f;
        Vector3 p = Vector3::cross(ray.direction, e2);
        float det = Vector3::dot(e1, p);
        if (det > -EPS && det < EPS) return false;
        float invDet = 1.f / det;
        Vector3 s = ray.origin - v0;
        float u = Vector3::dot(s, p) * invDet;
        if (u < 0.f || u > 1.f) return false;
        Vector3 q = Vector3::cross(s, e1);
        float v = Vector3::dot(ray.direction, q) * invDet;
        if (v < 0.f || u + v > 1.f) return false;
        float t = Vector3::dot(e2, q) * invDet;
        if (t <= EPS) return false;
        tOut = t;
        normalOut = normal;
        if (Vector3::dot(normalOut, ray.direction) > 0) normalOut = normalOut * -1.f;
        return true;
    }
    std::unique_ptr<Primitive> clone() const override { return std::make_unique<Triangle>(*this); }
};

#endif //RAYTRACINGDEMO_TRIANGLE_H
