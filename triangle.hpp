#ifndef RAYTRACINGDEMO_TRIANGLE_H
#define RAYTRACINGDEMO_TRIANGLE_H

#include "vector3.hpp"

class Triangle {
private:
    Vector3 v0, v1, v2; // Triangle vertices
    Vector3 normal; // Precomputed normal for the triangle
public:
    Triangle(const Vector3& v0, const Vector3& v1, const Vector3& v2)
        : v0(v0), v1(v1), v2(v2) {
        // Precompute the normal
        normal = Vector3::cross(v1 - v0, v2 - v0).normalize();
        printf("Triangle vertices: (%f, %f, %f), (%f, %f, %f), (%f, %f, %f)\n", v0.getX(), v0.getY(), v0.getZ(), v1.getX(), v1.getY(), v1.getZ(), v2.getX(), v2.getY(), v2.getZ());
    }
};


#endif //RAYTRACINGDEMO_TRIANGLE_H
