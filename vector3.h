//
// Created by garczorz on 9/19/25.
//

#ifndef RAYTRACINGDEMO_VECTOR3_H
#define RAYTRACINGDEMO_VECTOR3_H


#include <complex>

class Vector3 {
private:
    double x, y, z;
public:
    Vector3() : x(0.0), y(0.0), z(0.0) {}
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    static Vector3 add(const Vector3& v1, const Vector3& v2)  {
        return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
    }

    static Vector3 subtract(const Vector3& v1, const Vector3& v2)  {
        return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
    }

    static Vector3 multiply(const Vector3& v, double scalar)  {
        return {v.x * scalar, v.y * scalar, v.z * scalar};
    }

    static double dot(const Vector3& v1, const Vector3& v2)  {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    static double length(const Vector3& v)  {
        return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    static bool equals(const Vector3& v1, const Vector3& v2) {
        return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
    }
};


#endif //RAYTRACINGDEMO_VECTOR3_H
