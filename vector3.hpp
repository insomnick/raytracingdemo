//
// Created by garczorz on 9/19/25.
//

#ifndef RAYTRACINGDEMO_VECTOR3_HPP
#define RAYTRACINGDEMO_VECTOR3_HPP


#include <complex>

class Vector3 {
private:
    double x, y, z;
public:
    explicit Vector3() : Vector3 (0.0,0.0,0.0){}
    Vector3(const Vector3& v) : Vector3(v.x, v.y, v.z) {}
    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
    double getX() const { return x; }
    double getY() const { return y; }
    double getZ() const { return z; }

    friend Vector3 operator+(const Vector3& v1, const Vector3& v2)  {
        return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
    }

    friend Vector3 operator+(const Vector3& v1, Vector3&& v2)  {    //For performance (move semantics)
        v2.x += v1.x;
        v2.y += v1.y;
        v2.z += v1.z;
        return v2; //std::move(v2); //Normally move is necessary
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

    auto length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    friend bool operator==(const Vector3& v1, const Vector3& v2) {
        return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
    }

//    bool operator==(const Vector3& rhs) const {
//        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
//    }
};


#endif //RAYTRACINGDEMO_VECTOR3_HPP
