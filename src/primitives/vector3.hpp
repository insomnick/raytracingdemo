#ifndef RAYTRACINGDEMO_VECTOR3_HPP
#define RAYTRACINGDEMO_VECTOR3_HPP

#include <cmath>
#include <stdexcept>

class Vector3 {
private:
    double x, y, z;
public:
    explicit Vector3() : Vector3(0.0, 0.0, 0.0) {}

    Vector3(const Vector3 &v) : Vector3(v.x, v.y, v.z) {}

    Vector3(double x, double y, double z) : x(x), y(y), z(z) {}

    Vector3& operator=(const Vector3& other) {
        if (this != &other) {
            x = other.x;
            y = other.y;
            z = other.z;
        }
        return *this;
    }

    double getX() const { return x; }

    double getY() const { return y; }

    double getZ() const { return z; }

    double getAxis(int axis) const {
        switch(axis) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            default: throw std::out_of_range("Axis must be 0, 1, or 2");
        }
    }

    friend Vector3 operator+(const Vector3 &v1, const Vector3 &v2) {
        return {v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
    }

    friend Vector3 operator+(const Vector3 &v1, Vector3 &&v2) {    //For performance (move semantics)
        v2.x += v1.x;
        v2.y += v1.y;
        v2.z += v1.z;
        return v2; //std::move(v2); //Normally move is necessary
    }

    friend Vector3 operator-(const Vector3 &v1, const Vector3 &v2) {    //For performance (move semantics)
        return {v1.x - v2.x, v1.y - v2.y, v1.z - v2.z};
    }

    friend Vector3 operator-(const Vector3 &v1, Vector3 &&v2) {    //For performance (move semantics)
        v2.x = v1.x - v2.x;
        v2.y = v1.y - v2.y;
        v2.z = v1.z - v2.z;
        return v2; //std::move(v2); //Normally move is necessary
    }

    friend Vector3 operator*(const Vector3 &v, double scalar) {    //For performance (move semantics)
        return {v.x * scalar, v.y * scalar, v.z * scalar};
    }

    static double dot(const Vector3 &v1, const Vector3 &v2) {
        return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    static Vector3 cross(const Vector3 &v1, const Vector3 &v2) {
        return {
                v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x
        };
    }

    auto length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    friend bool operator==(const Vector3 &v1, const Vector3 &v2) {
        return (v1.x == v2.x) && (v1.y == v2.y) && (v1.z == v2.z);
    }

//    bool operator==(const Vector3& rhs) const {
//        return (x == rhs.x) && (y == rhs.y) && (z == rhs.z);
//    }

    auto normalize() const {
        auto len = length();
        if (len == 0) return Vector3(0, 0, 0);
        return Vector3(x / len, y / len, z / len);
    }
};


#endif //RAYTRACINGDEMO_VECTOR3_HPP
