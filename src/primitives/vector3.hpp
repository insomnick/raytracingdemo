//
// Created by garczorz on 9/19/25.
//

#ifndef RAYTRACINGDEMO_VECTOR3_HPP
#define RAYTRACINGDEMO_VECTOR3_HPP

#include <cmath>
#include <stdexcept>

class Vector3 {
    float x, y, z;
public:
    Vector3(): x(0), y(0), z(0) {}
    Vector3(float x_, float y_, float z_): x(x_), y(y_), z(z_) {}
    float getX() const { return x; }
    float getY() const { return y; }
    float getZ() const { return z; }
    float getAxis(int axis) const {
        switch(axis){ case 0: return x; case 1: return y; case 2: return z; default: throw std::out_of_range("axis"); }
    }
    Vector3 operator+(const Vector3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vector3 operator-(const Vector3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vector3 operator*(float s) const { return {x*s, y*s, z*s}; }
    friend Vector3 operator*(float s, const Vector3& v){ return {v.x*s, v.y*s, v.z*s}; }
    Vector3& operator+=(const Vector3& o){ x+=o.x; y+=o.y; z+=o.z; return *this; }
    static float dot(const Vector3& a, const Vector3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
    static Vector3 cross(const Vector3& a, const Vector3& b){ return { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x}; }
    float length2() const { return x*x + y*y + z*z; }
    float length() const { return std::sqrt(length2()); }
    Vector3 normalize() const { float len = length(); if(len==0) return {}; float inv = 1.0f/len; return {x*inv,y*inv,z*inv}; }
};

#endif //RAYTRACINGDEMO_VECTOR3_HPP
