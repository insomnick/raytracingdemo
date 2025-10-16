#ifndef RAYTRACINGDEMO_COLOR_HPP
#define RAYTRACINGDEMO_COLOR_HPP

#include <vector>
#include "primitives/vector3.hpp"

struct Color {
    Vector3 c;
    Color(): c({0.f, 0.f, 0.f}) {}
    Color(float r, float g, float b) : c({r, g, b}) {}
    float r() const { return c.getX(); }
    float g() const { return c.getY(); }
    float b() const { return c.getZ(); }
};

#endif //RAYTRACINGDEMO_COLOR_HPP
