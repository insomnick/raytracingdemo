#ifndef RAYTRACINGDEMO_COLOR_HPP
#define RAYTRACINGDEMO_COLOR_HPP

#include <vector>
#include "primitives/vector3.hpp"

struct Color {
    Vector3 c;
    Color(): c({0.0, 0.0, 0.0}) {}
    Color(double r, double g, double b) : c({r, g, b}) {}
    double r() const { return c.getX(); }
    double g() const { return c.getY(); }
    double b() const { return c.getZ(); }
};

#endif //RAYTRACINGDEMO_COLOR_HPP
