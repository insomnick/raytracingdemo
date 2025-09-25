#ifndef RAYTRACINGDEMO_COLOR_HPP
#define RAYTRACINGDEMO_COLOR_HPP

#include <vector>
#include "vector3.hpp"

struct Color {
    Vector3 c;
    Color(): c({0.0, 0.0, 0.0}) {}
    Color(double r, double g, double b) : c({r, g, b}) {}
    double r() { return c.getX(); }
    double g() { return c.getY(); }
    double b() { return c.getZ(); }
};

#endif //RAYTRACINGDEMO_COLOR_HPP
