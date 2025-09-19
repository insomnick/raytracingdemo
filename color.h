#ifndef RAYTRACINGDEMO_COLOR_H
#define RAYTRACINGDEMO_COLOR_H

#include <vector>
#include "vector3.h"

struct Color {
    Vector3 c;
    Color(): c({0.0, 0.0, 0.0}) {}
    Color(double r, double g, double b) : c({r, g, b}) {}
    double r() { return c.getX(); }
    double g() { return c.getY(); }
    double b() { return c.getZ(); }
};

#endif //RAYTRACINGDEMO_COLOR_H
