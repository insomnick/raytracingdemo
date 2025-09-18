#ifndef RAYTRACINGDEMO_COLOR_H
#define RAYTRACINGDEMO_COLOR_H

#include <vector>

struct Color {
    std::vector<double> v;
    Color(): v({0.0,0.0,0.0}) {}
    Color(double r, double g, double b) : v({r,g,b}) {}
    double r() { return v[0]; }
    double g() { return v[1]; }
    double b() { return v[2]; }
};


#endif //RAYTRACINGDEMO_COLOR_H
