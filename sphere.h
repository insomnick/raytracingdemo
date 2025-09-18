#ifndef RAYTRACINGDEMO_SPHERE_H
#define RAYTRACINGDEMO_SPHERE_H

#include <vector>

class Sphere {
private:
    std::vector<double> center;
    double radius;
public:
    Sphere() : center({0.0,0.0,0.0}), radius(1.0) {}
    Sphere(double cx, double cy, double cz, double r) : center({cx,cy,cz}), radius(r) {}
    std::vector<double> getCenter() { return center; }
    double getRadius() { return radius; }
};

#endif //RAYTRACINGDEMO_SPHERE_H
