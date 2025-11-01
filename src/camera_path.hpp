#ifndef RAYTRACINGDEMO_CAMERA_PATH_HPP
#define RAYTRACINGDEMO_CAMERA_PATH_HPP

#include <cmath>
#include "primitives/ray.hpp"
#include "primitives/vector3.hpp"

class CameraPath {
private:
    Vector3 circular_path_start = Vector3{0.0, 0.0, 5.0};
    Vector3 center;
    int resolution;
public:
    CameraPath(Vector3 center, int res) : center(center), resolution(res) {
    }
    //all Paths are in the range [0,1] scaled to resolution

    Ray circularPath(int step) const {
        const double angle = 2.0 * M_PI * (static_cast<double>(step % resolution) / resolution);
        const double x = circular_path_start.getX() * std::cos(angle) - circular_path_start.getZ() * std::sin(angle);
        const double z = circular_path_start.getX() * std::sin(angle) + circular_path_start.getZ() * std::cos(angle);
        const Vector3 position = center + Vector3{x, circular_path_start.getY(), z};

        const Vector3 direction = (center - position).normalize();
        return Ray{position, direction};
    }
};

#endif //RAYTRACINGDEMO_CAMERA_PATH_HPP
