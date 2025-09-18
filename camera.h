#ifndef RAYTRACINGDEMO_CAMERA_H
#define RAYTRACINGDEMO_CAMERA_H

#include <numbers>
#include <vector>


class Camera {
private:
    std::vector<double> direction;
    std::vector<double> position;
    std::vector<double> plane;

public:
    Camera() {
        direction = {1.0, 0.0, 0.0};
        position = {-1.0, 0.0, 0.0};
        plane = {0.0, 0.5, 0.5};
    }

    std::vector<double> getDirection() {
        return direction;
    }

    std::vector<double> getPosition() {
        return position;
    }

    std::vector<double> getPlane() {
        return plane;
    }
};

#endif //RAYTRACINGDEMO_CAMERA_H
