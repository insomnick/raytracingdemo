#ifndef RAYTRACINGDEMO_CAMERA_HPP
#define RAYTRACINGDEMO_CAMERA_HPP

#include <numbers>
#include <vector>


class Camera {
private:
    Vector3 direction;
    Vector3 position;
    Vector3 plane;

public:
    Camera() {
        direction = {1.0, 0.0, 0.0};
        position = {-1.0, 0.0, 0.0};
        plane = {0.0, 1.0, 1.0};
    }

    Vector3 getDirection() {
        return direction;
    }

    Vector3 getPosition() {
        return position;
    }

    Vector3 getPlane() {
        return plane;
    }

};

#endif //RAYTRACINGDEMO_CAMERA_HPP
