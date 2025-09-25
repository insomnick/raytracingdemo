#ifndef RAYTRACINGDEMO_CAMERA_HPP
#define RAYTRACINGDEMO_CAMERA_HPP

#include <numbers>
#include <vector>


class Camera {
private:
    Vector3 direction;
    Vector3 position;
    //Vector3 plane;
    double fov;

public:
    Camera() {
        direction = {1, 0.0, 0.0};
        position = {-1.0, 0.0, 0.0};
        //plane = {0.0, 1.0, 1.0};
        fov = 90.0 * (std::numbers::pi / 180.0); //Convert to Radians
    }

    Vector3 getDirection() const {
        return direction;
    }

    Vector3 getPosition() const{
        return position;
    }

    double getFov() const {
        return fov;
    }

    void rotateVertical(double angle) {
        // Rotate around Y axis for simplicity
        double cos_angle = std::cos(angle);
        printf("cos_angle: %f \n", cos_angle);
        double sin_angle = std::sin(angle);
        printf("sin_angle: %f \n", sin_angle);
        double new_x = direction.getX() * cos_angle - direction.getZ() * sin_angle;
        double new_z = direction.getX() * sin_angle + direction.getZ() * cos_angle;
        direction = {new_x, direction.getY(), new_z};
        //printf("Camera length: %f\n", direction.length());
    }

    void rotateHorizontal(double angle) {
        // Rotate relative to current direction's right vector
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);
        double new_y = direction.getY() * cos_angle - direction.getZ() * sin_angle;
        double new_z = direction.getY() * sin_angle + direction.getZ() * cos_angle;
        direction = {direction.getX(), new_y, new_z};
        //printf("Camera length: %f\n", direction.length());
    }

};

#endif //RAYTRACINGDEMO_CAMERA_HPP
