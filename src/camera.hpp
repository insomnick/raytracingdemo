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
        direction = {0.0, 0.0, -1.0};
        position = {0.0, 0.0, 0.0};
        //plane = {0.0, 1.0, 1.0};
        fov = 90.0 * (std::numbers::pi / 180.0); //Convert to Radians
    }

    void setPosition(const Vector3& pos) {
        position = pos;
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
        double sin_angle = std::sin(angle);
        double new_x = direction.getX() * cos_angle - direction.getZ() * sin_angle;
        double new_z = direction.getX() * sin_angle + direction.getZ() * cos_angle;
        direction = {new_x, direction.getY(), new_z};
        //printf("Camera length: %f\n", direction.length());
    }

    void rotateHorizontal(double angle) {
        // Rotate relative to current direction's right vector, also considering not rotating the wrong direction relative to y axis
        Vector3 world_up = {0.0, 1.0, 0.0};
        Vector3 right = Vector3::cross(direction, world_up).normalize();
        Vector3 up = Vector3::cross(right, direction).normalize();
        double cos_angle = std::cos(angle);
        double sin_angle = std::sin(angle);
        double new_x = direction.getX() * cos_angle + up.getX() * sin_angle;
        double new_y = direction.getY() * cos_angle + up.getY() * sin_angle;
        double new_z = direction.getZ() * cos_angle + up.getZ() * sin_angle;
        direction = {new_x, new_y, new_z};
        //printf("Camera length: %f\n", direction.length());
    }

    void moveForward(double distance) {
        Vector3 norm_dir = direction.normalize();
        position = position + (norm_dir * distance);
    }

    void moveRight(double distance) {
        Vector3 world_up = {0.0, 1.0, 0.0};
        Vector3 right = Vector3::cross(direction, world_up).normalize();
        position = position + (right * distance);
    }

    void moveLeft(double distance) {
        Vector3 world_up = {0.0, 1.0, 0.0};
        Vector3 right = Vector3::cross(direction, world_up).normalize();
        position = position - (right * distance);
    }

    void moveUp(double d) {
        Vector3 world_up = {0.0, 1.0, 0.0};
        position = position + (world_up * d);
    }
};

#endif //RAYTRACINGDEMO_CAMERA_HPP
