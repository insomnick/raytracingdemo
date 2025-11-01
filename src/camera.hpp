#ifndef RAYTRACINGDEMO_CAMERA_HPP
#define RAYTRACINGDEMO_CAMERA_HPP

#include <numbers>
#include <vector>
#include <cmath>
#include "primitives/vector3.hpp"


class Camera {
private:
    Vector3 direction;
    Vector3 position;
    //Vector3 plane;
    double fov;
    std::vector<double> pixel_x_cache;
    std::vector<double> pixel_y_cache;

public:
    Camera(unsigned int screen_width, unsigned int screen_height) {
        direction = {0.0, 0.0, -1.0};
        position = {0.0, 0.0, 0.0};
        //plane = {0.0, 1.0, 1.0};
        fov = 90.0 * (std::numbers::pi / 180.0); //Convert to Radians

        // Precompute pixel plane coefficients - this actually only has to be done one time if camera and screen size don't change
        const double fov_half_tan = std::tan(fov * 0.5);
        const double aspect_ratio = static_cast<double>(screen_width) / screen_height;

        pixel_x_cache.resize(screen_width);
        pixel_y_cache.resize(screen_height);
        const double inv_w = 1.0 / screen_width;
        const double inv_h = 1.0 / screen_height;
        for (int x = 0; x < screen_width; ++x)
            pixel_x_cache[x] = (2.0 * (x + 0.5) * inv_w - 1.0) * fov_half_tan * aspect_ratio;
        for (int y = 0; y < screen_height; ++y)
            pixel_y_cache[y] = (1.0 - 2.0 * (y + 0.5) * inv_h) * fov_half_tan;
    }

    void setPosition(const Vector3& pos) {
        position = pos;
    }

    void setDirection(const Vector3& dir) {
        direction = dir;
    }

    Vector3 getDirection() const {
        return direction;
    }

    Vector3 getPosition() const{
        return position;
    }

    double getPixelX(int i) const {
        return pixel_x_cache[i];
    }

    double getPixelY(int i) const {
        return pixel_y_cache[i];
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
