#ifndef RAYTRACINGDEMO_BVH_HPP
#define RAYTRACINGDEMO_BVH_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include "aabb.hpp"
#include "sphere.hpp"

class BVH {
private:
    // Bounding Volume Hierarchy class for acceleration structure
    AABB box;
    BVH* left;
    BVH* right;
    BVH(AABB box, BVH* left, BVH* right): box(std::move(box)), left(left), right(right) {}

public:

    static BVH stupidConstruct(std::vector<Sphere>& objects) {
        AABB box{
            {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
            {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()},
            objects
        };
        BVH bvh{box, nullptr, nullptr};
        return bvh;
    }

    //Split the objects based on median in one axis after the other
    static BVH medianSplitConstruct(std::vector<Sphere>& objects) {
        //TODO: not implemented yet
    }

    std::unique_ptr<Vector3> traverse(const Ray& ray) const {
        if (!box.hit(ray)) {
            return nullptr; // No intersection with bounding box
        }
        if (box.containsPrimitives()) {
            // Check intersection with all primitives in this leaf node
            std::unique_ptr<Vector3> closest_intersection = nullptr;
            double closest_distance = std::numeric_limits<double>::max();
            for (const auto& object : box.getPrimitives()) {
                auto intersection = object.intersect(ray);
                if (intersection != nullptr) {
                    double distance = (*intersection - ray.getOrigin()).length();
                    if (distance < closest_distance) {
                        closest_distance = distance;
                        closest_intersection = std::move(intersection);
                    }
                }
            }
            return closest_intersection;
        } else {
            // Recur for child nodes
            std::unique_ptr<Vector3> left_intersection = left ? left->traverse(ray) : nullptr;
            std::unique_ptr<Vector3> right_intersection = right ? right->traverse(ray) : nullptr;

            if (left_intersection && right_intersection) {
                double left_distance = (*left_intersection - ray.getOrigin()).length();
                double right_distance = (*right_intersection - ray.getOrigin()).length();
                return (left_distance < right_distance) ? std::move(left_intersection) : std::move(right_intersection);
            } else if (left_intersection) {
                return left_intersection;
            } else {
                return right_intersection;
            }
        }
    }
};

#endif //RAYTRACINGDEMO_BVH_HPP
