#ifndef RAYTRACINGDEMO_BVH_HPP
#define RAYTRACINGDEMO_BVH_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include "aabb.hpp"
#include "sphere.hpp"

class BVH {
private:
    AABB box;
    std::vector<BVH> children;
    BVH(AABB box, std::vector<BVH> children): box(std::move(box)), children(std::move(children)) {}

    // Private helper function for recursive construction
    static BVH medianSplitConstruction(std::vector<Sphere>& objects, size_t start, size_t end, int degree) {

        //Calculate axis aligned bounding box
        const size_t count = end - start;
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double minZ = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double maxY = std::numeric_limits<double>::lowest();
        double maxZ = std::numeric_limits<double>::lowest();
        for (size_t i = start; i < end; ++i) {
            const auto& object = objects[i];
            const auto center = object.getCenter();
            const double radius = object.getRadius();
            minX = std::min(minX, center.getX() - radius);
            maxX = std::max(maxX, center.getX() + radius);
            minY = std::min(minY, center.getY() - radius);
            maxY = std::max(maxY, center.getY() + radius);
            minZ = std::min(minZ, center.getZ() - radius);
            maxZ = std::max(maxZ, center.getZ() + radius);
        }

        // Leaf node condition
        if (count <= static_cast<size_t>(degree)) {
            std::vector<Sphere> primitiveLeaves;
            primitiveLeaves.reserve(count);
            for (size_t i = start; i < end; ++i) {
                primitiveLeaves.push_back(objects[i]);
            }
            AABB leafBox{{minX, minY, minZ}, {maxX, maxY, maxZ}, primitiveLeaves};
            return BVH{leafBox, {}};
        }

        // Longest axis
        const double lenX = maxX - minX;
        const double lenY = maxY - minY;
        const double lenZ = maxZ - minZ;
        int axis = 0;
        if (lenY > lenX && lenY >= lenZ) axis = 1;
        else if (lenZ > lenX && lenZ >= lenY) axis = 2;

        //TODO: For now binary split regardless of degree var
        size_t mid = start + count / 2;
        // Partition around median (nth_element is faster than sort)
        std::nth_element(objects.begin() + start,
                         objects.begin() + mid,
                         objects.begin() + end,
                         [&](const Sphere& a, const Sphere& b) {
                             return a.getCenter().getAxis(axis) < b.getCenter().getAxis(axis);
                         });

        // Create Node
        AABB internalBox{{minX, minY, minZ}, {maxX, maxY, maxZ}, {}};
        std::vector<BVH> kids;
        kids.reserve(2);    //TODO: Change for degree > 2
        kids.emplace_back(medianSplitConstruction(objects, start, mid, degree));
        kids.emplace_back(medianSplitConstruction(objects, mid, end, degree));
        return BVH{internalBox, std::move(kids)};
    }

public:
    static BVH stupidConstruct(std::vector<Sphere>& objects) {
        AABB box{
            {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
            {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()},
            objects
        };
        std::vector<BVH> children;
        BVH bvh{box, children};
        return bvh;
    }

    static BVH medianSplitConstruction(std::vector<Sphere>& objects, int degree = 2) {
        if (objects.empty()) {
            AABB emptyBox{
                {0,0,0},
                {0,0,0},
                {}
            };
            return BVH{emptyBox, {}};
        }
        return medianSplitConstruction(objects, 0, objects.size(), degree);
    }

    std::unique_ptr<Vector3> traverse(const Ray& ray) const {
        if (!box.hit(ray)){
            return nullptr;
        }
        if (children.empty()) { //or box.getPrimitives().empty()
            std::unique_ptr<Vector3> closest = nullptr;
            double closestDist = std::numeric_limits<double>::max();
            for (const auto &object: box.getPrimitives()) {
                if (object.intersect(ray)) {
                    auto p = object.getIntersectionPoint(ray);
                    const double d = (*p - ray.getOrigin()).length();
                    if (d < closestDist) {
                        closestDist = d;
                        closest = std::move(p);
                    }
                }
            }
            return closest;
        }
        // Traverse children recursively
        std::unique_ptr<Vector3> closest = nullptr;
        double closestDistance = std::numeric_limits<double>::max();
        for (const auto& child : children) {
            auto hit = child.traverse(ray);
            if (hit) {
                const double distance = (*hit - ray.getOrigin()).length();
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closest = std::move(hit);
                }
            }
        }
        return closest;
    }
};

#endif //RAYTRACINGDEMO_BVH_HPP

