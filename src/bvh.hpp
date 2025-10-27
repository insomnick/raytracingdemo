#ifndef RAYTRACINGDEMO_BVH_HPP
#define RAYTRACINGDEMO_BVH_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstddef>
#include "aabb.hpp"
#include "primitives/primitive.hpp"

class BVH {
private:
    AABB box;
    std::vector<BVH> children;

    BVH(AABB box, std::vector<BVH> children) : box(std::move(box)), children(std::move(children)) {}

    static BVH
    medianSplitConstruction(std::vector<std::unique_ptr<Primitive>> &objects, size_t start, size_t end, int degree) {

        //Calculate axis aligned bounding box
        const size_t count = end - start;
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double minZ = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double maxY = std::numeric_limits<double>::lowest();
        double maxZ = std::numeric_limits<double>::lowest();
        for (size_t i = start; i < end; ++i) {
            const auto &object = objects[i];
            minX = std::min(minX, object->getMin().getX());
            maxX = std::max(maxX, object->getMax().getX());
            minY = std::min(minY, object->getMin().getY());
            maxY = std::max(maxY, object->getMax().getY());
            minZ = std::min(minZ, object->getMin().getZ());
            maxZ = std::max(maxZ, object->getMax().getZ());
        }

        // Leaf node condition
        if (count == 0 || degree <= 1 || count < static_cast<size_t>(degree)) {
            std::vector<std::unique_ptr<Primitive>> primitiveLeaves;
            primitiveLeaves.reserve(count);
            for (size_t i = start; i < end; ++i) {
                primitiveLeaves.push_back(std::move(objects[i]));
            }
            AABB leafBox{{minX, minY, minZ}, {maxX, maxY, maxZ}, std::move(primitiveLeaves)};
            return BVH{leafBox, {}};
        }

        // Longest axis
        const double lenX = maxX - minX;
        const double lenY = maxY - minY;
        const double lenZ = maxZ - minZ;
        int axis = 0;
        if (lenY > lenX && lenY >= lenZ) axis = 1;
        else if (lenZ > lenX && lenZ >= lenY) axis = 2;

        AABB internalBox{{minX, minY, minZ},
                         {maxX, maxY, maxZ},
                         {}};

        size_t segmentBegin = start;
        for (int i = 1; i < degree; ++i) {
            size_t boundary = start + (count * i) / degree; // target index for this partition boundary
            std::nth_element(objects.begin() + segmentBegin,
                             objects.begin() + boundary,
                             objects.begin() + end,
                             [axis](const std::unique_ptr<Primitive> &a, const std::unique_ptr<Primitive> &b) {
                                 return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                             });
            segmentBegin = boundary; // next segment starts from here
        }

        std::vector<BVH> kids;
        kids.reserve(degree);
        size_t childStart = start;
        for (int i = 0; i < degree; ++i) {
            size_t childEnd = (i == degree - 1) ? end : start + (count * (i + 1)) / degree;
            kids.emplace_back(medianSplitConstruction(objects, childStart, childEnd, degree));
            childStart = childEnd;
        }

        return BVH{internalBox, std::move(kids)};
    }

    //TODO no code copying from medianSplitConstruction
    static BVH binarySurfaceAreaHeuristicConstruction(std::vector<std::unique_ptr<Primitive>> &objects, size_t start, size_t end) {
        //Calculate axis aligned bounding box
        const size_t count = end - start;
        double minX = std::numeric_limits<double>::max();
        double minY = std::numeric_limits<double>::max();
        double minZ = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double maxY = std::numeric_limits<double>::lowest();
        double maxZ = std::numeric_limits<double>::lowest();
        for (size_t i = start; i < end; ++i) {
            const auto &object = objects[i];
            minX = std::min(minX, object->getMin().getX());
            maxX = std::max(maxX, object->getMax().getX());
            minY = std::min(minY, object->getMin().getY());
            maxY = std::max(maxY, object->getMax().getY());
            minZ = std::min(minZ, object->getMin().getZ());
            maxZ = std::max(maxZ, object->getMax().getZ());
        }

        // Leaf node condition
        if (count < 2) {
            std::vector<std::unique_ptr<Primitive>> primitiveLeaves;
            primitiveLeaves.reserve(count);
            for (size_t i = start; i < end; ++i) {
                primitiveLeaves.push_back(std::move(objects[i]));
            }
            AABB leafBox{{minX, minY, minZ}, {maxX, maxY, maxZ}, std::move(primitiveLeaves)};
            return BVH{leafBox, {}};
        }

        // Longest axis
        const double lenX = maxX - minX;
        const double lenY = maxY - minY;
        const double lenZ = maxZ - minZ;
        int axis = 0;
        if (lenY > lenX && lenY >= lenZ) axis = 1;
        else if (lenZ > lenX && lenZ >= lenY) axis = 2;

        AABB internalBox{{minX, minY, minZ},
                         {maxX, maxY, maxZ},
                         {}};
        // we need to calculate area heuristic here to find the best split point
        std::sort(objects.begin() + start, objects.begin() + end,
                  [axis](const std::unique_ptr<Primitive>& a, const std::unique_ptr<Primitive>& b){
                      return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                  });
        size_t bestSplit = start + 1;;
        double bestCost = std::numeric_limits<double>::max();
        for (size_t i = start + 1; i < end; ++i) {
            double lminX = std::numeric_limits<double>::max();
            double lminY = std::numeric_limits<double>::max();
            double lminZ = std::numeric_limits<double>::max();
            double lmaxX = std::numeric_limits<double>::lowest();
            double lmaxY = std::numeric_limits<double>::lowest();
            double lmaxZ = std::numeric_limits<double>::lowest();
            for (size_t j = start; j < i; ++j) {
                const auto &object = objects[j];
                lminX = std::min(lminX, object->getMin().getX());
                lmaxX = std::max(lmaxX, object->getMax().getX());
                lminY = std::min(lminY, object->getMin().getY());
                lmaxY = std::max(lmaxY, object->getMax().getY());
                lminZ = std::min(lminZ, object->getMin().getZ());
                lmaxZ = std::max(lmaxZ, object->getMax().getZ());
            }
            double larea = 2.0 * ((lmaxX - lminX) * (lmaxY - lminY) +
                                  (lmaxY - lminY) * (lmaxZ - lminZ) +
                                  (lmaxZ - lminZ) * (lmaxX - lminX));
            double rminX = std::numeric_limits<double>::max();
            double rminY = std::numeric_limits<double>::max();
            double rminZ = std::numeric_limits<double>::max();
            double rmaxX = std::numeric_limits<double>::lowest();
            double rmaxY = std::numeric_limits<double>::lowest();
            double rmaxZ = std::numeric_limits<double>::lowest();
            for (size_t j = i; j < end; ++j) {
                const auto &object = objects[j];
                rminX = std::min(rminX, object->getMin().getX());
                rmaxX = std::max(rmaxX, object->getMax().getX());
                rminY = std::min(rminY, object->getMin().getY());
                rmaxY = std::max(rmaxY, object->getMax().getY());
                rminZ = std::min(rminZ, object->getMin().getZ());
                rmaxZ = std::max(rmaxZ, object->getMax().getZ());
            }
            double rarea = 2.0 * ((rmaxX - rminX) * (rmaxY - rminY) +
                                  (rmaxY - rminY) * (rmaxZ - rminZ) +
                                  (rmaxZ - rminZ) * (rmaxX - rminX));
            double cost = larea * (i - start) + rarea * (end - i);
            if (cost < bestCost) {
                bestCost = cost;
                bestSplit = i;
            }
        }

        std::vector<BVH> kids;
        kids.reserve(2);
        kids.emplace_back(binarySurfaceAreaHeuristicConstruction(objects, start, bestSplit));
        kids.emplace_back(binarySurfaceAreaHeuristicConstruction(objects, bestSplit, end));

        return BVH{internalBox, std::move(kids)};
    }

public:
    static BVH stupidConstruct(std::vector<std::unique_ptr<Primitive>>& objects) {
        AABB box(
                {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
                {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()},
                std::move(objects)
        );
        std::vector<BVH> children;
        BVH bvh{box, children};
        return bvh;
    }

    static BVH medianSplitConstruction(std::vector<std::unique_ptr<Primitive>>& objects, int degree = 2) {
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

    static BVH binarySurfaceAreaHeuristicConstruction(std::vector<std::unique_ptr<Primitive>>& objects) {
        if (objects.empty()) {
            AABB emptyBox{
                    {0,0,0},
                    {0,0,0},
                    {}
            };
            return BVH{emptyBox, {}};
        }
        return binarySurfaceAreaHeuristicConstruction(objects, 0, objects.size());
    }

    std::unique_ptr<Ray> traverse(const Ray& ray) const {
        if (!box.hit(ray)) {
            return nullptr;
        }
        if (children.empty()) { //or box.getPrimitives().empty()
            std::unique_ptr<Ray> closest = nullptr;
            double closestDist = std::numeric_limits<double>::max();
            for (const auto &object: box.getPrimitives()) {
                if (object->intersect(ray)) {
                    auto p = object->getIntersectionNormalAndDirection(ray);
                    const double d = (p->getOrigin() - ray.getOrigin()).length();
                    if (d < closestDist) {
                        closestDist = d;
                        closest = std::move(p);
                    }
                }
            }
            return closest;
        }
        // Traverse children recursively
        std::unique_ptr<Ray> closest = nullptr;
        double closestDistance = std::numeric_limits<double>::max();
        for (const auto& child : children) {
            auto hit = child.traverse(ray);
            if (hit) {
                const double distance = (hit->getOrigin() - ray.getOrigin()).length();
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

