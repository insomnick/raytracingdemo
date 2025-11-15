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
    medianSplitConstruction(std::vector<Primitive*> &objects, size_t start, size_t end, int degree) {

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
            std::vector<Primitive*> primitiveLeaves;
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

        AABB internalBox{{minX, minY, minZ},
                         {maxX, maxY, maxZ},
                         {}};

        size_t segmentBegin = start;
        for (int i = 1; i < degree; ++i) {
            size_t boundary = start + (count * i) / degree; // target index for this partition boundary
            std::nth_element(objects.begin() + segmentBegin,
                             objects.begin() + boundary,
                             objects.begin() + end,
                             [axis](const Primitive* a, const Primitive* b) {
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
    static BVH binarySurfaceAreaHeuristicConstruction(std::vector<Primitive*> &objects, size_t start, size_t end) {
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
            std::vector<Primitive*> primitiveLeaves;
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

        AABB internalBox{{minX, minY, minZ},
                         {maxX, maxY, maxZ},
                         {}};
        // we need to calculate area heuristic here to find the best split point
        std::sort(objects.begin() + start, objects.begin() + end,
                  [axis](const Primitive* a, const Primitive* b){
                      return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                  });

        size_t bestSplit = start + 1;;
        double bestCost = std::numeric_limits<double>::max();

        const size_t n = end - start;
        std::vector<double> leftMinX(n), leftMinY(n), leftMinZ(n),
                leftMaxX(n), leftMaxY(n), leftMaxZ(n);
        for (size_t k = 0; k < n; ++k) {
            const auto &obj = objects[start + k];
            double ominX = obj->getMin().getX();
            double ominY = obj->getMin().getY();
            double ominZ = obj->getMin().getZ();
            double omaxX = obj->getMax().getX();
            double omaxY = obj->getMax().getY();
            double omaxZ = obj->getMax().getZ();
            if (k == 0) {
                leftMinX[k] = ominX; leftMaxX[k] = omaxX;
                leftMinY[k] = ominY; leftMaxY[k] = omaxY;
                leftMinZ[k] = ominZ; leftMaxZ[k] = omaxZ;
            } else {
                leftMinX[k] = std::min(leftMinX[k-1], ominX);
                leftMaxX[k] = std::max(leftMaxX[k-1], omaxX);
                leftMinY[k] = std::min(leftMinY[k-1], ominY);
                leftMaxY[k] = std::max(leftMaxY[k-1], omaxY);
                leftMinZ[k] = std::min(leftMinZ[k-1], ominZ);
                leftMaxZ[k] = std::max(leftMaxZ[k-1], omaxZ);
            }
        }
        std::vector<double> rightMinX(n), rightMinY(n), rightMinZ(n),
                rightMaxX(n), rightMaxY(n), rightMaxZ(n);
        for (size_t k = n; k-- > 0;) {
            const auto &obj = objects[start + k];
            double ominX = obj->getMin().getX();
            double ominY = obj->getMin().getY();
            double ominZ = obj->getMin().getZ();
            double omaxX = obj->getMax().getX();
            double omaxY = obj->getMax().getY();
            double omaxZ = obj->getMax().getZ();
            if (k == n - 1) {
                rightMinX[k] = ominX; rightMaxX[k] = omaxX;
                rightMinY[k] = ominY; rightMaxY[k] = omaxY;
                rightMinZ[k] = ominZ; rightMaxZ[k] = omaxZ;
            } else {
                rightMinX[k] = std::min(rightMinX[k+1], ominX);
                rightMaxX[k] = std::max(rightMaxX[k+1], omaxX);
                rightMinY[k] = std::min(rightMinY[k+1], ominY);
                rightMaxY[k] = std::max(rightMaxY[k+1], omaxY);
                rightMinZ[k] = std::min(rightMinZ[k+1], ominZ);
                rightMaxZ[k] = std::max(rightMaxZ[k+1], omaxZ);
            }
        }

        for (size_t split = 1; split < n; ++split) {
            double lminX = leftMinX[split-1], lmaxX = leftMaxX[split-1];
            double lminY = leftMinY[split-1], lmaxY = leftMaxY[split-1];
            double lminZ = leftMinZ[split-1], lmaxZ = leftMaxZ[split-1];
            double larea = 2.0 * ((lmaxX - lminX) * (lmaxY - lminY) +
                                  (lmaxY - lminY) * (lmaxZ - lminZ) +
                                  (lmaxZ - lminZ) * (lmaxX - lminX));

            double rminX = rightMinX[split], rmaxX = rightMaxX[split];
            double rminY = rightMinY[split], rmaxY = rightMaxY[split];
            double rminZ = rightMinZ[split], rmaxZ = rightMaxZ[split];
            double rarea = 2.0 * ((rmaxX - rminX) * (rmaxY - rminY) +
                                  (rmaxY - rminY) * (rmaxZ - rminZ) +
                                  (rmaxZ - rminZ) * (rmaxX - rminX));

            double cost = larea * split + rarea * (n - split);
            if (cost < bestCost) {
                bestCost = cost;
                bestSplit = start + split;
            }
        }

        std::vector<BVH> kids;
        kids.reserve(2);
        kids.emplace_back(binarySurfaceAreaHeuristicConstruction(objects, start, bestSplit));
        kids.emplace_back(binarySurfaceAreaHeuristicConstruction(objects, bestSplit, end));

        return BVH{internalBox, std::move(kids)};
    }

public:
    static BVH stupidConstruct(std::vector<Primitive*>& objects) {
        AABB box(
                {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
                {std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest()},
                objects
        );
        std::vector<BVH> children;
        BVH bvh{box, children};
        return bvh;
    }

    static BVH medianSplitConstruction(std::vector<Primitive*>& objects, int degree = 2) {
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

    static BVH binarySurfaceAreaHeuristicConstruction(std::vector<Primitive*>& objects) {
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

