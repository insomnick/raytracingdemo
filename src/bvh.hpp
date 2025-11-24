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

    static BVH medianSplitConstruction(std::vector<Primitive*> &objects, size_t start, size_t end, int degree) {

        AABB boundingBox = calculateBoundingBox(objects, start, end, degree);
        if (!boundingBox.getPrimitives().empty()){
            return BVH{boundingBox, {}};
        }
        int splitAxis = calculateLongestAxis(boundingBox);

        sortAtMedianSplits(objects, start, end, degree, splitAxis);

        std::vector<BVH> children;
        children.reserve(degree);
        size_t childStart = start;
        size_t count = end - start;
        for (int i = 0; i < degree; ++i) {
            size_t childEnd = (i == degree - 1) ? end : start + (count * (i + 1)) / degree; // last child takes the rest
            children.emplace_back(medianSplitConstruction(objects, childStart, childEnd, degree));
            childStart = childEnd;
        }

        return BVH{boundingBox, std::move(children)};
    }

    static void sortAtMedianSplits(std::vector<Primitive *> &objects, size_t start, size_t end, int degree, int axis) {
        size_t count = end - start;
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
    }

    static BVH surfaceAreaHeuristicConstruction(std::vector<Primitive*> &objects, size_t start, size_t end, int degree) {
        AABB boundingBox = calculateBoundingBox(objects, start, end, degree);
        if (!boundingBox.getPrimitives().empty()) {
            return BVH{boundingBox, {}}; // leaf
        }

        const int splitAxis = calculateLongestAxis(boundingBox);
        std::sort(objects.begin() + start, objects.begin() + end,
                  [splitAxis](const Primitive* a, const Primitive* b) {
                      return a->getCenter().getAxis(splitAxis) < b->getCenter().getAxis(splitAxis);
                  });

        struct Segment { size_t begin; size_t end; }; // [begin, end)
        std::vector<Segment> segments;
        segments.push_back({start, end});

        auto computeArea = [](const Vector3 &minP, const Vector3 &maxP) {
            Vector3 extent = maxP - minP;
            double ex = extent.getX(), ey = extent.getY(), ez = extent.getZ();
            return 2.0 * (ex * ey + ey * ez + ez * ex);
        };

        auto accumulateBounds = [&](const Segment &s, Vector3 &minP, Vector3 &maxP) {
            double minX = std::numeric_limits<double>::max();
            double minY = std::numeric_limits<double>::max();
            double minZ = std::numeric_limits<double>::max();
            double maxX = std::numeric_limits<double>::lowest();
            double maxY = std::numeric_limits<double>::lowest();
            double maxZ = std::numeric_limits<double>::lowest();
            for (size_t i = s.begin; i < s.end; ++i) {
                const Primitive *obj = objects[i];
                minX = std::min(minX, obj->getMin().getX());
                maxX = std::max(maxX, obj->getMax().getX());
                minY = std::min(minY, obj->getMin().getY());
                maxY = std::max(maxY, obj->getMax().getY());
                minZ = std::min(minZ, obj->getMin().getZ());
                maxZ = std::max(maxZ, obj->getMax().getZ());
            }
            minP = {minX, minY, minZ};
            maxP = {maxX, maxY, maxZ};
        };

        auto segmentCost = [&](const Segment &s) {
            Vector3 minP, maxP;
            accumulateBounds(s, minP, maxP);
            double area = computeArea(minP, maxP);
            return area * static_cast<double>(s.end - s.begin);
        };

        // Greedy splitting: each iteration split the highest-cost segment until reaching desired degree
        while (segments.size() < static_cast<size_t>(degree)) {
            size_t indexOfSegmentToSplit = SIZE_MAX;
            double highestCost = -1.0;
            for (size_t i = 0; i < segments.size(); ++i) {
                const Segment &seg = segments[i];
                size_t count = seg.end - seg.begin;
                if (count < 2) continue; // cannot split further
                double cost = segmentCost(seg);
                if (cost > highestCost) {
                    highestCost = cost;
                    indexOfSegmentToSplit = i;
                }
            }
            if (indexOfSegmentToSplit == SIZE_MAX) break; // no splittable segment

            Segment seg = segments[indexOfSegmentToSplit];
            size_t count = seg.end - seg.begin;

            // Prefix (left) and suffix (right) cumulative bounds inside this segment
            std::vector<Vector3> prefixMin(count), prefixMax(count);
            for (size_t k = 0; k < count; ++k) {
                const Primitive *obj = objects[seg.begin + k];
                Vector3 omin = obj->getMin();
                Vector3 omax = obj->getMax();
                if (k == 0) {
                    prefixMin[k] = omin;
                    prefixMax[k] = omax;
                } else {
                    const Vector3 &prevMin = prefixMin[k - 1];
                    const Vector3 &prevMax = prefixMax[k - 1];
                    prefixMin[k] = {std::min(prevMin.getX(), omin.getX()),
                                    std::min(prevMin.getY(), omin.getY()),
                                    std::min(prevMin.getZ(), omin.getZ())};
                    prefixMax[k] = {std::max(prevMax.getX(), omax.getX()),
                                    std::max(prevMax.getY(), omax.getY()),
                                    std::max(prevMax.getZ(), omax.getZ())};
                }
            }

            std::vector<Vector3> suffixMin(count), suffixMax(count);
            for (size_t k = count; k-- > 0;) {
                const Primitive *obj = objects[seg.begin + k];
                Vector3 omin = obj->getMin();
                Vector3 omax = obj->getMax();
                if (k == count - 1) {
                    suffixMin[k] = omin;
                    suffixMax[k] = omax;
                } else {
                    const Vector3 &nextMin = suffixMin[k + 1];
                    const Vector3 &nextMax = suffixMax[k + 1];
                    suffixMin[k] = {std::min(nextMin.getX(), omin.getX()),
                                    std::min(nextMin.getY(), omin.getY()),
                                    std::min(nextMin.getZ(), omin.getZ())};
                    suffixMax[k] = {std::max(nextMax.getX(), omax.getX()),
                                    std::max(nextMax.getY(), omax.getY()),
                                    std::max(nextMax.getZ(), omax.getZ())};
                }
            }

            size_t bestLocalOffset = 1; // position inside segment
            double bestLocalCost = std::numeric_limits<double>::max();
            for (size_t split = 1; split < count; ++split) {
                const Vector3 &leftMin = prefixMin[split - 1];
                const Vector3 &leftMax = prefixMax[split - 1];
                const Vector3 &rightMin = suffixMin[split];
                const Vector3 &rightMax = suffixMax[split];
                double leftArea = computeArea(leftMin, leftMax);
                double rightArea = computeArea(rightMin, rightMax);
                double cost = leftArea * split + rightArea * (count - split);
                if (cost < bestLocalCost) {
                    bestLocalCost = cost;
                    bestLocalOffset = split;
                }
            }

            size_t absoluteSplitIndex = seg.begin + bestLocalOffset;
            Segment left{seg.begin, absoluteSplitIndex};
            Segment right{absoluteSplitIndex, seg.end};
            segments.erase(segments.begin() + indexOfSegmentToSplit);
            segments.push_back(left);
            segments.push_back(right);
        }

        std::vector<BVH> childNodes;
        childNodes.reserve(segments.size());
        for (const Segment &seg : segments) {
            childNodes.emplace_back(surfaceAreaHeuristicConstruction(objects, seg.begin, seg.end, degree));
        }
        return BVH{boundingBox, std::move(childNodes)};
    }

    static AABB calculateBoundingBox(std::vector<Primitive*> &objects, size_t start, size_t end, int degree){
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
        if (count == 0 || degree <= 1 || count <static_cast<size_t>(degree)) {
            std::vector<Primitive*> primitiveLeaves;
            primitiveLeaves.reserve(count);
            for (size_t i = start; i < end; ++i) {
                primitiveLeaves.push_back(objects[i]);
            }
            return {{minX, minY, minZ}, {maxX, maxY, maxZ}, primitiveLeaves};   //leaf node
        }
        return {{minX, minY, minZ}, {maxX, maxY, maxZ}, {}};
    }

    static int calculateLongestAxis(const AABB &boundingBox) {
        Vector3 axisLengths = boundingBox.getMax() - boundingBox.getMin();
        const double lenX = axisLengths.getX();
        const double lenY = axisLengths.getY();
        const double lenZ = axisLengths.getZ();
        int axis = 0;
        if (lenY > lenX && lenY >= lenZ) axis = 1;
        else if (lenZ > lenX && lenZ >= lenY) axis = 2;
        return axis;
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

    static BVH medianSplitConstruction(std::vector<Primitive*>& objects, int degree) {
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

    static BVH surfaceAreaHeuristicConstruction(std::vector<Primitive*>& objects, int degree) {
        if (objects.empty()) {
            AABB emptyBox{
                    {0,0,0},
                    {0,0,0},
                    {}
            };
            return BVH{emptyBox, {}};
        }
        return surfaceAreaHeuristicConstruction(objects, 0, objects.size(), degree);
    }

    //first-hit-traversal
    static std::unique_ptr<Ray> traverse(const BVH& bvh, const Ray& ray) {
        std::vector<const BVH*> stack;
        stack.reserve(64);  //Won't be deeper than 64 levels, no reallocs
        stack.push_back(&bvh);

        std::unique_ptr<Ray> closest = nullptr;
        double closestDist = std::numeric_limits<double>::max();

        while (!stack.empty()) {
            const BVH* node = stack.back();
            stack.pop_back();

            if (!node->box.hit(ray)) {
                continue;
            }

            if (node->children.empty()) {
                for (const Primitive* primitive : node->box.getPrimitives()) {
                    if (primitive->intersect(ray)) {
                        auto hitRay = primitive->getIntersectionNormalAndDirection(ray);
                        if (const double dist = (hitRay->getOrigin() - ray.getOrigin()).length(); dist < closestDist) {
                            closestDist = dist;
                            closest = std::move(hitRay);
                        }
                    }
                }
            } else {
                for (const BVH& child : node->children) {
                    stack.push_back(&child);
                }
            }
        }
        return closest;
    }
};

#endif //RAYTRACINGDEMO_BVH_HPP

