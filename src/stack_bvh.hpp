#ifndef RAYTRACINGDEMO_STACK_BVH_HPP
#define RAYTRACINGDEMO_STACK_BVH_HPP

#include <vector>
#include <limits>

#include "aabb.hpp"
#include "primitives/primitive.hpp"
#include "primitives/ray.hpp"

class StackBVH {
private:
    struct BVHNode {
        AABB box;
        std::vector<Primitive*>::iterator begin;
        std::vector<Primitive*>::iterator end;
        std::vector<BVHNode> children;
    };

    BVHNode root;
    std::vector<Primitive*> primitives;

    explicit StackBVH(BVHNode rootNode, std::vector<Primitive*> primitives)
        : root(std::move(rootNode)), primitives(std::move(primitives)) {}

    static void findBounds(const std::vector<Primitive*>::const_iterator& begin,
                           const std::vector<Primitive*>::const_iterator& end,
                           Vector3& min, Vector3& max) {
        if (begin == end) {
            min = Vector3{0.0, 0.0, 0.0};
            max = Vector3{0.0, 0.0, 0.0};
            return;
        }
        //calculate bounds of root
        auto minimumX = std::numeric_limits<double>::max();
        auto minimumY = std::numeric_limits<double>::max();
        auto minimumZ = std::numeric_limits<double>::max();
        auto maximumX = std::numeric_limits<double>::lowest();
        auto maximumY = std::numeric_limits<double>::lowest();
        auto maximumZ = std::numeric_limits<double>::lowest();
        for (auto iterator = begin; iterator != end; ++iterator) {
            const Primitive* primitive = *iterator;
            minimumX = std::min(minimumX,primitive->getMin().getX());
            minimumY = std::min(minimumY,primitive->getMin().getY());
            minimumZ = std::min(minimumZ,primitive->getMin().getZ());
            maximumX = std::max(maximumX,primitive->getMax().getX());
            maximumY = std::max(maximumY,primitive->getMax().getY());
            maximumZ = std::max(maximumZ,primitive->getMax().getZ());
        }

        min = Vector3{minimumX, minimumY, minimumZ};
        max = Vector3{maximumX, maximumY, maximumZ};
    }

    static int calculateLongestAxis(const Vector3 &min, const Vector3 &max) {
        const Vector3 axisLengths = max - min;
        const double lenX = axisLengths.getX();
        const double lenY = axisLengths.getY();
        const double lenZ = axisLengths.getZ();
        int axis = 0;
        if (lenY > lenX && lenY >= lenZ) axis = 1;
        else if (lenZ > lenX && lenZ >= lenY) axis = 2;
        return axis;
    }

    static double calculateSurfaceArea (const Vector3 &minP, const Vector3 &maxP) {
        const Vector3 extent = maxP - minP;
        const double ex = extent.getX(), ey = extent.getY(), ez = extent.getZ();
        return 2.0 * (ex * ey + ey * ez + ez * ex);
    };

    static bool isLeaf(const int count, const int degree) {
        if (count <= degree || degree < 2) {
            return true; // Leaf node
        }
        return false;
    }

    // median split partitioning, returns splitting point
    static std::vector<std::size_t> medianSplit(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis, const int degree) {

        const auto count = end - begin;
        if (isLeaf(count, degree)) return {};

        std::vector<std::size_t> splitIndices;
        for (int i = 1; i < degree; ++i) {
            size_t split = (count * i) / degree;
            if (split == 0 || split >= static_cast<std::size_t>(count)) break;
            splitIndices.push_back(split);
        }

        for (const unsigned long splitIndex : splitIndices) {
            std::nth_element(begin, begin + static_cast<std::ptrdiff_t>(splitIndex), end,
                [axis](const Primitive* a, const Primitive* b) {
                    return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                });
        }
        return splitIndices;
    }

public:

    static std::vector<std::size_t> median2Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return medianSplit(begin, end, axis, 2);
    }

    static std::vector<std::size_t> median4Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return medianSplit(begin, end, axis, 4);
    }

    static std::vector<std::size_t> median8Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return medianSplit(begin, end, axis, 8);
    }

    static std::vector<std::size_t> sah2Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {

        std::sort(begin, end,
                  [axis](const Primitive* a, const Primitive* b) {
                      return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                  });
        const size_t count = end - begin;
        std::vector<Vector3> leftMins(count), leftMaxs(count), rightMins(count), rightMaxs(count);

        // prefix bounds for left side
        findBounds(begin, begin + 1, leftMins[0], leftMaxs[0]);

        for (size_t i = 1; i < count; ++i) {
            Vector3 tmpMin, tmpMax;
            findBounds(begin + static_cast<long>(i), begin + static_cast<long>(i + 1), tmpMin, tmpMax);
            const auto minX = std::min(leftMins[i - 1].getX(), tmpMin.getX());
            const auto minY = std::min(leftMins[i - 1].getY(), tmpMin.getY());
            const auto minZ = std::min(leftMins[i - 1].getZ(), tmpMin.getZ());
            const auto maxX = std::max(leftMaxs[i - 1].getX(), tmpMax.getX());
            const auto maxY = std::max(leftMaxs[i - 1].getY(), tmpMax.getY());
            const auto maxZ = std::max(leftMaxs[i - 1].getZ(), tmpMax.getZ());
            leftMins[i] = Vector3{minX, minY, minZ};
            leftMaxs[i] = Vector3{maxX, maxY, maxZ};
        }

        // suffix bounds for right side
        findBounds(begin + static_cast<long>(count - 1), begin + static_cast<long>(count), rightMins[count - 1], rightMaxs[count - 1]);
        for (size_t i = count - 1; i-- > 0;) {
            Vector3 tmpMin, tmpMax;
            findBounds(begin + static_cast<long>(i), begin + static_cast<long>(i + 1), tmpMin, tmpMax);
            const auto minX = std::min(rightMins[i + 1].getX(), tmpMin.getX());
            const auto minY = std::min(rightMins[i + 1].getY(), tmpMin.getY());
            const auto minZ = std::min(rightMins[i + 1].getZ(), tmpMin.getZ());
            const auto maxX = std::max(rightMaxs[i + 1].getX(), tmpMax.getX());
            const auto maxY = std::max(rightMaxs[i + 1].getY(), tmpMax.getY());
            const auto maxZ = std::max(rightMaxs[i + 1].getZ(), tmpMax.getZ());
            rightMins[i] = Vector3{minX, minY, minZ};
            rightMaxs[i] = Vector3{maxX, maxY, maxZ};
        }

        size_t split = count / 2;
        double minCost = std::numeric_limits<double>::max();
        for (size_t i = 1; i < count - 1; ++i) {
            const double leftArea = calculateSurfaceArea(leftMins[i - 1], leftMaxs[i - 1]);
            const double rightArea = calculateSurfaceArea(rightMins[i], rightMaxs[i]);
            if (const double cost = leftArea * static_cast<double>(i) + rightArea * static_cast<double>((count - i)); minCost > cost) {
                minCost = cost;
                split = i;
            }
        }

        std::vector<std::size_t> splitVec;
        splitVec.push_back(split);
        return splitVec;
    }

    // BVH construction with lambda for splitting
    template<typename PartitionFunction>
    static StackBVH build(std::vector<Primitive*>& inputPrimitives,
                                PartitionFunction partitionFunction) {
        // move input primitives into a single owned vector inside the BVH
        std::vector ownedPrimitives(inputPrimitives.begin(), inputPrimitives.end());
        if (ownedPrimitives.empty()) {
            BVHNode emptyRoot{AABB{{0,0,0}, {0,0,0}}, ownedPrimitives.begin(), ownedPrimitives.end(), {}};
            return StackBVH{std::move(emptyRoot), std::move(ownedPrimitives)};
        }

        Vector3 globalMin;
        Vector3 globalMax;
        findBounds(ownedPrimitives.begin(), ownedPrimitives.end(), globalMin, globalMax);
        const AABB box{globalMin, globalMax};

        BVHNode rootNode{box, ownedPrimitives.begin(), ownedPrimitives.end(), {}};
        StackBVH bvh{std::move(rootNode), std::move(ownedPrimitives)};

        std::vector<BVHNode*> nodes;
        nodes.push_back(&bvh.root); // push root to stack

        while (!nodes.empty()) {
            BVHNode* node = nodes.back();
            nodes.pop_back();

            const int axis = calculateLongestAxis(node->box.getMin(), node->box.getMax());
            const std::vector<std::size_t> splitIndices = partitionFunction(node->begin, node->end, axis);
            if (splitIndices.empty()) continue; //leaf node

            // allow for multiple splits
            auto rangeBegin = node->begin;
            for (unsigned long splitIndex : splitIndices) {
                if (splitIndex == 0 || splitIndex >= static_cast<std::size_t>(node->end - node->begin)) {
                    throw std::out_of_range("invalid split position"); //invalid split
                }
                auto rangeEnd = node->begin + static_cast<std::ptrdiff_t>(splitIndex);
                Vector3 min, max;
                findBounds(rangeBegin, rangeEnd, min, max);
                node->children.emplace_back(BVHNode{AABB{min, max}, rangeBegin, rangeEnd, {}});
                rangeBegin = rangeEnd;
            }

            //everything right of the last split
            Vector3 min, max;
            findBounds(rangeBegin, node->end, min, max);
            node->children.emplace_back(BVHNode{AABB{min, max}, rangeBegin, node->end, {}});

            for (auto& child : node->children) {
                nodes.push_back(&child);
            }
        }

        return bvh;
    }

    //let nodes absorb their children
    static void collapse(StackBVH &bvh) {
        std::vector<BVHNode*> stack;
        stack.reserve(64);  //should min be ~ log(ownedPrimitives.size())
        stack.push_back(&bvh.root);

        while (!stack.empty()) {
            BVHNode* node = stack.back();
            stack.pop_back();

            if (node->children.empty()) {
                continue;
            }

            std::vector<BVHNode> newChildren;
            newChildren.reserve(node->children.size());

            for (auto &child : node->children) {
                // if child has children, lift them up one level
                if (!child.children.empty()) {
                    for (auto &grandChild : child.children) {
                        newChildren.push_back(std::move(grandChild));
                    }
                } else {
                    // keep leaf child
                    newChildren.push_back(std::move(child));
                }
            }

            node->children = std::move(newChildren);
            //Next Iteration
            for (auto &child : node->children) {
                stack.push_back(&child);
            }
        }
    }

    //first-hit-traversal
    static std::unique_ptr<Ray> traverse(const StackBVH& bvh, const Ray& ray) {
        std::vector<const BVHNode*> stack;
        stack.reserve(64);  //Won't be deeper than 64 levels, no reallocs
        stack.push_back(&bvh.root);

        std::unique_ptr<Ray> closest = nullptr;
        double closestDist = std::numeric_limits<double>::max();

        while (!stack.empty()) {
            const BVHNode* node = stack.back();
            stack.pop_back();

            if (!node->box.hit(ray)) {
                continue;
            }

            if (node->children.empty()) {
                for (int i = 0; i < static_cast<int>(node->end - node->begin); ++i) {
                    if (const Primitive* primitive = *(node->begin + i); primitive->intersect(ray)) {
                        auto hitRay = primitive->getIntersectionNormalAndDirection(ray);
                        if (const double dist = (hitRay->getOrigin() - ray.getOrigin()).length(); dist < closestDist) {
                            closestDist = dist;
                            closest = std::move(hitRay);
                        }
                    }
                }
            }

            for (const auto& child : node->children) {
                stack.push_back(&child);
            }
        }
        return closest;
    }
};

#endif // RAYTRACINGDEMO_STACK_BVH_HPP

