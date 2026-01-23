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

    static void findBounds(const std::vector<Primitive*>::iterator& begin,
                           const std::vector<Primitive*>::iterator& end,
                           Vector3& min, Vector3& max) {
        if (begin == end) {
            min = Vector3{0.0, 0.0, 0.0};
            max = Vector3{0.0, 0.0, 0.0};
            return;
        }
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

    static bool isLeaf(const size_t count, const int degree) {
        if (count <= static_cast<size_t>(degree) || degree < 2) {
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


    static std::vector<std::size_t> sahSplit(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis, const int degree) {

        const auto range = end - begin;
        if (isLeaf(range, degree)) return {};

        std::sort(begin, end,
                  [axis](const Primitive* a, const Primitive* b) {
                      return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                  });

        struct Segment { size_t begin; size_t end; }; // [begin, end)
        std::vector<Segment> segments;
        segments.push_back(Segment{0, static_cast<size_t>(range)
        });

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
                const Primitive *obj = *(begin + i);
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

        std::vector<size_t> splits;

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
                const Primitive *obj = *(begin + seg.begin + k);
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
                const Primitive *obj = *(begin + seg.begin + k);
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
            splits.emplace_back(absoluteSplitIndex);
            Segment left{seg.begin, absoluteSplitIndex};
            Segment right{absoluteSplitIndex, seg.end};
            segments.erase(segments.begin() + indexOfSegmentToSplit);
            segments.push_back(left);
            segments.push_back(right);
        }

        return splits;
    }

    static std::vector<size_t> binnedSahSplit(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis, const int degree) {

        const int BIN_SIZE = 16;

        const auto range = end - begin;
        if (isLeaf(range, degree)) return {};

        std::sort(begin, end,
                  [axis](const Primitive* a, const Primitive* b) {
                      return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                  });

        struct Bin {
            int count = 0;
            Vector3 min = Vector3{std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max(),
                                    std::numeric_limits<double>::max()};
            Vector3 max = Vector3{std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::lowest(),
                                    std::numeric_limits<double>::lowest()};
        };

        auto computeArea = [](const Vector3 &minP, const Vector3 &maxP) {
            Vector3 extent = maxP - minP;
            double ex = extent.getX(), ey = extent.getY(), ez = extent.getZ();
            return 2.0 * (ex * ey + ey * ez + ez * ex);
        };

        std::vector<Bin> bins(BIN_SIZE);    //TODO Init Bins?
        // Assign primitives to bins
        for (auto it = begin; it != end; ++it) {
            const Primitive* primitive = *it;
            const double centerValue = primitive->getCenter().getAxis(axis);
            // Determine bin index
            int binIndex = static_cast<int>(((centerValue - (*begin)->getMin().getAxis(axis)) /
                                            ((*(end-1))->getMax().getAxis(axis) - (*begin)->getMin().getAxis(axis))) * BIN_SIZE);
            binIndex = std::clamp(binIndex, 0, BIN_SIZE - 1);

            auto &[count, min, max] = bins[binIndex];
            min = {
                std::min(min.getX(), primitive->getMin().getX()),
                std::min(min.getY(), primitive->getMin().getY()),
                std::min(min.getZ(), primitive->getMin().getZ()),
            };
            max = {
                std::max(max.getX(), primitive->getMax().getX()),
                std::max(max.getY(), primitive->getMax().getY()),
                std::max(max.getZ(), primitive->getMax().getZ()),
            };
            count++;
        }

        struct Segment { size_t begin; size_t end; }; // [begin, end)
        std::vector<Segment> segments;
        segments.push_back(Segment{0, static_cast<size_t>(BIN_SIZE)
        });

        std::vector<size_t> splits;
        while (segments.size() < static_cast<size_t>(degree)) {
            Segment seg = segments.back();
            segments.pop_back();
            size_t count = seg.end - seg.begin;

            // calculate prefix and suffix
            // Prefix (left) and suffix (right) cumulative bounds inside this segment
            std::vector<Vector3> prefixMin(count), prefixMax(count);
            for (size_t k = 0; k < count; ++k) {
                const Bin &bin = bins[k];
                if (k == 0) {
                    prefixMin[k] = bin.min;
                    prefixMax[k] = bin.max;
                } else {
                    const Vector3 &prevMin = prefixMin[k - 1];
                    const Vector3 &prevMax = prefixMax[k - 1];
                    prefixMin[k] = {std::min(prevMin.getX(), bin.min.getX()),
                                    std::min(prevMin.getY(), bin.min.getY()),
                                    std::min(prevMin.getZ(), bin.min.getZ())};
                    prefixMax[k] = {std::max(prevMax.getX(), bin.max.getX()),
                                    std::max(prevMax.getY(), bin.max.getY()),
                                    std::max(prevMax.getZ(), bin.max.getZ())};
                }
            }

            std::vector<Vector3> suffixMin(count), suffixMax(count);
            for (size_t k = count; k-- > 0;) {
                const Bin &bin = bins[k];
                if (k == count - 1) {
                    suffixMin[k] = bin.min;
                    suffixMax[k] = bin.max;
                } else {
                    const Vector3 &nextMin = suffixMin[k + 1];
                    const Vector3 &nextMax = suffixMax[k + 1];
                    suffixMin[k] = {std::min(nextMin.getX(), bin.min.getX()),
                                    std::min(nextMin.getY(), bin.min.getY()),
                                    std::min(nextMin.getZ(), bin.min.getZ())};
                    suffixMax[k] = {std::max(nextMax.getX(), bin.max.getX()),
                                    std::max(nextMax.getY(), bin.max.getY()),
                                    std::max(nextMax.getZ(), bin.max.getZ())};
                }
            }

            //calculate best splits
            // TODO allow multiple splits
            size_t bestSplit = 1; // position inside segment
            int leftCount = 0;
            int rightCount = static_cast<int>(range);
            double bestLocalCost = std::numeric_limits<double>::max();
            for (size_t split = 1; split < count; ++split) {
                leftCount += bins[split - 1].count;
                rightCount -= bins[split - 1].count;

                const Vector3 &leftMin = prefixMin[split - 1];
                const Vector3 &leftMax = prefixMax[split - 1];
                const Vector3 &rightMin = suffixMin[split];
                const Vector3 &rightMax = suffixMax[split];
                double leftArea = computeArea(leftMin, leftMax);
                double rightArea = computeArea(rightMin, rightMax);
                double cost = leftArea * leftCount + rightArea * rightCount;
                if (cost < bestLocalCost) {
                    bestLocalCost = cost;
                    bestSplit = leftCount;

                }
            }

            size_t absoluteSplitIndex = seg.begin + bestSplit;
            splits.emplace_back(absoluteSplitIndex);
            Segment left{seg.begin, absoluteSplitIndex};
            Segment right{absoluteSplitIndex, seg.end};
            segments.push_back(left);
            segments.push_back(right);
        }

        for (const unsigned long splitIndex : splits) {
            std::nth_element(begin, begin + static_cast<std::ptrdiff_t>(splitIndex), end,
                [axis](const Primitive* a, const Primitive* b) {
                    return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                });
        }
        return splits;
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

    static std::vector<std::size_t> median16Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return medianSplit(begin, end, axis, 16);
    }

    static std::vector<std::size_t> sah2Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return sahSplit(begin, end, axis, 2);
    }

    static std::vector<std::size_t> sah4Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return sahSplit(begin, end, axis, 4);
    }

    static std::vector<std::size_t> sah8Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return sahSplit(begin, end, axis, 8);
    }

    static std::vector<std::size_t> sah16Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return sahSplit(begin, end, axis, 16);
    }

    static std::vector<std::size_t> binnedSah2Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return binnedSahSplit(begin, end, axis, 2);
    }

    static std::vector<std::size_t> binnedSah4Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return binnedSahSplit(begin, end, axis, 4);
    }

    static std::vector<std::size_t> binnedSah8Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return binnedSahSplit(begin, end, axis, 8);
    }

    static std::vector<std::size_t> binnedSah16Split(const std::vector<Primitive*>::iterator& begin, const std::vector<Primitive*>::iterator& end, const int axis) {
        return binnedSahSplit(begin, end, axis, 16);
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

            const size_t nodeSize = node->end - node->begin;
            if (nodeSize <= 1) {
                continue; // This is a leaf node
            }

            const int axis = calculateLongestAxis(node->box.getMin(), node->box.getMax());
            std::vector<std::size_t> splitIndices = partitionFunction(node->begin, node->end, axis);
            if (splitIndices.empty()) continue; //leaf node

            // Sort split indices to ensure they are in ascending order
            std::sort(splitIndices.begin(), splitIndices.end());

            // allow for multiple splits
            auto rangeBegin = node->begin;
            for (unsigned long splitIndex : splitIndices) {
                if (splitIndex == 0 || splitIndex >= static_cast<std::size_t>(node->end - node->begin)) {
                    throw std::out_of_range("invalid split position"); //invalid split
                }
                // splitIndex is absolute position in node's range, convert to proper iterator
                auto rangeEnd = node->begin + static_cast<std::ptrdiff_t>(splitIndex);

                // Validation to prevent invalid ranges
                if (rangeBegin >= rangeEnd) {
                    throw std::out_of_range("Invalid iterator range");
                }

                Vector3 min, max;
                findBounds(rangeBegin, rangeEnd, min, max);
                node->children.emplace_back(BVHNode{AABB{min, max}, rangeBegin, rangeEnd, {}});
                rangeBegin = rangeEnd;
            }

            //everything right of the last split
            Vector3 min, max;
            findBounds(rangeBegin, node->end, min, max);
            node->children.emplace_back(BVHNode{AABB{min, max}, rangeBegin, node->end, {}});

            // Add children to stack AFTER all children are created to avoid pointer invalidation
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
    static std::unique_ptr<Ray> traverse(const StackBVH& bvh, const Ray& ray, int* aabbTestCountPositive, int* aabbTestCountNegative, int* triTestCountPositive, int* triTestCountNegative) {
        std::vector<const BVHNode*> stack;
        stack.reserve(64);  //Won't be deeper than 64 levels, no reallocs
        stack.push_back(&bvh.root);

        std::unique_ptr<Ray> closest = nullptr;
        double closestDist = std::numeric_limits<double>::max();

        while (!stack.empty()) {
            const BVHNode* node = stack.back();
            stack.pop_back();

            if (!node->box.hit(ray)) {
                (*aabbTestCountNegative)++;
                continue;
            }
            (*aabbTestCountPositive)++;
            if (node->children.empty()) {
                for (int i = 0; i < static_cast<int>(node->end - node->begin); ++i) {
                    (*triTestCountPositive)++;
                    if (const Primitive* primitive = *(node->begin + i); primitive->intersect(ray)) {
                        (*triTestCountPositive)++;
                        auto hitRay = primitive->getIntersectionNormalAndDirection(ray);
                        if (const double dist = (hitRay->getOrigin() - ray.getOrigin()).length(); dist < closestDist) {
                            closestDist = dist;
                            closest = std::move(hitRay);
                        }
                    }else {
                        (*triTestCountNegative)++;
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

