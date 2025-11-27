#ifndef RAYTRACINGDEMO_STACK_BVH_HPP
#define RAYTRACINGDEMO_STACK_BVH_HPP

#include "aabb.hpp"
#include "bvh.hpp"

class StackBVH {
private:
    struct BVHNode {
        AABB box;
        std::vector<Primitive*> primitives;
        std::vector<BVHNode> children;
    };

    BVHNode root;

    explicit StackBVH(BVHNode root) : root(std::move(root)) {}

    static void findBounds(const std::vector<Primitive*>::const_iterator& begin, const std::vector<Primitive*>::const_iterator& end, Vector3& min, Vector3& max) {
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

    static bool isLeaf(const BVHNode &node) {
        if (node.primitives.size() < 2) {   //Binary
            return true; // Leaf node
        }
        return false;
    }

public:
    // median split partitioning, returns splitting point
    static std::size_t medianSplit(std::vector<Primitive *>& primitives, const int axis) {

        const size_t split = primitives.size() / 2;

        std::nth_element(primitives.begin(),
                             primitives.begin() + split,
                             primitives.end(),
                             [axis](const Primitive* a, const Primitive* b) {
                                 return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                             });
        return split;
    }

    static std::size_t sahSplit(std::vector<Primitive *>& primitives, const int axis) {

        std::sort(primitives.begin(), primitives.end(),
                  [axis](const Primitive* a, const Primitive* b) {
                      return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                  });
        const size_t count = primitives.size();
        std::vector<Vector3> leftMins(count), leftMaxs(count), rightMins(count), rightMaxs(count);

        // prefix bounds for left side
        findBounds(primitives.begin(), primitives.begin() + 1, leftMins[0], leftMaxs[0]);

        for (size_t i = 1; i < count; ++i) {
            Vector3 tmpMin, tmpMax;
            findBounds(primitives.begin() + static_cast<long>(i), primitives.begin() + static_cast<long>(i + 1), tmpMin, tmpMax);
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
        findBounds(primitives.begin() + static_cast<long>(count - 1), primitives.begin() + static_cast<long>(count), rightMins[count - 1], rightMaxs[count - 1]);
        for (size_t i = count - 1; i-- > 0;) {
            Vector3 tmpMin, tmpMax;
            findBounds(primitives.begin() + static_cast<long>(i), primitives.begin() + static_cast<long>(i + 1), tmpMin, tmpMax);
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
        return split;
    }

    // binary BVH construction with lambda for splitting
    template<typename PartitionFunction>
    static StackBVH binaryBuild(std::vector<Primitive*>& primitives, PartitionFunction partitionFunction) {
        if (primitives.empty()) {
            return StackBVH{BVHNode{AABB{{0,0,0}, {0,0,0}}, {}, {}}};
        }

        Vector3 globalMin;
        Vector3 globalMax;
        findBounds(primitives.begin(), primitives.end(), globalMin, globalMax);
        const AABB box{globalMin, globalMax};

        BVHNode root = {box, primitives, {}};
        std::vector<BVHNode*> nodes;
        nodes.push_back(&root); //Push root to stack

        while (!nodes.empty()) {
            BVHNode* node = nodes.back();
            nodes.pop_back();
            if (isLeaf(*node)) continue;

            const size_t splitIndex = partitionFunction(node->primitives, calculateLongestAxis(node->box.getMin(), node->box.getMax()));
            //split primitives into left and right child
            auto rightNodes = std::vector(
                    node->primitives.begin() + splitIndex,
                    node->primitives.end()
            );
            node->primitives.resize(splitIndex);  //reuse as left child primitives
            node->primitives.shrink_to_fit();

            Vector3 leftMin, leftMax, rightMin, rightMax;
            findBounds(node->primitives.begin(), node->primitives.end(), leftMin, leftMax);
            findBounds(rightNodes.begin(), rightNodes.end(), rightMin, rightMax);

            node->children.emplace_back(BVHNode{AABB{leftMin, leftMax}, node->primitives, {}});
            node->children.emplace_back(BVHNode{AABB{rightMin, rightMax}, std::move(rightNodes), {}});
            node->primitives.clear();

            //Put children on stack
            for (auto &child: node->children) {
                nodes.push_back(&child);
            }
        }

        return StackBVH{std::move(root)};
    }

    //let nodes absorb their children
    static void collapse(StackBVH &bvh) {
        std::vector<BVHNode*> stack;
        stack.reserve(64);  //should min be ~ log(primitives.size())
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

            if (!node->primitives.empty()) {
                for (const Primitive* primitive : node->primitives) {
                    if (primitive->intersect(ray)) {
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

