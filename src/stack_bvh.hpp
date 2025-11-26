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

    static void findBounds(const std::vector<Primitive *> &primitives, Vector3& min, Vector3& max) {
        //calculate bounds of root
        auto minimumX = std::numeric_limits<double>::max();
        auto minimumY = std::numeric_limits<double>::max();
        auto minimumZ = std::numeric_limits<double>::max();
        auto maximumX = std::numeric_limits<double>::lowest();
        auto maximumY = std::numeric_limits<double>::lowest();
        auto maximumZ = std::numeric_limits<double>::lowest();
        for (auto primitive: primitives) {
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

    // binary BVH construction with lambda for splitting
    template<typename PartitionFunction>
    static StackBVH binaryBuild(std::vector<Primitive*>& primitives, PartitionFunction partitionFunction) {
        if (primitives.empty()) {
            return StackBVH{BVHNode{AABB{{0,0,0}, {0,0,0}}, {}, {}}};
        }

        Vector3 globalMin;
        Vector3 globalMax;
        findBounds(primitives, globalMin, globalMax);
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
            );;
            node->primitives.resize(splitIndex);  //reuse as left child primitives
            node->primitives.shrink_to_fit();

            Vector3 leftMin, leftMax, rightMin, rightMax;
            findBounds(node->primitives, leftMin, leftMax);
            findBounds(rightNodes, rightMin, rightMax);

            node->children.emplace_back(BVHNode{AABB{leftMin, leftMax}, node->primitives, {}});
            node->children.emplace_back(BVHNode{AABB{rightMin, rightMax}, rightNodes, {}});
            node->primitives.clear();

            //Put children on stack
            for (auto &child: node->children) {
                nodes.push_back(&child);
            }
        }

        return StackBVH{root};
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

