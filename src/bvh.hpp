#ifndef RAYTRACINGDEMO_BVH_HPP
#define RAYTRACINGDEMO_BVH_HPP

#include <utility>
#include <vector>
#include <algorithm>
#include <limits>
#include "aabb.hpp"
#include "primitives/primitive.hpp"

class BVH {
private:
    struct BVHNode {
        BVHNode(AABB box, size_t primitives_left, size_t primitives_right, size_t children_left, size_t children_right) : box(std::move(box)), primitives_left(primitives_left), primitives_right(primitives_right), children_left(children_left), children_right(children_right) {}
        AABB box;
        size_t primitives_left; //inclusive
        size_t primitives_right; //exclusive
        size_t children_left; //inclusive
        size_t children_right; //exclusive
    };

    std::vector<BVHNode> nodes;
    std::vector<Primitive*> primitives;

public:

    static BVH medianConstruction(const std::vector<Primitive*>& primitives, int degree = 2) {
        BVH bvh;
        if (primitives.empty()) return bvh;
        bvh.primitives = primitives;

        //TODO just binary for now
        BVHNode root_node{AABB(primitives, 0, 0), 0, primitives.size(), 1, 3}; //Todo + degree
        bvh.nodes.emplace_back(root_node);

        size_t index = 0;
        while (index < bvh.nodes.size()) {
            BVHNode &current = bvh.nodes[index];
            size_t left = current.primitives_left;
            size_t right = current.primitives_right;
            size_t count = right - left;
            if (count <= 1) {
                current.children_left = -1;
                current.children_right = -1;
                ++index;
                continue;
            }

            int axis = current.box.getLongestAxis();
            size_t split_pos = left + count / 2;
            //if (split_pos <= left || split_pos >= right) { ++index; continue; }

            std::nth_element(bvh.primitives.begin() + left,
                             bvh.primitives.begin() + split_pos,
                             bvh.primitives.begin() + right,
                             [axis](Primitive *a, Primitive *b) {
                                 return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis);
                             });

            bvh.nodes.emplace_back(AABB(bvh.primitives, left, split_pos), left, split_pos, bvh.primitives.size(), bvh.primitives.size() + 2);
            bvh.nodes.emplace_back(AABB(bvh.primitives, split_pos, right), split_pos, right, bvh.primitives.size(), bvh.primitives.size() + 2);
            ++index;
        }
        return bvh;
    }

    std::unique_ptr<Ray> traverse(Ray& ray){
        std::vector<size_t> stack;
        stack.push_back(0); //start with root node

        std::unique_ptr<Ray> closest_hit = nullptr;
        double closest_t = std::numeric_limits<double>::infinity();

        while (!stack.empty()) {
            size_t node_index = stack.back();
            stack.pop_back();
            const BVHNode& node = nodes[node_index];

            if (!node.box.hit(ray)) {
                continue; // No intersection with this node's AABB
            }

            // Leaf node
            if (node.children_left == -1 || node.children_right == -1) {
                for (size_t i = node.primitives_left; i < node.primitives_right; ++i) {
                    std::unique_ptr<Ray> hit = primitives[i]->getIntersectionNormalAndDirection(ray);
                    if (hit) {
                        double t = (hit->getOrigin() - ray.getOrigin()).length();
                        if (t < closest_t) {
                            closest_t = t;
                            closest_hit = std::move(hit);
                        }
                    }
                }
            } else {
                // Push child nodes onto the stack
                stack.push_back(node.children_right);
                stack.push_back(node.children_left);
            }
        }

        return closest_hit;
    }
};

#endif //RAYTRACINGDEMO_BVH_HPP
