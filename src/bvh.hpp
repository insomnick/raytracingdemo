#ifndef RAYTRACINGDEMO_BVH_HPP
#define RAYTRACINGDEMO_BVH_HPP

#include <vector>
#include <limits>
#include <algorithm>
#include "aabb.hpp"
#include "primitives/primitive.hpp"

struct HitInfo { bool hit=false; float t=0.f; Vector3 normal{0,0,0}; };

class BVH {
    struct Node {
        Vector3 minB{0,0,0};
        Vector3 maxB{0,0,0};
        Vector3 center{0,0,0}; // precomputed for ordering
        int left=-1;
        int right=-1;
        int firstPrim=0;
        int primCount=0;
        bool isLeaf() const { return primCount>0; }
    };
    std::vector<Node> nodes;
    std::vector<Primitive*> prims;

    inline bool boxIntersect(const Node& n, const Ray& r, float bestT) const {
        float tmin = ( (r.invDir.getX()>=0? n.minB.getX(): n.maxB.getX()) - r.origin.getX()) * r.invDir.getX();
        float tmax = ( (r.invDir.getX()>=0? n.maxB.getX(): n.minB.getX()) - r.origin.getX()) * r.invDir.getX();
        float tymin = ( (r.invDir.getY()>=0? n.minB.getY(): n.maxB.getY()) - r.origin.getY()) * r.invDir.getY();
        float tymax = ( (r.invDir.getY()>=0? n.maxB.getY(): n.minB.getY()) - r.origin.getY()) * r.invDir.getY();
        if ((tmin > tymax) || (tymin > tmax)) return false;
        if (tymin > tmin) tmin = tymin; if (tymax < tmax) tmax = tymax;
        float tzmin = ( (r.invDir.getZ()>=0? n.minB.getZ(): n.maxB.getZ()) - r.origin.getZ()) * r.invDir.getZ();
        float tzmax = ( (r.invDir.getZ()>=0? n.maxB.getZ(): n.minB.getZ()) - r.origin.getZ()) * r.invDir.getZ();
        if ((tmin > tzmax) || (tzmin > tmax)) return false;
        if (tzmin > tmin) tmin = tzmin; if (tzmax < tmax) tmax = tzmax;
        if (tmax < 0.f) return false;
        return tmin <= bestT;
    }

public:
    BVH() = default;

    static BVH build(std::vector<std::unique_ptr<Primitive>>& objects, int leafSize=8){
        std::vector<Primitive*> primPtrs; primPtrs.reserve(objects.size());
        for(auto& p: objects) primPtrs.push_back(p.get());
        BVH b; b.prims = primPtrs;
        b.nodes.reserve(std::max<size_t>(1, primPtrs.size()/leafSize*2));
        b.buildRecursive(0,(int)b.prims.size(), leafSize);
        return b;
    }

    HitInfo traverse(const Ray& r) const {
        HitInfo best; float bestT = std::numeric_limits<float>::infinity();
        if(nodes.empty()) return best;
        int stack[64]; int sp=0; stack[sp++] = 0;
        while(sp){
            const int idx = stack[--sp];
            const Node& n = nodes[idx];
            if(!boxIntersect(n,r,bestT)) continue;
            if(n.isLeaf()){
                Primitive* const* leafPrim = prims.data() + n.firstPrim;
                for(int i=0;i<n.primCount;++i){ float t; Vector3 N; if(leafPrim[i]->intersect(r,t,N) && t < bestT){ best.hit=true; bestT=t; best.t=t; best.normal=N; } }
            } else {
                if(n.left>=0 && n.right>=0){
                    const Node& L = nodes[n.left];
                    const Node& R = nodes[n.right];
                    float dL = Vector3::dot(L.center - r.origin, r.direction);
                    float dR = Vector3::dot(R.center - r.origin, r.direction);
                    if(dL < dR){ if(sp<63) stack[sp++] = n.right; if(sp<63) stack[sp++] = n.left; }
                    else { if(sp<63) stack[sp++] = n.left; if(sp<63) stack[sp++] = n.right; }
                } else { if(n.left>=0 && sp<64) stack[sp++] = n.left; if(n.right>=0 && sp<64) stack[sp++] = n.right; }
            }
        }
        return best;
    }

private:
    int buildRecursive(int start, int end, int leafSize){
        Node node;
        Vector3 minB( std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        Vector3 maxB(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        for(int i=start;i<end;++i){ auto* p=prims[i]; Vector3 mn=p->getMin(); Vector3 mx=p->getMax();
            minB = { std::min(minB.getX(), mn.getX()), std::min(minB.getY(), mn.getY()), std::min(minB.getZ(), mn.getZ())};
            maxB = { std::max(maxB.getX(), mx.getX()), std::max(maxB.getY(), mx.getY()), std::max(maxB.getZ(), mx.getZ())}; }
        node.minB = minB; node.maxB = maxB; node.center = (minB + maxB) * 0.5f;
        int count = end-start;
        if(count <= leafSize || count <= 2){
            node.firstPrim = start; node.primCount = count;
            int idx=(int)nodes.size(); nodes.push_back(node); return idx;
        }
        float mean[3] = {0.f,0.f,0.f};
        for(int i=start;i<end;++i){ Vector3 c = prims[i]->getCenter(); mean[0]+=c.getX(); mean[1]+=c.getY(); mean[2]+=c.getZ(); }
        float invCount = 1.f / count; mean[0]*=invCount; mean[1]*=invCount; mean[2]*=invCount;
        float var[3] = {0.f,0.f,0.f};
        for(int i=start;i<end;++i){ Vector3 c = prims[i]->getCenter(); float dx=c.getX()-mean[0]; float dy=c.getY()-mean[1]; float dz=c.getZ()-mean[2]; var[0]+=dx*dx; var[1]+=dy*dy; var[2]+=dz*dz; }
        int axis = 0; if(var[1]>var[axis]) axis=1; if(var[2]>var[axis]) axis=2;
        int mid = start + count/2;
        std::nth_element(prims.begin()+start, prims.begin()+mid, prims.begin()+end, [axis](Primitive* a, Primitive* b){ return a->getCenter().getAxis(axis) < b->getCenter().getAxis(axis); });
        int idx = (int)nodes.size(); nodes.push_back(node);
        int leftIdx = buildRecursive(start, mid, leafSize);
        int rightIdx = buildRecursive(mid, end, leafSize);
        nodes[idx].left = leftIdx; nodes[idx].right = rightIdx; return idx;
    }
};

#endif //RAYTRACINGDEMO_BVH_HPP
