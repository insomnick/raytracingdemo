#ifndef RAYTRACINGDEMO_OBJECT_LOADER_HPP
#define RAYTRACINGDEMO_OBJECT_LOADER_HPP

#include <limits>
#include "../lib/OBJ_Loader.h"
#include "primitives/vector3.hpp"
#include "primitives/triangle.hpp"

class ObjectLoader {
public:
    //Credits to Robert Smith
    //https://github.com/Bly7/OBJ-Loader
    //Created by AI with some modifications by me to handle translation to domain
    static std::vector<Triangle> loadFromFile(const std::string& path, const double scale = 1.0, bool centerAndNormalize = false, double targetExtent = 1.0) {
        objl::Loader loader; if(!loader.LoadFile(path)) throw std::runtime_error("Failed to load OBJ file: "+path);
        std::vector<Triangle> triangles; bool needBounds = centerAndNormalize;
        Vector3 minB( std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity());
        Vector3 maxB(-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity());
        if(needBounds){ for(const auto& mesh: loader.LoadedMeshes) for(const auto& v: mesh.Vertices){ minB = { std::min(minB.getX(), v.Position.X), std::min(minB.getY(), v.Position.Y), std::min(minB.getZ(), v.Position.Z)}; maxB = { std::max(maxB.getX(), v.Position.X), std::max(maxB.getY(), v.Position.Y), std::max(maxB.getZ(), v.Position.Z)}; } }
        Vector3 center(0,0,0); float normScale=1.f; if(needBounds){ center = (minB+maxB)*0.5f; Vector3 size = maxB - minB; float maxE = std::max({size.getX(), size.getY(), size.getZ()}); if(maxE>0 && targetExtent>0) normScale = (float)targetExtent / maxE; }
        for(const auto& mesh: loader.LoadedMeshes){ const auto& verts=mesh.Vertices; const auto& idx=mesh.Indices; if(idx.empty()){
            for(size_t i=0;i+2<verts.size();i+=3){ Vector3 v0(verts[i].Position.X,verts[i].Position.Y,verts[i].Position.Z); Vector3 v1(verts[i+1].Position.X,verts[i+1].Position.Y,verts[i+1].Position.Z); Vector3 v2(verts[i+2].Position.X,verts[i+2].Position.Y,verts[i+2].Position.Z); if(needBounds){ v0=(v0-center)*normScale; v1=(v1-center)*normScale; v2=(v2-center)*normScale;} float s=(float)scale; triangles.emplace_back(v0*s, v1*s, v2*s);} }
            else { for(size_t i=0;i+2<idx.size();i+=3){ unsigned i0=idx[i], i1=idx[i+1], i2=idx[i+2]; if(i0>=verts.size()||i1>=verts.size()||i2>=verts.size()) continue; Vector3 v0(verts[i0].Position.X,verts[i0].Position.Y,verts[i0].Position.Z); Vector3 v1(verts[i1].Position.X,verts[i1].Position.Y,verts[i1].Position.Z); Vector3 v2(verts[i2].Position.X,verts[i2].Position.Y,verts[i2].Position.Z); if(needBounds){ v0=(v0-center)*normScale; v1=(v1-center)*normScale; v2=(v2-center)*normScale;} float s=(float)scale; triangles.emplace_back(v0*s,v1*s,v2*s);} }
        }
        return triangles;
    }
};

#endif //RAYTRACINGDEMO_OBJECT_LOADER_HPP
