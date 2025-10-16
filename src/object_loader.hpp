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
        objl::Loader loader;
        if (!loader.LoadFile(path)) {
            throw std::runtime_error("Failed to load OBJ file: " + path);
        }
        std::vector<Triangle> triangles;
        // First pass gather bounds if centering or normalizing
        bool needBounds = centerAndNormalize;
        Vector3 minB{ std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), std::numeric_limits<double>::max() };
        Vector3 maxB{ std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest(), std::numeric_limits<double>::lowest() };
        if (needBounds) {
            for (const auto& mesh : loader.LoadedMeshes) {
                for (const auto& v : mesh.Vertices) {
                    minB = { std::min(minB.getX(), static_cast<double>(v.Position.X)), std::min(minB.getY(), static_cast<double>(v.Position.Y)), std::min(minB.getZ(), static_cast<double>(v.Position.Z)) };
                    maxB = { std::max(maxB.getX(), static_cast<double>(v.Position.X)), std::max(maxB.getY(), static_cast<double>(v.Position.Y)), std::max(maxB.getZ(), static_cast<double>(v.Position.Z)) };
                }
            }
        }
        Vector3 center{0,0,0};
        double normScale = 1.0;
        if (needBounds) {
            center = (minB + maxB) * 0.5;
            Vector3 size = maxB - minB;
            double maxExtent = std::max({size.getX(), size.getY(), size.getZ()});
            if (maxExtent > 0.0 && targetExtent > 0.0) normScale = targetExtent / maxExtent;
        }
        for (const auto& mesh : loader.LoadedMeshes) {
            const auto& verts = mesh.Vertices;
            const auto& idx = mesh.Indices;
            if (idx.empty()) {
                size_t triCount = verts.size() / 3;
                triangles.reserve(triangles.size() + triCount);
                for (size_t i = 0; i + 2 < verts.size(); i += 3) {
                    Vector3 v0{verts[i].Position.X, verts[i].Position.Y, verts[i].Position.Z};
                    Vector3 v1{verts[i+1].Position.X, verts[i+1].Position.Y, verts[i+1].Position.Z};
                    Vector3 v2{verts[i+2].Position.X, verts[i+2].Position.Y, verts[i+2].Position.Z};
                    if (needBounds) { v0 = (v0 - center) * normScale; v1 = (v1 - center) * normScale; v2 = (v2 - center) * normScale; }
                    triangles.emplace_back(v0 * scale, v1 * scale, v2 * scale);
                }
            } else {
                size_t triCount = idx.size() / 3;
                triangles.reserve(triangles.size() + triCount);
                for (size_t i = 0; i + 2 < idx.size(); i += 3) {
                    unsigned int i0 = idx[i];
                    unsigned int i1 = idx[i+1];
                    unsigned int i2 = idx[i+2];
                    if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;
                    Vector3 v0{verts[i0].Position.X, verts[i0].Position.Y, verts[i0].Position.Z};
                    Vector3 v1{verts[i1].Position.X, verts[i1].Position.Y, verts[i1].Position.Z};
                    Vector3 v2{verts[i2].Position.X, verts[i2].Position.Y, verts[i2].Position.Z};
                    if (needBounds) { v0 = (v0 - center) * normScale; v1 = (v1 - center) * normScale; v2 = (v2 - center) * normScale; }
                    triangles.emplace_back(v0 * scale, v1 * scale, v2 * scale);
                }
            }
        }
        return triangles;
    }
};

#endif //RAYTRACINGDEMO_OBJECT_LOADER_HPP
