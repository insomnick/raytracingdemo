#ifndef RAYTRACINGDEMO_OBJECT_LOADER_HPP
#define RAYTRACINGDEMO_OBJECT_LOADER_HPP

#include "lib/OBJ_Loader.h"
#include "vector3.hpp"
#include "triangle.hpp"

class ObjectLoader {
public:
    static std::vector<Triangle> loadFromFile(const std::string& path) {
        //Credits to Robert Smith
        //https://github.com/Bly7/OBJ-Loader
        objl::Loader loader;
        if (!loader.LoadFile(path)) {
            throw std::runtime_error("Failed to load OBJ file: " + path);
        }
        std::vector<Triangle> triangles;
        for (const auto& mesh : loader.LoadedMeshes) {
            const auto& vertices = mesh.Vertices;
            for (size_t i = 0; i + 2 < vertices.size(); i += 3) {
                Vector3 v0{vertices[i].Position.X, vertices[i].Position.Y, vertices[i].Position.Z};
                Vector3 v1{vertices[i + 1].Position.X, vertices[i + 1].Position.Y, vertices[i + 1].Position.Z};
                Vector3 v2{vertices[i + 2].Position.X, vertices[i + 2].Position.Y, vertices[i + 2].Position.Z};
                triangles.emplace_back(v0, v1, v2);
            }
        }
        return triangles;
    }
};

#endif //RAYTRACINGDEMO_OBJECT_LOADER_HPP
