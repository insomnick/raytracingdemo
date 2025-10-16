#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <array>
#include <cmath>
#include <memory>
#include <algorithm>
#include <vector>

#include "color.hpp"
#include "camera.hpp"
#include "primitives/sphere.hpp"
#include "primitives/ray.hpp"
#include "bvh.hpp"
#include "object_loader.hpp"

static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

struct Hit {
    bool hit = false;
    Vector3 position{};
    Vector3 normal{};
};
static std::vector<Hit> ray_hits(SCREEN_WIDTH * SCREEN_HEIGHT); //positions and normals of hits
static std::array<std::array<Color, SCREEN_HEIGHT>, SCREEN_WIDTH> screen; // pixel colors

const int obj_size = 3;
std::vector<std::unique_ptr<Primitive>> objects; //512 primitives in the scene
BVH bvh = BVH::stupidConstruct(objects);


Camera camera{SCREEN_WIDTH, SCREEN_HEIGHT};

void drawScreen();
double mapToScreen(int j, const int height);

void calculateScreen(BVH& bvh);
static void shadeScreen(const Vector3& camera_pos);

int main(void) {
    //Init GLFW
    GLFWwindow* window;
    if (!glfwInit())
        return -1;
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "raytracing demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    if(!gladLoadGL())
        return -1;
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (key == GLFW_KEY_I && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateHorizontal(0.1);
            //printf("2: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_K && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateHorizontal(-0.1);
            //printf("8: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_J && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateVertical(-0.1);
            //printf("4: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_L && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateVertical(0.1);
            //printf("6: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.moveForward(0.1);
            //printf("W: Camera Position: %f, %f, %f\n", camera.getPosition().getX(), camera.getPosition().getY(), camera.getPosition().getZ());
        }
        if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.moveForward(-0.1);
            //printf("S: Camera Position: %f, %f, %f\n", camera.getPosition().getX(), camera.getPosition().getY(), camera.getPosition().getZ());
        }
        if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.moveRight(-0.1);
            //printf("A: Camera Position: %f, %f, %f\n", camera.getPosition().getX(), camera.getPosition().getY(), camera.getPosition().getZ());
        }
        if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
           camera.moveRight(0.1);
            //printf("D: Camera Position: %f, %f, %f\n", camera.getPosition().getX(), camera.getPosition().getY(), camera.getPosition().getZ());
        }
        if (key == GLFW_KEY_Q && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.moveUp(0.1);
            //printf("D: Camera Position: %f, %f, %f\n", camera.getPosition().getX(), camera.getPosition().getY(), camera.getPosition().getZ());
        }
        if (key == GLFW_KEY_E && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.moveUp(-0.1);
        }
        if (key == GLFW_KEY_M && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            printf(" Rebuilding BVH using Median Split...\n");
            bvh = BVH::medianSplitConstruction(objects);
        }
        if (key == GLFW_KEY_N && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            printf(" Rebuilding BVH using Stupid Split...\n");
            bvh = BVH::stupidConstruct(objects);
        }
    });
    //Load primitives
//    objects.reserve(obj_size * obj_size * obj_size);
//    for (int i = 0; i < obj_size; ++i) {
//        for (int j = 0; j < obj_size; ++j) {
//            for (int k = 0; k < obj_size; ++k) {
//                const auto x = i * 2.0;
//                const auto y = j * 2.0 - obj_size - 2.0;
//                const auto z = k * 2.0 - obj_size - 2.0;
//                objects.emplace_back(std::make_unique<Sphere>(x, y, z, 0.5));
//            }
//        }
//    }

    std::vector<Triangle> loaded_object;
    auto bunny = ObjectLoader::loadFromFile("../example/stanford-bunny.obj", 30.0);
    loaded_object.insert(loaded_object.end(), bunny.begin(), bunny.end());
    //auto teapot = ObjectLoader::loadFromFile("../example/teapot.obj", 1.0);
    //loaded_object.insert(loaded_object.end(), teapot.begin(), teapot.end());
    //BROKEN std::vector<Triangle> loaded_object = ObjectLoader::loadFromFile("../example/suzanne.obj", 5.0);
    printf("Loaded %zu triangles from OBJ file.\n", loaded_object.size());


    //calculate object center ( center of all centers
    Vector3 object_center{0.0, 0.0, 0.0};
    for (auto & obj : loaded_object) {
        object_center = object_center + obj.getCenter();
    }
    object_center = object_center * (1.0 / loaded_object.size());
    camera.setPosition(object_center + Vector3{0.0, 0.0, 5.0});


    //save in objects vector
    objects.reserve(loaded_object.size());
    for (auto & obj : loaded_object) {
        objects.push_back(std::make_unique<Triangle>(obj));
    }

    //Build BVH time calculation
    double previous_seconds_bvh = glfwGetTime();
    bvh = BVH::medianSplitConstruction(objects);
    //bvh = BVH::stupidConstruct(objects);
    double current_seconds_bvh = glfwGetTime();
    double elapsed_seconds_bvh = current_seconds_bvh - previous_seconds_bvh;
    printf("Time build BVH using Median Split: %f \n", elapsed_seconds_bvh);

    //for (auto & object : objects) {
    //    printf("Object Center: %f, %f, %f\n", object.getCenter().getX(), object.getCenter().getY(), object.getCenter().getZ());
    //}

    //App loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //Time Calculate
        double previous_seconds_c = glfwGetTime();
        calculateScreen(bvh);                  // ray generation + BVH traversal only
        double current_seconds_c = glfwGetTime();
        double elapsed_seconds_c = current_seconds_c - previous_seconds_c;
        printf("Time calculate Screen: %f \n", elapsed_seconds_c);


        //Time Shade
        double previous_seconds_s = glfwGetTime();
        shadeScreen(camera.getPosition());     // lighting pass
        double current_seconds_s = glfwGetTime();
        double elapsed_seconds_s = current_seconds_s - previous_seconds_s;
        printf("Time shade Screen: %f \n", elapsed_seconds_s);

        //Time Draw
        double previous_seconds = glfwGetTime();
        drawScreen();   //just screen drawing (out of scope for now)
        double current_seconds = glfwGetTime();
        double elapsed_seconds = current_seconds - previous_seconds;
        printf("Time draw Screen: %f \n", elapsed_seconds);

        //GLFW magic for screen drawing and events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

void calculateScreen(BVH& bvh) {
    const Vector3 camera_pos = camera.getPosition();
    const Vector3 camera_dir = camera.getDirection();
    const Vector3 world_up{0.0, 1.0, 0.0};
    Vector3 right = Vector3::cross(camera_dir, world_up);
    if (right.length() < 1e-8) right = Vector3{0.0, 0.0, 1.0};
    right = right.normalize();
    Vector3 up = Vector3::cross(right, camera_dir).normalize();

    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        const double px = camera.getPixelX(i);
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            const double py = camera.getPixelY(j); // FIX: was getPixelY(i)
            Vector3 dir = camera_dir + up * py + right * px;
            dir = dir * (1.0 / dir.length());
            Ray ray{camera_pos, dir};

            const int idx = j + i * SCREEN_HEIGHT;
            auto hitRay = bvh.traverse(ray); // Ray(origin=hitPos, direction=normal) or nullptr
            if (hitRay) {
                ray_hits[idx].hit = true;
                ray_hits[idx].position = hitRay->getOrigin();
                ray_hits[idx].normal = hitRay->getDirection();
            } else {
                ray_hits[idx].hit = false;
            }
        }
    }
}

static void shadeScreen(const Vector3& camera_pos) {
    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            const int idx = j + i * SCREEN_HEIGHT;
            const Hit& h = ray_hits[idx];
            if (!h.hit) {
                screen[i][j] = Color(0.0, 0.0, 0.0);
                continue;
            }
            Vector3 N = h.normal;
            double nl = N.length();
            if (nl > 0.0) N = N * (1.0 / nl);
            Vector3 L = camera_pos - h.position;
            double dist = L.length();
            if (dist > 0.0) L = L * (1.0 / dist);

            const double ambient = 0.45;
            double diffuse = std::max(0.0, Vector3::dot(N, L)) * 1.35;
            double attenuation = 1.0 / (1.0 + 0.05 * dist * dist);
            double intensity = std::clamp((ambient + diffuse * attenuation) * 1.25, 0.0, 1.0);

            double nx = 0.5 * (N.getX() + 1.0);
            double ny = 0.5 * (N.getY() + 1.0);
            double nz = 0.5 * (N.getZ() + 1.0);
            screen[i][j] = Color(nx * intensity, ny * intensity, nz * intensity);
        }
    }
}


void drawScreen() {
    glBegin(GL_POINTS);
    /* Render here */
    for(int i = 0; i < SCREEN_WIDTH; i++)  {
        for(int j = 0; j < SCREEN_HEIGHT; j++) {
            Color c = screen[i][j];
            glColor3d(c.r(),c.g(),c.b());
            // invert j so row 0 (treated as top in calculateScreen) is rendered at y=+1
            glVertex2d(mapToScreen(i, SCREEN_WIDTH),
                       mapToScreen(SCREEN_HEIGHT - 1 - j, SCREEN_HEIGHT));
        }
    }
    glEnd();
}

double mapToScreen(int j, const int height) {
return ((2.0*j)/height) - 1.0;
}
