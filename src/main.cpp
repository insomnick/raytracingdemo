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
#include "utils/object_loader.hpp"
#include "utils/timer.hpp"

struct Hit {
    bool hit = false;
    Vector3 position{};
    Vector3 normal{};
};

static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

static std::vector<Hit> ray_hits(SCREEN_WIDTH * SCREEN_HEIGHT); //positions and normals of hits
static std::array<std::array<Color, SCREEN_HEIGHT>, SCREEN_WIDTH> screen; // pixel colors

std::vector<std::unique_ptr<Primitive>> objects; //512 primitives in the scene
BVH bvh = BVH::stupidConstruct(objects);
Camera camera{SCREEN_WIDTH, SCREEN_HEIGHT};
Timer timer;

void drawScreen();
double mapToScreen(int j, const int height);

void calculateScreen(BVH& bvh);
static void shadeScreen(const Vector3& camera_pos);

void setupScene();

int main(void) {

    setupScene();

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

    //Build BVH time calculation
    timer.reset();
    bvh = BVH::medianSplitConstruction(objects);
    //bvh = BVH::stupidConstruct(objects);
    printf("Time build BVH using Median Split: %f \n", timer.elapsed());

    //App loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        timer.reset();
        calculateScreen(bvh);
        printf("Time calculate Screen: %f \n", timer.elapsed());

        timer.reset();
        shadeScreen(camera.getPosition());     // lighting pass
        printf("Time shade Screen: %f \n", timer.elapsed());

        timer.reset();
        drawScreen();   //just screen drawing (out of scope for now)
        printf("Time draw Screen: %f \n", timer.elapsed());

        //GLFW magic for screen drawing and events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

void setupScene() {

    std::vector<Triangle> loaded_object;
    //auto sponza = ObjectLoader::loadFromFile("../example/sponza.obj", 1.0);
    //loaded_object.insert(loaded_object.end(), sponza.begin(), sponza.end());
    //auto bunny = ObjectLoader::loadFromFile("../example/stanford-bunny.obj", 30.0);
    //loaded_object.insert(loaded_object.end(), bunny.begin(), bunny.end());
    //auto teapot = ObjectLoader::loadFromFile("../example/teapot.obj", 1.0);
    //loaded_object.insert(loaded_object.end(), teapot.begin(), teapot.end());
    auto suzanne = ObjectLoader::loadFromFile("../example/suzanne.obj", 3.0);
    loaded_object.insert(loaded_object.end(), suzanne.begin(), suzanne.end());
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
            //printf("Calculating pixel (%d, %d)\n", i, j);
            const double py = camera.getPixelY(j);
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
