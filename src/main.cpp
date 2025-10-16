#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <array>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>

#include "color.hpp"
#include "camera.hpp"
#include "primitives/sphere.hpp"
#include "primitives/ray.hpp"
#include "bvh.hpp"
#include "object_loader.hpp"

static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

struct Hit { bool hit=false; Vector3 position{}; Vector3 normal{}; };
static std::vector<Hit> ray_hits(SCREEN_WIDTH * SCREEN_HEIGHT); //positions and normals of hits
static std::array<std::array<Color, SCREEN_HEIGHT>, SCREEN_WIDTH> screen; // pixel colors

const int obj_size = 3;
std::vector<std::unique_ptr<Primitive>> objects; //512 primitives in the scene
BVH bvh; // built later

Camera camera{SCREEN_WIDTH, SCREEN_HEIGHT};

void drawScreen();
double mapToScreen(int j, const int height);

void calculateScreen(BVH& bvh);
static void shadeScreen(const Vector3& camera_pos);

void setupScene();

static bool cameraDirty = true; // mark when camera changed
#ifdef USE_DIR_CACHE
static std::vector<Vector3> rayDirs(SCREEN_WIDTH * SCREEN_HEIGHT);
static void updateRayDirs(){
    const Vector3 camera_dir = camera.getDirection();
    const Vector3 world_up{0.0f, 1.0f, 0.0f};
    Vector3 right = Vector3::cross(camera_dir, world_up); if(right.length()<1e-6f) right = {0,0,1}; right = right.normalize();
    Vector3 up = Vector3::cross(right, camera_dir).normalize();

#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<SCREEN_WIDTH;++i){
        float pxBase = (float)camera.getPixelX(i);
        for(int j=0;j<SCREEN_HEIGHT;++j){
            float py = (float)camera.getPixelY(j);
            Vector3 dir = (camera_dir + up * py + right * pxBase).normalize();
            rayDirs[j + i*SCREEN_HEIGHT] = dir;
        }
    }
}
#endif

#ifdef FAST_SINGLE_PASS
static void renderScreenSinglePass(BVH& bvh) {
    const Vector3 camera_pos = camera.getPosition();
#ifndef USE_DIR_CACHE
    const Vector3 camera_dir = camera.getDirection();
    const Vector3 world_up{0.0f, 1.0f, 0.0f};
    Vector3 right = Vector3::cross(camera_dir, world_up); if(right.length()<1e-6f) right = {0,0,1}; right = right.normalize();
    Vector3 up = Vector3::cross(right, camera_dir).normalize();
#endif
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic,4)
#endif
    for (int i = 0; i < SCREEN_WIDTH; ++i) {
#ifndef USE_DIR_CACHE
        const float pxBase = (float)camera.getPixelX(i);
#endif
        Color* row = screen[i].data();
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            const int idx = j + i*SCREEN_HEIGHT;
#ifdef USE_DIR_CACHE
            const Vector3& dir = rayDirs[idx];
#else
            const float py = (float)camera.getPixelY(j);
            Vector3 dir = (camera_dir + up * py + right * pxBase).normalize();
#endif
            Ray ray{camera_pos, dir};
            HitInfo h = bvh.traverse(ray);
            if(!h.hit){ row[j] = Color(0.f,0.f,0.f); continue; }
            Vector3 N = h.normal; float nl2 = N.length2(); if(nl2>0) N = N * (1.0f/std::sqrt(nl2));
            Vector3 pos = ray.origin + dir * h.t;
            Vector3 L = camera_pos - pos; float dl2 = L.length2(); float dist=0.f; if(dl2>0){ dist = std::sqrt(dl2); L = L * (1.0f/dist);}
            float ndotl = Vector3::dot(N,L); if(ndotl<0) ndotl=0; // diffuse
            float attenuation = 1.0f/(1.0f+0.02f*dist*dist);
            float intensity = 0.25f + ndotl*attenuation; if(intensity>1.f) intensity=1.f; // clamp
            float nx=0.5f*(N.getX()+1.0f), ny=0.5f*(N.getY()+1.0f), nz=0.5f*(N.getZ()+1.0f);
            row[j] = Color(intensity*nx, intensity*ny, intensity*nz);
        }
    }
}
#endif // FAST_SINGLE_PASS

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
        if (key == GLFW_KEY_I && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.rotateHorizontal(0.1); cameraDirty = true; }
        if (key == GLFW_KEY_K && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.rotateHorizontal(-0.1); cameraDirty = true; }
        if (key == GLFW_KEY_J && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.rotateVertical(-0.1); cameraDirty = true; }
        if (key == GLFW_KEY_L && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.rotateVertical(0.1); cameraDirty = true; }
        if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.moveForward(1.0); cameraDirty = true; }
        if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.moveForward(-1.0); cameraDirty = true; }
        if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.moveRight(-0.1); cameraDirty = true; }
        if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.moveRight(0.1); cameraDirty = true; }
        if (key == GLFW_KEY_Q && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.moveUp(0.1); cameraDirty = true; }
        if (key == GLFW_KEY_E && (action == GLFW_PRESS || action == GLFW_REPEAT)) { camera.moveUp(-0.1); cameraDirty = true; }
        if (key == GLFW_KEY_M && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            printf(" Rebuilding BVH (leafSize=8) ...\n");
            bvh = BVH::build(objects, 8);
        }
        if (key == GLFW_KEY_N && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            printf(" Rebuilding BVH (leafSize=4) ...\n");
            bvh = BVH::build(objects, 4);
        }
    });

    //Build BVH time calculation
    double previous_seconds_bvh = glfwGetTime();
    bvh = BVH::build(objects, 8);
    double current_seconds_bvh = glfwGetTime();
    double elapsed_seconds_bvh = current_seconds_bvh - previous_seconds_bvh;
    printf("Time build BVH (leafSize=8): %f \n", elapsed_seconds_bvh);

    //App loop
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
#ifdef USE_DIR_CACHE
        if(cameraDirty){ updateRayDirs(); cameraDirty=false; }
#endif
#ifdef FAST_SINGLE_PASS
        double t0 = glfwGetTime();
        renderScreenSinglePass(bvh);
        double t1 = glfwGetTime();
#ifdef ENABLE_TIMINGS
        printf("Time single-pass render: %f \n", (t1-t0));
#endif
#else
        double previous_seconds_c = glfwGetTime();
        calculateScreen(bvh);
        double current_seconds_c = glfwGetTime();
#ifdef ENABLE_TIMINGS
        printf("Time calculate Screen: %f \n", (current_seconds_c-previous_seconds_c));
#endif
        double previous_seconds_s = glfwGetTime();
        shadeScreen(camera.getPosition());
        double current_seconds_s = glfwGetTime();
#ifdef ENABLE_TIMINGS
        printf("Time shade Screen: %f \n", (current_seconds_s-previous_seconds_s));
#endif
#endif
        double previous_seconds = glfwGetTime();
        drawScreen();
        double current_seconds = glfwGetTime();
#ifdef ENABLE_TIMINGS
        printf("Time draw Screen: %f \n", (current_seconds-previous_seconds));
#endif
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

void setupScene() {
    // Load primitives
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
    auto sponza = ObjectLoader::loadFromFile("../example/sponza.obj", 1.0);
    loaded_object.insert(loaded_object.end(), sponza.begin(), sponza.end());
    //auto bunny = ObjectLoader::loadFromFile("../example/stanford-bunny.obj", 30.0);
    //loaded_object.insert(loaded_object.end(), bunny.begin(), bunny.end());
    //auto teapot = ObjectLoader::loadFromFile("../example/teapot.obj", 1.0);
    //loaded_object.insert(loaded_object.end(), teapot.begin(), teapot.end());
    //auto suzanne = ObjectLoader::loadFromFile("../example/suzanne.obj", 3.0);
    //loaded_object.insert(loaded_object.end(), suzanne.begin(), suzanne.end());
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
    const Vector3 world_up{0.0f, 1.0f, 0.0f};
    Vector3 right = Vector3::cross(camera_dir, world_up);
    if (right.length() < 1e-6f) right = Vector3{0.0f, 0.0f, 1.0f};
    right = right.normalize();
    Vector3 up = Vector3::cross(right, camera_dir).normalize();
#ifdef USE_OMP
#pragma omp parallel for schedule(dynamic,4)
#endif
    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        float px = (float)camera.getPixelX(i);
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            float py = (float)camera.getPixelY(j);
            Vector3 dir = (camera_dir + up * py + right * px).normalize();
            Ray ray{camera_pos, dir};
            HitInfo h = bvh.traverse(ray);
            int idx = j + i * SCREEN_HEIGHT;
            if (h.hit) {
                ray_hits[idx].hit = true;
                ray_hits[idx].normal = h.normal;
                ray_hits[idx].position = ray.origin + dir * h.t;
            } else {
                ray_hits[idx].hit = false;
            }
        }
    }
}

static void shadeScreen(const Vector3& camera_pos) {
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            int idx = j + i * SCREEN_HEIGHT;
            const Hit& h = ray_hits[idx];
            if (!h.hit) { screen[i][j] = Color(0.0,0.0,0.0); continue; }
            // basic diffuse shading toward camera as point light at camera position
            Vector3 N = h.normal; double nl2 = N.length2(); if(nl2>0) N = N * (1.0/std::sqrt(nl2));
            Vector3 L = (camera_pos - h.position); double dl2 = L.length2(); double dist=0; if(dl2>0){ dist= std::sqrt(dl2); L = L * (1.0/dist); }
            double ambient = 0.25;
            double ndotl = Vector3::dot(N, L); if(ndotl < 0.0) ndotl = 0.0; // one-sided lighting
            double attenuation = 1.0 / (1.0 + 0.02 * dist * dist);
            double intensity = std::clamp(ambient + ndotl * attenuation, 0.0, 1.0);
            // visualize normal (mapped 0..1) modulated by intensity
            double nx = 0.5 * (N.getX() + 1.0);
            double ny = 0.5 * (N.getY() + 1.0);
            double nz = 0.5 * (N.getZ() + 1.0);
            screen[i][j] = Color(intensity * nx, intensity * ny, intensity * nz);
        }
    }
}


void drawScreen() {
    glBegin(GL_POINTS);
    for(int i = 0; i < SCREEN_WIDTH; i++)  {
        for(int j = 0; j < SCREEN_HEIGHT; j++) {
            const Color& c = screen[i][j];
            glColor3f(c.r(),c.g(),c.b());
            glVertex2d(mapToScreen(i, SCREEN_WIDTH),
                       mapToScreen(SCREEN_HEIGHT - 1 - j, SCREEN_HEIGHT));
        }
    }
    glEnd();
}

double mapToScreen(int j, const int height) {
return ((2.0*j)/height) - 1.0;
}
