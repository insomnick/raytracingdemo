#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <array>
#include <cmath>
#include <memory>
#include <algorithm>

#include "color.hpp"
#include "camera.hpp"
#include "primitives/sphere.hpp"
#include "primitives/ray.hpp"
#include "bvh.hpp"
#include "object_loader.hpp"

static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

//Building Screen with Color
//An Array with Dimensions of Screen and color
Color s[SCREEN_WIDTH][SCREEN_HEIGHT];
const int obj_size = 3;
std::vector<std::unique_ptr<Primitive>> objects; //512 primitives in the scene
BVH bvh = BVH::stupidConstruct(objects);


Camera camera{};

void drawScreen();
double mapToScreen(int j, const int height);

void calculateScreen(BVH& bvh);

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

    std::vector<Triangle> loaded_objects = ObjectLoader::loadFromFile("../example/stanford-bunny.obj", 10.0);
    //save in objects vector
    objects.reserve(loaded_objects.size());
    for (auto & obj : loaded_objects) {
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
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the framebuffer
        //Time Calculate
        double previous_seconds_c = glfwGetTime();
        calculateScreen(bvh);  //Actual Calculations
        double current_seconds_c = glfwGetTime();
        double elapsed_seconds_c = current_seconds_c - previous_seconds_c;
        printf("Time calculate Screen: %f \n", elapsed_seconds_c);

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
    //Pre-calculate constants outside loops
    const auto camera_pos = camera.getPosition();
    const auto camera_dir = camera.getDirection();
    const auto fov_half_tan = std::tan(camera.getFov() * 0.5);
    const auto aspect_ratio = static_cast<double>(SCREEN_WIDTH) / SCREEN_HEIGHT;
    const auto inv_width = 1.0 / SCREEN_WIDTH;
    const auto inv_height = 1.0 / SCREEN_HEIGHT;
    // Calculate proper camera basis vectors from camera direction
    const auto world_up = Vector3{0.0, 1.0, 0.0};
    const auto camera_right = Vector3::cross(camera_dir, world_up).normalize();
    const auto camera_up = Vector3::cross(camera_right, camera_dir).normalize();

    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        const auto px_base = (2.0 * (i + 0.5) * inv_width - 1.0) * fov_half_tan * aspect_ratio;
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            const auto py = (1.0 - 2.0 * (j + 0.5) * inv_height) * fov_half_tan;
            auto ray_direction = camera_dir + (camera_up * py) + (camera_right * px_base);
            // Fast normalize using reciprocal
            const auto inv_length = 1.0 / ray_direction.length();
            ray_direction = ray_direction * inv_length;
            const Ray ray{camera_pos, ray_direction};

            //This is the place where I can do the BVH acceleration structure
            //Traverse BVH and get intersection point and normal, then calc color
            auto intersection = bvh.traverse(ray);
            if (intersection != nullptr) {
                Vector3 hitPos = intersection->getOrigin();
                Vector3 normal = intersection->getDirection();
                double normal_length = normal.length();
                if (normal_length > 0.0) normal = normal * (1.0 / normal_length);

                Vector3 light_direction = camera_pos - hitPos;  //camera pos as light source
                double light_distance = light_direction.length();
                if (light_distance > 0.0) light_direction = light_direction * (1.0 / light_distance);

                // Lighting terms
                const double ambient = 0.12;
                double diffuse = std::max(0.0, Vector3::dot(normal, light_direction));
                double attenuation = 1.0 / (1.0 + 0.35 * light_distance * light_distance); // quadratic falloff
                double intensity = ambient + diffuse * attenuation;
                intensity = std::clamp(intensity, 0.0, 1.0);

                // Base color encodes normal (for visibility) then scaled by light
                double nx = 0.5 * (normal.getX() + 1.0);
                double ny = 0.5 * (normal.getY() + 1.0);
                double nz = 0.5 * (normal.getZ() + 1.0);
                s[i][j] = Color(nx * intensity,
                                ny * intensity,
                                nz * intensity);
            } else {
                s[i][j] = Color(0.0, 0.0, 0.0); // Blank
            }
        }
    }
}


void drawScreen() {
    glBegin(GL_POINTS);
    /* Render here */
    for(int i = 0; i < SCREEN_WIDTH; i++)  {
        for(int j = 0; j < SCREEN_HEIGHT; j++) {
            Color c = s[i][j];
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
