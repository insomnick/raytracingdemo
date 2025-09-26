#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <valarray>
#include <array>
#include <cmath>
#include "color.hpp"
#include "camera.hpp"
#include "sphere.hpp"
#include "ray.hpp"
#include "bvh.hpp"

static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

//Building Screen with Color
//An Array with Dimensions of Screen and color
Color s[SCREEN_WIDTH][SCREEN_HEIGHT];
const int obj_size = 2;
std::vector<Sphere> objects; //512 primitives in the scene

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
    });
    //Load primitives
    objects.reserve(obj_size * obj_size * obj_size);
    for (int i = 0; i < obj_size; ++i) {
        for (int j = 0; j < obj_size; ++j) {
            for (int k = 0; k < obj_size; ++k) {
                const auto x = i * 2.0;
                const auto y = j * 2.0 - obj_size - 2.0;
                const auto z = k * 2.0 - obj_size - 2.0;
                objects.emplace_back(x, y, z, 0.5);
            }
        }
    }

    for (auto & object : objects) {
        printf("Object Center: %f, %f, %f\n", object.getCenter().getX(), object.getCenter().getY(), object.getCenter().getZ());
    }

    //Build BVH from primitives here
    //TODO: better BVH construction
    BVH bvh = BVH::stupidConstruct(objects);

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

    //TODO: make the whole screen buffer thing smarter
    Color ns[SCREEN_WIDTH][SCREEN_HEIGHT];
    // Initialize screen to black
    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            ns[i][j] = Color(0.0, 0.0, 0.0);
        }
    }

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
            //Traverse BVH and get intersection point, then calc color based on distance or Material
            auto intersection = bvh.traverse(ray);
            if (intersection != nullptr) {
                // Simple shading based on distance
                double distance = (*intersection - camera_pos).length();
                double intensity = std::exp(-0.1 * distance); // Exponential falloff
                intensity = std::fmin(intensity, 1.0); // Clamp to [0, 1]
                ns[i][j] = Color(intensity, intensity, intensity);
            }
        }
    }
    //TODO: also stupid
    //Copy new screen to current screen
    for(int i = 0; i < SCREEN_WIDTH; i++) {
        for (int j = 0; j < SCREEN_HEIGHT; j++) {
            s[i][j] = ns[i][j];
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
            glVertex2d(mapToScreen(i, SCREEN_WIDTH),mapToScreen(j, SCREEN_HEIGHT));
        }
    }
    glEnd();
}

double mapToScreen(int j, const int height) {
return ((2.0*j)/height) - 1.0;
}
