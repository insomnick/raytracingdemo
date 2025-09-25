#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <valarray>
#include <array>
#include <cmath>
#include "color.hpp"
#include "camera.hpp"
#include "sphere.hpp"
#include "ray.hpp"

static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

//Building Screen with Color
//An Array with Dimensions of Screen and color
Color s[SCREEN_WIDTH][SCREEN_HEIGHT];
const int obj_size = 2;
Sphere objects [obj_size*obj_size*obj_size]; //512 objects in the scene

Camera camera{};


void drawScreen();
double mapToScreen(int j, const int height);

void calculateScreen();

int main(void)
{
    //Init
    GLFWwindow* window;
    /* Initialize the library */
    if (!glfwInit())
        return -1;
    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "raytracing demo", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }
    /* Make the window's context current */
    glfwMakeContextCurrent(window);
    if(!gladLoadGL())
        return -1;

    //make key callback
    glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        if (key == GLFW_KEY_KP_2 && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateHorizontal(0.1);
            printf("2: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_KP_8 && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateHorizontal(-0.1);
            printf("8: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_KP_4 && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateVertical(-0.1);
            printf("4: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
        if (key == GLFW_KEY_KP_6 && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
            camera.rotateVertical(0.1);
            printf("6: Camera Direction: %f, %f, %f\n", camera.getDirection().getX(), camera.getDirection().getY(), camera.getDirection().getZ());
        }
    });


    //Demo Screen with Gradient Color
    /*
    for( int i = 0; i < SCREEN_WIDTH; i++ ) {
        for( int j = 0; j < SCREEN_HEIGHT; j++ ) {
            double r = (double) i / SCREEN_WIDTH;
            double g = (double) j / SCREEN_HEIGHT;
            double b = (r + g) /2.0;
            s[i][j] = Color(r, g, b);
        }
    }
    */


    //Load objects
    for (int i = 0; i < obj_size; ++i) {
        for (int j = 0; j < obj_size; ++j) {
            for (int k = 0; k < obj_size; ++k) {
                const auto x = i * 2.0;
                const auto y = j * 2.0 - obj_size - 2.0;
                const auto z = k * 2.0 - obj_size - 2.0;
                objects[i * obj_size * obj_size + j * obj_size + k] = Sphere(x, y, z, 0.5);
            }
        }
    }

    for (auto & object : objects) {
        printf("Object Center: %f, %f, %f\n", object.getCenter().getX(), object.getCenter().getY(), object.getCenter().getZ());
    }

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        //Time every frame
        static double previous_seconds = glfwGetTime();
        double current_seconds = glfwGetTime();
        double elapsed_seconds = current_seconds - previous_seconds;
        previous_seconds = current_seconds;
        //printf("Time calculate Screen: %f \n", elapsed_seconds);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the framebuffer
        calculateScreen();
        drawScreen();
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        /* Poll for and process events */
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

void calculateScreen() {

    Color ns[SCREEN_WIDTH][SCREEN_HEIGHT];

    // Pre-calculate constants outside loops
    const auto camera_pos = camera.getPosition();
    const auto camera_dir = camera.getDirection();
    const auto fov_half_tan = std::tan(camera.getFov() * 0.5);
    const auto aspect_ratio = static_cast<double>(SCREEN_WIDTH) / SCREEN_HEIGHT;
    const auto inv_width = 1.0 / SCREEN_WIDTH;
    const auto inv_height = 1.0 / SCREEN_HEIGHT;

    // Pre-calculate camera basis vectors for proper orientation
    const auto camera_right = Vector3{0.0, 0.0, 1.0}; // Assuming Z is right
    const auto camera_up = Vector3{0.0, 1.0, 0.0};    // Assuming Y is up

    for (int i = 0; i < SCREEN_WIDTH; ++i) {
        const auto px_base = (2.0 * (i + 0.5) * inv_width - 1.0) * fov_half_tan * aspect_ratio;

        for (int j = 0; j < SCREEN_HEIGHT; ++j) {
            const auto py = (1.0 - 2.0 * (j + 0.5) * inv_height) * fov_half_tan;
            auto ray_direction = camera_dir + (camera_up * py) + (camera_right * px_base);

            // Fast normalize using reciprocal, should already be normalized
            const auto inv_length = 1.0 / ray_direction.length();
            ray_direction = ray_direction * inv_length;

            const Ray ray{camera_pos, ray_direction};

            auto closest_distance = std::numeric_limits<double>::max();

            // Find closest intersection only
            for (const auto& object : objects) {
                const auto intersection = object.intersect(ray);
                if (intersection != nullptr) {
                    const auto distance = (*intersection - camera_pos).length();
                    if (distance < closest_distance) {
                        closest_distance = distance;
                        auto c = 1.0 - (distance * 0.1);
                        c = std::clamp(c, 0.0, 1.0);
                        ns[i][j] = Color(c, c, c);
                    }
                }
            }
        }
    }
    // Copy new screen to current screen
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
