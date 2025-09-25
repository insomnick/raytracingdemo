#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <valarray>
#include "color.hpp"
#include "camera.hpp"
#include "sphere.hpp"
#include "ray.hpp"

static const int SCREEN_WIDTH = 720;
static const int SCREEN_HEIGHT = 720;

//Building Screen with Color
//An Array with Dimensions of Screen and color
Color s[SCREEN_WIDTH][SCREEN_HEIGHT];
const int obj_size = 6;
Sphere objects [obj_size*obj_size*obj_size]; //512 objects in the scene

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
                //Center coordinates
                double x = i * 2.0 + 0.0;
                double y = j * 2.0 - ((obj_size) / 2.0) - 2.0;
                double z = k * 2.0 - ((obj_size) / 2.0) - 2.0;
                objects[i*obj_size*obj_size + j*obj_size + k] = Sphere(x, y, z, 0.5);
            }
        }
    }

    for (auto & object : objects) {
        printf("Object Center: %f, %f, %f\n", object.getCenter().getX(), object.getCenter().getY(), object.getCenter().getZ());
    }

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
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

    Camera camera{};

    //1. Loop over every pixel of the screen with corresponding Ray
    //2. Loop over every object in the scene for Collision detection
    for(int i = 0; i < SCREEN_WIDTH; i++) {
        for (int j = 0; j < SCREEN_HEIGHT; j++) {
            //1. Create Ray for every Pixel
            //Calculate coordinate of plane
            double x = camera.getPlane().getX();
            double y = mapToScreen(i, SCREEN_WIDTH) * camera.getPlane().getY();
            double z = mapToScreen(j, SCREEN_HEIGHT) * camera.getPlane().getZ();

            auto ray_direction = camera.getDirection() + Vector3 {x, y, z}; //offset

            //Normalize ray direction
            auto length = ray_direction.length();
            ray_direction = Vector3::multiply(ray_direction, 1.0/length);

            Ray ray {camera.getPosition() ,ray_direction};
            //printf("Pixel: %d, %d ", i, j);
            //printf("Ray Direction: %f, %f, %f\n", ray.direction.getX(), ray.direction.getY(), ray.direction.getZ());

            //2. Loop over every object in the scene for Collision detection (dumb and slow way)
            // If Collision detected color the pixel with the color of the object (in this moment black) else white
            //s[i][j] = Color(ray_direction.getX()*ray_direction.getX(), ray_direction.getY()*ray_direction.getY(), ray_direction.getZ()*ray_direction.getZ());

            for (const auto & object : objects) {
                const auto intersection = object.intersect(ray.origin, ray.direction);
                if(!(intersection == nullptr)){
                    double c = Vector3::subtract(*intersection, camera.getPosition()).length();
                    c = 1.0 - (c / 10.0);
                    if(c < 0.0) c = 0.0;
                    if(c > 1.0) c = 1.0;
                    if(s[i][j].r() < c) //If multiple objects are in the line of sight only take the closest one
                    s[i][j] = Color(c,c,c);
                }
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
            glVertex2d(mapToScreen(i, SCREEN_WIDTH),mapToScreen(j, SCREEN_HEIGHT));
        }
    }
    glEnd();
}

double mapToScreen(int j, const int height) {
return ((2.0*j)/height) - 1.0;
}
