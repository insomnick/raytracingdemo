#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <valarray>
#include "color.h"
#include "camera.h"
#include "sphere.h"
#include "ray.h"

static const int SCREEN_WIDTH = 720;
static const int SCREEN_HEIGHT = 720;

//Building Screen with Color
//An Array with Dimensions of Screen and color
Color s[SCREEN_WIDTH][SCREEN_HEIGHT];

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
    for( int i = 0; i < SCREEN_WIDTH; i++ ) {
        for( int j = 0; j < SCREEN_HEIGHT; j++ ) {
            double r = (double) i / SCREEN_WIDTH;
            double g = (double) j / SCREEN_HEIGHT;
            double b = (r + g) /2.0;
            s[i][j] = Color(1, 1, 1);
        }
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

    Sphere objects [3] = {
            Sphere(0.0, 0.0, 0.0, 0.25),
            Sphere(5.0, 1.0, 0.0, 2.0),
            Sphere(0.5, 0.5, 0.0, 0.1)
    };
    Camera camera;

    //1. Loop over every pixel of the screen with corresponding Ray
    //2. Loop over every object in the scene for Collision detection
    for(int i = 0; i < SCREEN_WIDTH; i++) {
        for (int j = 0; j < SCREEN_HEIGHT; j++) {
            //1. Create Ray for every Pixel

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
