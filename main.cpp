#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>

static const int SCREEN_WIDTH = 1280;
static const int SCREEN_HEIGHT = 720;

struct Color {
    double r, g, b;
    Color(): r(0), g(0), b(0) {}
    Color(double r, double g, double b) : r(r), g(g), b(b) {}
};

//Building Screen with Color
//An Array with Dimensions of Screen and color
Color s[SCREEN_WIDTH][SCREEN_HEIGHT];

void drawScreen();
double mapToScreen(int j, const int height);

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
            s[i][j] = Color(r, g, b );
        }
    }


    std::vector<float> vertices = {
        -0.5f, -0.5f, 0.0f,
         0.5f, -0.5f, 0.0f,
         0.0f,  0.5f, 0.0f
    };

    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the framebuffer
        drawScreen();
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        /* Poll for and process events */
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}

void drawScreen() {
    glBegin(GL_POINTS);
    /* Render here */
    for(int i = 0; i < SCREEN_WIDTH; i++)  {
        for(int j = 0; j < SCREEN_HEIGHT; j++) {
            Color c = s[i][j];
            glColor3d(c.r,c.g,c.b);
            glVertex2d(mapToScreen(i, SCREEN_WIDTH),mapToScreen(j, SCREEN_HEIGHT));
        }
    }
    glEnd();
}

double mapToScreen(int j, const int height) {
return ((2.0*j)/height) - 1.0;
}
