#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <array>
#include <memory>
#include <algorithm>
#include <vector>
#include <map>

#include "color.hpp"
#include "camera.hpp"
#include "primitives/sphere.hpp"
#include "primitives/ray.hpp"
#include "bvh.hpp"
#include "utils/object_loader.hpp"
#include "utils/timer.hpp"
#include "utils/benchmark.hpp"
#include "camera_path.hpp"

struct Hit {
    bool hit = false;
    Vector3 position{};
    Vector3 normal{};
};

struct TestrunConfiguration {
    std::string object_file;
    double object_scale;
    std::string bvh_algorithm;
    int bvh_degree;
    int camera_path_resolution;
    bool no_window;
};


static const int SCREEN_WIDTH = 500;
static const int SCREEN_HEIGHT = 500;

static std::vector<Hit> ray_hits(SCREEN_WIDTH * SCREEN_HEIGHT); //positions and normals of hits
static std::array<std::array<Color, SCREEN_HEIGHT>, SCREEN_WIDTH> screen; // pixel colors

GLFWwindow* window;

void drawScreen();
double mapToScreen(int j, const int height);
void calculateScreen(BVH& bvh, Camera& camera);
static void shadeScreen(const Vector3& camera_pos);
std::vector<Primitive*> setupScene(const std::string& obj_filename, double obj_scale=1.0);
int setupOpenGL();
void runTest(const TestrunConfiguration& config);


int main(void) {

    std::multimap<std::string, int> bvh_algorithms = {
            { "sah",    2 },
            { "sah",    4 },
            { "median", 2 },
            { "median", 4 },
            { "median", 8 },
    };
    std::map<std::string, double> object_files= {
              { "stanford-bunny.obj", 30.0}
            , { "teapot.obj",        1.0 }
            , { "suzanne.obj",       3.0 }
            , { "sponza.obj",         1.0 }
    };
    const int camera_path_resolution = 36;
    bool no_window = true;

    for (const auto& [bvh_algorithm, bvh_degree] : bvh_algorithms) {
        for (const auto &[object_file, object_scale]: object_files) {
            TestrunConfiguration config;
            config = TestrunConfiguration{
                    .object_file = object_file,
                    .object_scale = object_scale,
                    .bvh_algorithm = bvh_algorithm,
                    .bvh_degree = bvh_degree,
                    .camera_path_resolution = camera_path_resolution,
                    .no_window = no_window
            };
            runTest(config);
        };
    }
    return 0;
}

void runTest(const TestrunConfiguration& config) {

    Camera camera{SCREEN_WIDTH, SCREEN_HEIGHT};

    Timer timer;
    Benchmark bm;
    std::string object_file = config.object_file;
    int bvh_degree = config.bvh_degree;
    std::string algorithm_name = config.bvh_algorithm + "-" + std::to_string(bvh_degree); // "sah", "median", "stupid"
    std::vector<Primitive*> objects = setupScene(object_file, config.object_scale);

    //calculate object center ( center of all centers )
    Vector3 object_center{0.0, 0.0, 0.0};
    for (auto & obj : objects) {
        object_center = object_center + obj->getCenter();
    }
    object_center = object_center * (1.0 / objects.size());
    camera.setPosition(object_center + Vector3{0.0, 0.0, 5.0});

    //Build BVH time calculation
    BVH bvh = BVH::medianSplitConstruction(objects, 4); //TODO No stupid dummy
    double elapsed = 0.0;
    for (int i = 0; i < 1; i++) {
        timer.reset();
        if (algorithm_name.starts_with("sah")) {
            printf("Building BVH using Surface Area Heuristic Construction...\n");
            bvh = BVH::surfaceAreaHeuristicConstruction(objects, bvh_degree);
        } else if (algorithm_name.starts_with("median")) {
            printf("Building BVH using Median Split Construction...\n");
            bvh = BVH::medianSplitConstruction(objects, bvh_degree);
        } else {
            printf("Building BVH using Stupid Construction...\n");
            bvh = BVH::stupidConstruct(objects);
        }
        elapsed = timer.elapsed();
        printf("Time build BVH using %s Split: %f \n", algorithm_name.c_str(), elapsed);
        bm.saveDataFrame("bvh_build_times.csv", object_file, config.object_scale, algorithm_name, camera, elapsed);
    }

    if(!config.no_window) {
        if (setupOpenGL() != 0) {
            std::cerr << "Failed to initialize OpenGL context.\n";
            return;
        }
    }
    //App loop
    int resolution = config.camera_path_resolution;
    CameraPath camera_path(camera.getPosition() - Vector3{0.0, 0.0, 5.0}, resolution);   //TODO: hacky position change later
    int path_step = 0;
    while (true) {
        //Camera update for path
        if(path_step >= resolution) {
            break;  //Testrun end
        }

        Ray cam_ray = camera_path.circularPath(path_step);
        camera.setPosition(cam_ray.getOrigin());
        camera.setDirection(cam_ray.getDirection());
        path_step++;

        if(!config.no_window) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        timer.reset();
        calculateScreen(bvh, camera);   //MAGIC: fills ray_hits
        elapsed = timer.elapsed();
        //printf("Time calculate Screen: %f \n", elapsed);
        bm.saveDataFrame("render_times.csv", object_file, config.object_scale, algorithm_name, camera, elapsed);

        timer.reset();
        shadeScreen(camera.getPosition());
        elapsed = timer.elapsed();
        //printf("Time shade Screen: %f \n", elapsed);
        bm.saveDataFrame("shading_times.csv", object_file, config.object_scale,algorithm_name, camera, elapsed);
        //TODO: save image to file here for each step?


        if(!config.no_window) {
            timer.reset();
            drawScreen();   //just screen drawing (out of scope for now)
            elapsed = timer.elapsed();
            //printf("Time draw Screen: %f \n", elapsed);
            bm.saveDataFrame("drawing_times.csv", object_file, config.object_scale, algorithm_name, camera, elapsed);

            if(glfwWindowShouldClose(window)) {
                break;
            }
            //GLFW magic for screen drawing and events
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }
    if(!config.no_window) {
        glfwTerminate();
    }

    //delete objects TODO: do this smarter later
    for (auto & obj : objects) {
        delete obj;
    }
}

int setupOpenGL() {

    //Init GLFW
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

    return 0;
}

std::vector<Primitive*> setupScene(const std::string& obj_filename, double obj_scale) {

    std::vector<Triangle> loaded_object;
    auto file = ObjectLoader::loadFromFile("example/" + obj_filename, obj_scale);
    loaded_object.insert(loaded_object.end(), file.begin(), file.end());
    printf("Loaded %zu triangles from OBJ file \'%s\'\n", loaded_object.size(), obj_filename.c_str());
    //convert to ptrs
    std::vector<Primitive*> loaded_object_ptrs;
    loaded_object_ptrs.reserve(loaded_object.size());
    for (auto & obj : loaded_object) {
        loaded_object_ptrs.push_back(new Triangle(obj));
    }
    return loaded_object_ptrs;
}

void calculateScreen(BVH& bvh, Camera& camera) {
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
            const double py = camera.getPixelY(j);
            Vector3 dir = camera_dir + up * py + right * px;
            dir = dir * (1.0 / dir.length());
            Ray ray{camera_pos, dir};

            const int idx = j + i * SCREEN_HEIGHT;
            if (const auto hitRay = BVH::traverse(bvh, ray)) {
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
