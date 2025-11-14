#ifndef RAYTRACINGDEMO_BENCHMARK_HPP
#define RAYTRACINGDEMO_BENCHMARK_HPP

#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include "../camera.hpp"

class Benchmark {
private:
    inline static int testrun_counter = 0;
    std::filesystem::path run_directory;
public:
    Benchmark() {
        // Find the first non-existing testrun_n directory and create it so no data is overwritten
        while(true)  {
            std::error_code ec;
            std::filesystem::path candidate{"testruns/testrun_" + std::to_string(testrun_counter)};
            bool exists = std::filesystem::exists(candidate, ec);
            if (ec) {
                std::cerr << "Error checking directory '" << candidate << "': " << ec.message() << "\n";
                testrun_counter++;
                continue;
            }
            if (!exists) {
                std::filesystem::create_directories(candidate, ec);
                if (ec) {
                    std::cerr << "Failed to create directory '" << candidate << "': " << ec.message() << "\n";
                    // Try next number; maybe a race or permission issue on this one
                    testrun_counter++;
                }
                run_directory = candidate;
                testrun_counter++;
                break;
            }
            // Already exists: try next number
            testrun_counter++;
        };
    }

    const std::filesystem::path& directory() const { return run_directory; }

    void saveDataFrame(std::string file_name, std::string model_name, double model_scale, std::string alorithm_name, Camera camera_params, double render_calculation_seconds) {
        std::error_code ec;
        if (!std::filesystem::exists(run_directory, ec)) {
            // Attempt recreate if removed after construction
            std::filesystem::create_directories(run_directory, ec);
            if (ec) {
                std::cerr << "Cannot (re)create benchmark directory '" << run_directory << "': " << ec.message() << "\n";
                return;
            }
        }
        const auto filePath = run_directory / file_name;

        // Check if file exists to determine if we need to write header
        bool file_exists = std::filesystem::exists(filePath, ec);
        if (ec) {
            std::cerr << "Error checking file existence '" << filePath << "': " << ec.message() << "\n";
            file_exists = false; // Assume it doesn't exist if we can't check
        }

        std::ofstream myfile(filePath, std::ios::app); // append for data series
        if (!myfile.is_open()) {
            std::cerr << "Could not open file for writing: " << filePath << "\n";
            return;
        }

        // Write header if file is new
        if (!file_exists) {
            myfile << "file_name,model_name,model_scale,algorithm_name,cam_pos_x,cam_pos_y,cam_pos_z,cam_dir_x,cam_dir_y,cam_dir_z,time_seconds\n";
        }

        // Write data row
        myfile << file_name <<",";
        myfile << model_name <<",";
        myfile << model_scale<<",";
        myfile << alorithm_name <<",";
        myfile << camera_params.getPosition().getX() <<","<< camera_params.getPosition().getY() <<","<< camera_params.getPosition().getZ() <<",";
        myfile << camera_params.getDirection().getX() <<","<< camera_params.getDirection().getY() <<","<< camera_params.getDirection().getZ() <<",";
        myfile << render_calculation_seconds << "\n";
        myfile.close();
    }
};
#endif //RAYTRACINGDEMO_BENCHMARK_HPP
