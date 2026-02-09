# raytracingdemo

This is a simple ray tracing demo project written in C++ using OpenGL for rendering. 
It compares the performance of different BVH (Bounding Volume Hierarchy) construction methods for accelerating ray tracing in static scenes.
The focus relies on the performance differences between different Wide BVH construction algorithms and degrees.
A script is provided to benchmark the different methods and visualize the results.

To run the demo and benchmarks, you need the following dependencies:

Ubuntu / debian:
```bash
# System packages
sudo apt update
sudo apt install cmake make g++ libgl1-mesa-dev libglfw3-dev libglew-dev python3 python3-pip

# Python packages
pip3 install pandas matplotlib numpy scipy seaborn
```
To build and run the demo:

```bash
./run_analysis.sh
```

The results of the benchmarks are saved in the `testruns/results` folder using the provided Python script `bvh_analysis.py`.

For more detailed analysis and visualization of the results, you can check out the other python scrips in the `scripts` folder.
