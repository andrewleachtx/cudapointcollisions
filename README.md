# CUDA Point-Plane Collisions
The goal of this is to maximize the number of particles possible in a physics simulation with 6 plane checks.

# Linux
## Dependencies
For code that is run on the Linux machine for access to a 4090, this is what was necessary:

1. To get the VM CUDA to run, append
```sh
export PATH=/usr/local/cuda-11.7/bin$PATH
```
to `~/.profile` and `source ~/.bashrc`. To know it works, run `nvcc --version`.

2. Install GLM somewhere - I added an environment variable `export GLM_INCLUDE_DIR=~/packages/glm` to my `.profile` and got the code from [here](https://github.com/g-truc/glm/tree/master). Feel free to just modify `CMakeLists.txt` to point to your glm installation if its local to the project, or elsewhere.
3. `CMakeLists.txt` needed to be modified to find the header files in `/usr/` using [this](https://stackoverflow.com/questions/13167598/error-cuda-runtime-h-no-such-file-or-directory/75559127#75559127).
4. To run `ncu` meaningfully, you need root access. https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters

## Helpful Commands
- `cat /etc/os-release` shows the distro and device architecture.
- `nvidia-smi` provides GPU information (assuming it is a NVIDIA gpu).
- `ncu --target-processes all -o <report name> <executable> [args]` will generate a compute report over all kernels given some executable and its args. 

## Notes
1. The program evaluates for 15 seconds as an arbitrary constant time across all benchmarks. Arguably, the program will converge at much faster rates, and this would depend on the last active particle.
   1. `atomicSub` with a global counter (i.e. `d_activeParticles`) is an option to determine true convergence time, but these operations are limited to `unsigned int` which is smaller than `size_t`, and I didn't want to differ the simulation code from the initial rendered portion which proved its functionality.
   2. It is probably possible to create an atomicSubSize_t, as I have seen user-made "`atomicSubFloat`" functions online.
   3. This **15 second period** means "total kernel time" is really out of 15 seconds, or 15000 ms. In that sense, the `kernel time / 15000` is a better predictor of kernel usage when comparing relatively.
2. 