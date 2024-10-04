#include "include.h"

#include "constants.h"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

#include <chrono>

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;
using glm::vec3, glm::vec4, glm::mat4;

static size_t g_maxParticles;
float g_curTime(0.0f), g_nextDisplayTime(1.0f);
long long g_curStep(0);

// Device Hyperparameters - Constant Space //
__constant__ size_t d_maxParticles;
__constant__ size_t d_numPlanes;
__constant__ vec3 d_planeP[6];
__constant__ vec3 d_planeN[6];

bool g_is_simFrozen(true);

ParticleData g_particles;
PlaneData g_planes;

static void init() {
    // srand(static_cast<unsigned int>(time(nullptr)));
    srand(0);

    // CUDA //
        gpuErrchk(cudaSetDevice(0));

    // Planes //
        const float plane_width = 540.0f;
        g_planes = PlaneData(6, plane_width);
        g_planes.initPlanes();
        g_planes.copyToDevice();

    // Particles //
        g_particles = ParticleData(g_maxParticles);
        g_particles.init(0.5f);
        g_particles.copyToDevice();
}

/*
    Instead of iterating over each particle, we will make a kernel that runs for each particle
*/

// Assume mass is 1; F / 1 = A
__device__ vec3 getAcceleration(int idx, vec3* v) {
    float mass = 1.0f;

    // Simple force composed of gravity and air resistance
    vec3 F_total = vec3(0.0f, GRAVITY, 0.0f) - ((AIR_FRICTION / mass) * v[idx]);

    return F_total;
}

__device__ void solveConstraints(int idx, const vec3* x, const vec3* v, const float* radii, 
                                 vec3& x_new, vec3& v_new, float& dt, const vec3& a) {
    // Plane Collisions //
    for (int i = 0; i < d_numPlanes; i++) {
        // TODO: Move plane_p and plane_n to constant space
        const vec3& p(d_planeP[i]), n(d_planeN[i]);
        const vec3& x(x[idx]), v(v[idx]);

        vec3 new_p = p + (radii[idx] * n);
        float d_0 = dot((x - new_p), n);
        float d_n = dot((x_new - new_p), n);
        
        vec3 v_tan = v - (dot(v, n) * n);
        v_tan = (1 - FRICTION) * v_tan;

        if (d_n < FLOAT_EPS) {
            float f = d_0 / (d_0 - d_n);
            dt = f * dt;

            vec3 v_collision = (v + (dt * a)) * RESTITUTION;    
            vec3 x_collision = x;

            x_new = x_collision;
            v_new = (abs(dot(v_collision, n)) * n) + (v_tan);

            // Naive jitter handling (could also check to make sure acceleration opposite to normal)
            if (abs(dot(v_new, n)) < STOP_VELOCITY) {
                v_new = v_tan;
            }
        }
    }
}

__global__ void simulateKernel(vec3* positions, vec3* velocities, float* radii) {
    /* To retrieve the index in this 1D instance, we do this: */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Because we are effectively doing a ceiling function for threads, some amount will not have a particle associated
    if (idx >= d_maxParticles) {
        return;
    }

    // Use of __constant__ space on the kernel https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory
    float dt_remaining = DT_SIMULATION;
    float dt = dt_remaining;

    int max_iter = 10;

    while (max_iter && dt_remaining > 0.0f) {
        const vec3& x_cur(positions[idx]), v_cur(velocities[idx]);
        vec3 a = getAcceleration(idx, velocities);

        // Integrate over timestep to update
        vec3 x_new = x_cur + (v_cur * dt);
        vec3 v_new = v_cur + (a * dt);

        // Solve any constraints imposed by update positions
        solveConstraints(idx, positions, velocities, radii, x_new, v_new, dt, a);

        // Update particle state
        positions[idx] = x_new;
        velocities[idx] = v_new;

        // Update remaining time
        dt_remaining -= dt;
        max_iter--;
    }
}

static const int g_timeSampleSz = 50;
static int g_timeSampleCt = 0;
static float g_totalKernelTimes = 0.0f;

void launchSimulateKernel() {
    /*
        Hierarchically, there is a grid composed of multiple thread blocks
            - Each thread block has multiple threads, or instances of the kernel

        The <<<...>>> notation expresses <<<blocksPerGrid, threadsPerBlock>>>
            - You can use dim3(x), dim3(x, y), or dim3(x, y, z) to describe how many unique
              indices are necessary for your vector, matrix, or volume operation; we want
              to use a dim3(x) in this case, as we have a vector of n particles.
            - blocksPerGrid is calculated as (problem_sz + block_sz - 1) / block_sz, which you
              would calculate for each of x, y, z in the bpg dim3 constructor.

        https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#occupancy-calculator
    */

    // Start with arbitrary sizes, optimize later
    dim3 threadsPerBlock(NUM_THREADS);
    size_t problem_sz = g_particles.h_maxParticles;
    dim3 blocksPerGrid((problem_sz + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    simulateKernel<<<blocksPerGrid, threadsPerBlock>>>(g_particles.d_position, g_particles.d_velocity, g_particles.d_radii);

    gpuErrchk(cudaGetLastError());
}

int main(int argc, char**argv) {
    if (argc < 2) {
        cout << "Usage: ./executable <number of particles>" << endl;
        return 0;
    }

    g_maxParticles = stoi(argv[1]);

    // Initialize planes, particles, cuda buffers
    init();

    // The renderFrozen variable is important, as it is flipped on window resize
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::seconds(180);

    while (std::chrono::high_resolution_clock::now() < end) {
        if (!g_is_simFrozen) {
            launchSimulateKernel();
        }
    }

    // Print Timings //
        if (BENCHMARK) {
            if (g_timeSampleCt >= g_timeSampleSz) {
                cout << "Average Kernel Time: " << g_totalKernelTimes / g_timeSampleSz << "ms" << endl;
                cout << "g_TimeSampleCt = " << g_timeSampleCt << endl;
            }

        }

    // CUDA Cleanup //

    return 0;
}