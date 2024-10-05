#include "include.h"

#include "constants.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>

using std::cout, std::cerr, std::endl;
using std::vector, std::string, std::make_shared, std::shared_ptr;
using std::stoi, std::stoul, std::min, std::max, std::numeric_limits, std::abs;

static size_t g_maxParticles;
float g_curTime(0.0f);
long long g_curStep(0);

// Device Hyperparameters - Constant Space //
__device__ size_t d_activeParticles;
__constant__ size_t d_maxParticles;
__constant__ size_t d_numPlanes;
__constant__ glm::vec3 d_planeP[6];
__constant__ glm::vec3 d_planeN[6];

bool g_is_simFrozen(false);
cudaEvent_t interKernelStart, interKernelStop, intraKernelStart, intraKernelStop;

ParticleData g_particles;
PlaneData g_planes;

static void init() {
    srand(0);

    // CUDA //
        gpuErrchk(cudaSetDevice(0));
        cudaEventCreate(&intraKernelStart);
        cudaEventCreate(&intraKernelStop);
        cudaEventCreate(&interKernelStart);
        cudaEventCreate(&interKernelStop);

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
__device__ glm::vec3 getAcceleration(int idx, glm::vec3* v) {
    float mass = 1.0f;

    // Simple force composed of gravity and air resistance
    glm::vec3 F_total = glm::vec3(0.0f, GRAVITY, 0.0f) - ((AIR_FRICTION / mass) * v[idx]);

    return F_total;
}

__device__ void solveConstraints(int idx, const glm::vec3* pos, const glm::vec3* vel, const float* radii, 
                                 glm::vec3& x_new, glm::vec3& v_new, float& dt, const glm::vec3& a) {
    // Avoid at rest particles
    // if (glm::dot(v_new, v_new) < 0.25f * STOP_VELOCITY) {
    //     return;
    // }

    // Grab position and velocity
    const glm::vec3& x(pos[idx]), v(vel[idx]);
    
    // Plane Collisions //
    for (int i = 0; i < d_numPlanes; i++) {
        // TODO: Move plane_p and plane_n to constant space
        const glm::vec3& p(d_planeP[i]), n(d_planeN[i]);

        glm::vec3 new_p = p + (radii[idx] * n);

        float d_0 = glm::dot((x - new_p), n);
        float d_n = glm::dot(glm::vec3(x_new - new_p), n);

        glm::vec3 v_tan = v - (glm::dot(v, n) * n);
        v_tan = (1 - FRICTION) * v_tan;

        if (d_n < FLOAT_EPS) {
            float f = d_0 / (d_0 - d_n);
            dt = f * dt;

            glm::vec3 v_collision = (v + (dt * a)) * RESTITUTION;    
            glm::vec3 x_collision = x;

            x_new = x_collision;
            v_new = (abs(glm::dot(v_collision, n)) * n) + (v_tan);

            // Naive jitter handling (could also check to make sure acceleration opposite to normal)
            if (abs(glm::dot(v_new, n)) < STOP_VELOCITY) {
                v_new = v_tan;
            }
        }
    }

    // If |v_idx| < STOP_VELOCITY we can assume a particle has "converged", and we should reduce the counter.
    // this works because each particle has a significant nonzero initial velocity. Also, we can use |v_idx|2 norm
    // if (glm::dot(v_new, v_new) < 0.25f * STOP_VELOCITY) {
    //     atomicSub(&d_activeParticles, 1)
    // }
}

__global__ void simulateKernel(glm::vec3* positions, glm::vec3* velocities, float* radii) {
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
        const glm::vec3& x_cur(positions[idx]), v_cur(velocities[idx]);
        glm::vec3 a = getAcceleration(idx, velocities);

        // Integrate over timestep to update
        glm::vec3 x_new = x_cur + (v_cur * dt);
        glm::vec3 v_new = v_cur + (a * dt);

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

// Start with arbitrary sizes, optimize later
dim3 threadsPerBlock(NUM_THREADS);
size_t problem_sz = g_particles.h_maxParticles;
dim3 blocksPerGrid((problem_sz + threadsPerBlock.x - 1) / threadsPerBlock.x);

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

    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#elapsed-time
    gpuErrchk(cudaEventRecord(intraKernelStart, 0));
    simulateKernel<<<blocksPerGrid, threadsPerBlock>>>(g_particles.d_position, g_particles.d_velocity, g_particles.d_radii);
    gpuErrchk(cudaEventRecord(intraKernelStop, 0));
    gpuErrchk(cudaEventSynchronize(intraKernelStop));

    float elapsed;
    gpuErrchk(cudaEventElapsedTime(&elapsed, intraKernelStart, intraKernelStop));

    if (g_timeSampleCt < g_timeSampleSz) {
        g_totalKernelTimes += elapsed;
        g_timeSampleCt++;
    }

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

    // Program "converges" at 15 seconds.
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::seconds(15);

    while (std::chrono::high_resolution_clock::now() < end) {
        if (!g_is_simFrozen) {
            launchSimulateKernel();
        }
    }

    // Print Timings //
    if (BENCHMARK) {
        if (g_timeSampleCt >= g_timeSampleSz) {
            printf("Overall Kernel Time Out of 15 second \"convergence\" period: %f ms\n", g_totalKernelTimes);
            printf("Average Kernel Time over %d samples: %f ms\n", g_timeSampleSz, (g_totalKernelTimes / g_timeSampleSz));
        }
        else {
            printf("Not enough time to reach %d samples, perhaps run longer for \"convergence\"\n?", g_timeSampleSz);
        }
    }

    // CUDA Cleanup //
    cudaEventDestroy(interKernelStart);
    cudaEventDestroy(interKernelStop);
    cudaEventDestroy(intraKernelStart);
    cudaEventDestroy(intraKernelStop);

    return 0;
}
