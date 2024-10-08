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

static unsigned int g_deadParticles;

// Device Hyperparameters - Constant Space //
__device__ unsigned int d_deadParticles;
__constant__ size_t d_maxParticles;
__constant__ size_t d_numPlanes;
__constant__ glm::vec3 d_planeP[6];
__constant__ glm::vec3 d_planeN[6];

bool g_is_simFrozen(false);
cudaEvent_t kernel_simStart, kernel_simStop;

// Start with arbitrary sizes, optimize later
dim3 g_threadsPerBlock;
dim3 g_blocksPerGrid;

// static const int g_timeSampleSz = KERNEL_TIMING_SAMPLESZ;
static size_t g_timeSampleCt = 0;
static float g_totalKernelTimes = 0.0f;

ParticleData g_particles;
PlaneData g_planes;

static void init() {
    srand(0);

    // CUDA //
        gpuErrchk(cudaSetDevice(0));
        cudaEventCreate(&kernel_simStart);
        cudaEventCreate(&kernel_simStop);

    // Planes //
        const float plane_width = 540.0f;
        g_planes = PlaneData(6, plane_width);
        g_planes.initPlanes();
        g_planes.copyToDevice();

    // Particles //
        g_particles = ParticleData(g_maxParticles);
        g_particles.init(0.5f);
        g_particles.copyToDevice();

    size_t problem_sz = g_particles.h_maxParticles;
    g_blocksPerGrid = dim3((problem_sz + g_threadsPerBlock.x - 1) / g_threadsPerBlock.x);
}

/*
    Instead of iterating over each particle, we will make a kernel that runs for each particle
*/

// Assume mass is 1; F / 1 = A
__device__ glm::vec3 getAcceleration(int idx, glm::vec4* v) {
    float mass = 1.0f;

    // Simple force composed of gravity and air resistance
    glm::vec3 F_total = glm::vec3(glm::vec4(0.0f, GRAVITY, 0.0f, 0.0f) - ((AIR_FRICTION / mass) * v[idx]));

    return F_total;
}

__device__ void solveConstraints(int idx, const glm::vec4* pos, const glm::vec4* vel, const float* radii, 
                                 glm::vec3& x_new, glm::vec3& v_new, float& dt, const glm::vec3& a) {
    // Avoid at rest particles otherwise our counter will be inaccurate
    if (glm::length(v_new) < STOP_VELOCITY) {
        return;
    }

    // Truncate the w component
    const glm::vec3& x(pos[idx]), v(vel[idx]);

    // Plane Collisions //
    for (int i = 0; i < d_numPlanes; i++) {
        const glm::vec3& p(d_planeP[i]), n(d_planeN[i]);
        // const glm::vec3& x(pos[idx]), v(vel[idx]); // FIXME: UNCOALESCED GLOBAL ACCESS (UNNECESSARY IN LOOP)

        glm::vec3 new_p = p + (radii[idx] * n);

        float d_0 = glm::dot(x - new_p, n);
        float d_n = glm::dot(x_new - new_p, n);

        glm::vec3 v_tan = v - (glm::dot(v, n) * n);
        v_tan = (1 - FRICTION) * v_tan;

        if (d_n < FLOAT_EPS) {
            float f = d_0 / (d_0 - d_n);
            dt = f * dt;

            glm::vec3 v_collision = (v + (dt * a)) * RESTITUTION;    
            glm::vec3 x_collision = x;

            x_new = x_collision;
            v_new = (abs(glm::dot(v_collision, n)) * n) + (v_tan);

            // Because behavior is pretty standard, naive jitter handling is okay here. Two more checks are possible.
            if (abs(glm::dot(v_new, n)) < STOP_VELOCITY) {
                v_new = v_tan;
            }
        }
    }

    // If |v_idx| < STOP_VELOCITY we can assume a particle has "converged", and we should reduce the counter.
    // this works because each particle has a significant nonzero initial velocity.
    if (glm::length(v_new) < STOP_VELOCITY) {
        atomicAdd(&d_deadParticles, 1);
    }
}

// FIXME: Generally, this program suffers from warps being allowed 32 bytes, but we are passing in vec3, vec3 or 12 + 12 = 24 bytes. Padding to vec4 could help with this.
__global__ void simulateKernel(glm::vec4* positions, glm::vec4* velocities, float* radii) {
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
    const glm::vec3& x_cur(positions[idx]), v_cur(velocities[idx]);

    while (max_iter && dt_remaining > 0.0f) {
        // const glm::vec3& x_cur(positions[idx]), v_cur(velocities[idx]); // FIXME: UNCOALESCED GLOBAL ACCESS
        glm::vec3 a = getAcceleration(idx, velocities);

        // Integrate over timestep to update
        glm::vec3 x_new = x_cur + (v_cur * dt);
        glm::vec3 v_new = v_cur + (a * dt);

        // Solve any constraints imposed by update positions
        solveConstraints(idx, positions, velocities, radii, x_new, v_new, dt, a);

        // Update particle state
        positions[idx] = glm::vec4(x_new, 0.0f); // FIXME: UNCOALESCED GLOBAL ACCESS
        velocities[idx] = glm::vec4(v_new, 0.0f); // FIXME: UNCOALESCED GLOBAL ACCESS

        // Update remaining time
        dt_remaining -= dt;
        max_iter--;
    }
}

long long ctr=10e9;

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
    gpuErrchk(cudaEventRecord(kernel_simStart, 0));
    simulateKernel<<<g_blocksPerGrid, g_threadsPerBlock>>>(g_particles.d_position, g_particles.d_velocity, g_particles.d_radii);
    gpuErrchk(cudaEventRecord(kernel_simStop, 0));
    gpuErrchk(cudaEventSynchronize(kernel_simStop));
    
    // TODO: Potentially add an event to avoid the constant memcpy.
    gpuErrchk(cudaMemcpyFromSymbol(&g_deadParticles, d_deadParticles, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost));

    // glm::vec3 posbuf;
    // gpuErrchk(cudaMemcpy(&posbuf, g_particles.d_position, sizeof(glm::vec3), cudaMemcpyDeviceToHost));
    // if (ctr % 1 == 0) {
        // printvec3(posbuf);
        // cout << g_activeParticles << endl;
    // }
    // ctr--;
 
    float elapsed;
    gpuErrchk(cudaEventElapsedTime(&elapsed, kernel_simStart, kernel_simStop));

    g_totalKernelTimes += elapsed;
    g_timeSampleCt++;

    gpuErrchk(cudaGetLastError());
}

int main(int argc, char**argv) {
    if (argc < 3) {
        cout << "Usage: ./executable <number of particles> <threads per block>" << endl;
        return 0;
    }

    g_maxParticles = stoi(argv[1]);
    int threadsPerBlock = stoi(argv[2]);
    g_threadsPerBlock = dim3(threadsPerBlock);

    g_deadParticles = 0;
    gpuErrchk(cudaMemcpyToSymbol(d_deadParticles, &g_deadParticles, sizeof(unsigned int)));

    // Initialize planes, particles, cuda buffers
    init();

    // Program converges when the last moving particle "stops", or the max time is exceeded.
    auto start = std::chrono::high_resolution_clock::now();
    auto end = start + std::chrono::seconds(MAX_SIMULATE_TIME_SECONDS);
    
    while ((std::chrono::high_resolution_clock::now() < end) && (g_deadParticles < g_maxParticles)) {
        launchSimulateKernel();
    }

    // Convergence time
    auto conv_time = std::chrono::high_resolution_clock::now() - start;
    auto conv_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(conv_time).count();
    printf("Actual program time: %ld ms\n", conv_time_ms);

    // Print Timings //
    if (BENCHMARK) {
        float overall = g_totalKernelTimes;
        float avg = g_totalKernelTimes / g_timeSampleCt;
        float usage = g_totalKernelTimes / (conv_time_ms);

        printf("Number of threads: %d, number of blocks (per grid): %d\n", g_threadsPerBlock.x, g_blocksPerGrid.x);
        printf("Average simulateKernel() execution time over %d samples: %f ms\n", g_timeSampleCt, avg);
        printf("Overall kernel time before convergence: %f ms\n", overall);
        printf("Kernel time / total program time: %f\n", usage);
    }

    // CUDA Cleanup //
    cudaEventDestroy(kernel_simStart);
    cudaEventDestroy(kernel_simStop);

    return 0;
}
