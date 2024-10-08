#include "ParticleData.h"
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <random>
#include "../../include.h"
using std::vector, std::cerr, std::cout, std::endl;
using glm::vec3, glm::vec4;

// Grabs from main.cu
extern __constant__ size_t d_maxParticles;

__host__ __device__ ParticleData::ParticleData() : h_maxParticles(0) {
    d_position = nullptr;
    d_velocity = nullptr;
    d_radii = nullptr;
}

__host__ __device__ ParticleData::ParticleData(size_t max_particles) : h_maxParticles(max_particles) {
    size_t mem_required = max_particles * (2 * sizeof(vec4) + sizeof(float));
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    if (mem_required > free) {
        // std::cerr << __LINE__ <<
        cerr << "FATAL: " << __FILE__ << ": " << __LINE__ << endl;
        cerr << "FATAL: Insufficient memory to cudaMalloc " << mem_required << " bytes on device for particles " << endl;
        exit(1);
    }
    else {
        cout << "Allocating " << mem_required << " of " << total << " bytes on device" << endl;
    }

    gpuErrchk(cudaMalloc(&d_position, max_particles * sizeof(vec4)));
    gpuErrchk(cudaMalloc(&d_velocity, max_particles * sizeof(vec4)));
    gpuErrchk(cudaMalloc(&d_radii, max_particles * sizeof(float)));
}

__host__ __device__ ParticleData::~ParticleData() {
    // if (d_position) {
    //     cout << "cudaFree'ing d_position* = " << d_position << endl; 
    //     gpuErrchk(cudaFree(&d_position));
    // }
    // if (d_velocity) {
    //     gpuErrchk(cudaFree(&d_velocity));
    // }
    // if (d_radii) {
    //     gpuErrchk(cudaFree(&d_radii));
    // }

    // printf("Destructor called %d time(s) for this=%p\n", ++destructorCt, this);
}

// Copies physics data from host to device (this should happen only once!)
__host__ __device__ void ParticleData::copyToDevice() {
    gpuErrchk(cudaMemcpyToSymbol(&d_maxParticles, &h_maxParticles, sizeof(size_t)));

    gpuErrchk(cudaMemcpy(d_position, h_position.data(), h_maxParticles * sizeof(vec4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_velocity, h_velocity.data(), h_maxParticles * sizeof(vec4), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radii, h_radii.data(), h_maxParticles * sizeof(float), cudaMemcpyHostToDevice));
}

// Arbitrary population of on-host memory for rendering with vsh & fsh
void ParticleData::init(const float& radius) {
    assert(h_maxParticles > 0);

    h_position.resize(h_maxParticles);
    std::random_device rd;
    // std::mt19937 gen(rd());
    std::mt19937 gen(0);
    float minF(180.0f), maxF(240.0f);
    std::uniform_real_distribution<float> dist_pos(minF, maxF);
    std::uniform_real_distribution<float> dist_vel(-1.0f, 1.0f);

    for (vec4& x : h_position) {
        float rand_x(dist_pos(gen));
        float rand_y(dist_pos(gen));
        float rand_z(dist_pos(gen));

        x = vec4(rand_x - maxF, rand_y + minF, rand_z - maxF, 0.0f);
        // cout << x.x << " " << x.y << " " << x.z << endl;
    }

    h_velocity.resize(h_maxParticles);
    float push = 90.0f;
    for (vec4& v : h_velocity) {
        float rand_x = dist_vel(gen);
        float rand_y = dist_vel(gen);
        float rand_z = dist_vel(gen);

        v = vec4(push * rand_x, push * rand_y, push * rand_z, 0.0f);
    }

    h_radii.resize(h_maxParticles);
    for (float& r : h_radii) {
        r = radius;
    }

    assert(h_position.size() == h_maxParticles);
    assert(h_velocity.size() == h_maxParticles);
    assert(h_radii.size() == h_maxParticles);
}