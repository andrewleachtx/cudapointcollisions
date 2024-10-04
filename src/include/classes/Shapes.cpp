#include "Shapes.h"
#include <cuda_runtime.h>

#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>

using glm::vec3, glm::vec4;
using std::cerr, std::cout, std::endl;

#include "../../include.h"

extern __constant__ size_t d_numPlanes;
extern __constant__ vec3 d_planeP[6];
extern __constant__ vec3 d_planeN[6];

PlaneData::PlaneData(size_t num_planes, float plane_width) : num_planes(num_planes), plane_width(plane_width) {}

PlaneData::~PlaneData() {}

void PlaneData::copyToDevice() {
    gpuErrchk(cudaMemcpyToSymbol(&d_numPlanes, &num_planes, sizeof(size_t)));
    gpuErrchk(cudaMemcpyToSymbol(d_planeP, h_points.data(), num_planes * sizeof(vec3)));
    gpuErrchk(cudaMemcpyToSymbol(d_planeN, h_normals.data(), num_planes * sizeof(vec3)));
}

void PlaneData::initPlanes() {
    // 6 planes form a box
    assert(num_planes == 6);

    h_points.resize(num_planes);
    h_normals.resize(num_planes);
    h_rotations.resize(num_planes);

    h_points[0] = vec3(0.0f, 0.0f, 0.0f);
    h_normals[0] = vec3(0.0f, 1.0f, 0.0f);
    h_rotations[0] = vec4(1.0f, 0.0f, 0.0f, (float)(M_PI / 2.0));

    h_points[1] = vec3(0.0f, plane_width * 2.0f, 0.0f);
    h_normals[1] = vec3(0.0f, -1.0f, 0.0f);
    h_rotations[1] = vec4(1.0f, 0.0f, 0.0f, -(float)(M_PI / 2.0));

    h_points[2] = vec3(-plane_width, plane_width, 0.0f);
    h_normals[2] = vec3(1.0f, 0.0f, 0.0f);
    h_rotations[2] = vec4(0.0f, 1.0f, 0.0f, (float)(M_PI / 2.0));

    h_points[3] = vec3(plane_width, plane_width, 0.0f);
    h_normals[3] = vec3(-1.0f, 0.0f, 0.0f);
    h_rotations[3] = vec4(0.0f, 1.0f, 0.0f, (float)(M_PI / 2.0));

    h_points[4] = vec3(0.0f, plane_width, -plane_width);
    h_normals[4] = vec3(0.0f, 0.0f, 1.0f);
    h_rotations[4] = vec4(1.0f, 0.0f, 0.0f, 0.0f);

    h_points[5] = vec3(0.0f, plane_width, plane_width);
    h_normals[5] = vec3(0.0f, 0.0f, -1.0f);
    h_rotations[5] = vec4(1.0f, 0.0f, 0.0f, 0.0f);
}