#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "../include.h"

#include <cuda_runtime.h>

#include <glm/glm.hpp>
#include <string>
using std::string;

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api/14038590#14038590
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
__inline __host__ void gpuAssert(cudaError_t code, char *file, int line, 
                                 bool abort=true) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),
          file, line);
      if (abort) exit(code);
   }
}

#define printvec3(var) pv3(#var, var)
__inline __host__ __device__ void pv3(const char* varname, glm::vec3 vec) {
    printf("%s: %f, %f, %f\n", varname, vec.x, vec.y, vec.z);
}

float randFloat();
glm::vec3 randXYZ();

#endif // UTILS_H