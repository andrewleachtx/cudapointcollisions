#ifndef SHAPES_H
#define SHAPES_H

#include <glm/glm.hpp>
#include <vector>

using glm::vec3, glm::vec4;
using std::vector;

/*
    For our plane, all we really need is a point on the plane, and the normal. As far as CUDA goes,
    we don't care about the angle or color of this plane, just the physics.
*/
class PlaneData {
    public:
        vector<vec3> h_points;
        vector<vec3> h_normals;
        vector<vec4> h_rotations;

        size_t num_planes;
        float plane_width;

        PlaneData() {}
        PlaneData(size_t num_planes, float plane_width);
        ~PlaneData();

        void copyToDevice();
        void initPlanes();
};


#endif // SHAPES_H
