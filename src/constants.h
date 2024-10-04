#ifndef CONSTANTS_H
#define CONSTANTS_H

// Hyperparameters //
#define DT_SIMULATION (1.0f / 20.0f)
#define DT_RENDER (1.0f / 144.0f)
#define FLOAT_EPS 1e-8f
#define GRAVITY -9.8f
#define AIR_FRICTION 0.0f
#define FRICTION 0.1f
#define RESTITUTION 0.85f
#define BENCHMARK false
#define STOP_VELOCITY 4.5f

// Threading //
#define NUM_THREADS 256
#define NUM_BLOCKS 256

#endif // CONSTANTS_H