#ifndef CONSTANTS_H
#define CONSTANTS_H

// Hyperparameters //
#define DT_SIMULATION (1.0f / 60.0f)
#define FLOAT_EPS 1e-8f
#define GRAVITY -9.8f
#define AIR_FRICTION 0.0f
#define FRICTION 0.1f
#define RESTITUTION 0.85f
#define BENCHMARK true
#define STOP_VELOCITY 4.5f
#define MAX_SIMULATE_TIME_SECONDS 500
// Arbitrary, but usually we ended up with ~2.5mil
#define KERNEL_TIMING_SAMPLESZ 1000000


#endif // CONSTANTS_H