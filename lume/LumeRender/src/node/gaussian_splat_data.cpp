/*
 * gaussian_splat_data.cpp — Storage for shared gaussian splatting data
 */
#include "gaussian_splat_data.h"

__attribute__((visibility("default"))) const GpuGaussianData* g_gaussianSplatData = nullptr;
__attribute__((visibility("default"))) uint32_t g_gaussianSplatCount = 0;
__attribute__((visibility("default"))) GaussianCameraData g_gaussianCameraData = {};
__attribute__((visibility("default"))) bool g_gaussianDataReady = false;
__attribute__((visibility("default"))) bool g_gaussianCameraReady = false;
