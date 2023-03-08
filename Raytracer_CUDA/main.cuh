#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <stdio.h>
#include <iostream>

#include "vec3.cuh"
#include "color.cuh"
#include "ray.cuh"
#include "hittablelist.cuh"
#include "sphere.cuh"
#include "camera.cuh"
#include "material.cuh"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"