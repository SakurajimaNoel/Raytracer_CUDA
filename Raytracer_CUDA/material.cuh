#pragma once

#include "ray.cuh"
#include "hittable.cuh"
#include <curand_kernel.h>

struct hit_record;

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state),curand_uniform(local_rand_state));

__device__ vec3 random_in_hemisphere(const vec3& normal, curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1.0f, 1.0f, 1.0f);
    } while (p.length_squared() >= 1.0f);
    if (dot(p, normal) > 0.0)
        return p;
    else
        return -p;
}

__device__ vec3 random_in_unit_sphere(curandState* local_rand_state) {
    vec3 p;
    do {
        p = 2.0f * RANDVEC3 - vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n)
{
    return v - 2.0f * dot(v, n) * n;
}



__device__ vec3 refract(const vec3& uv, const vec3& n, float ni_over_nt) {
    float cos_theta = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp = ni_over_nt * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

class material
{
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const = 0;

};

class lambertian : public material 
{
public:
    vec3 albedo;
    __device__ lambertian(const vec3& a) : albedo(a){}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const override
    {
        vec3 target = rec.p + random_in_hemisphere(rec.normal, local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    };
};

class metal : public material
{
public:
    vec3 albedo;
    float fuzz;

    __device__ metal(const vec3& a, float f) : albedo(a)
    {
        if (f < 1) fuzz = f;
        else fuzz = 1.0f;
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state)const override
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return(dot(scattered.direction(), rec.normal) > 0.0f);
    }
};

class dielectric : public material
{
public:
    float ior;
    __device__ dielectric(float index_of_refraction) : ior(index_of_refraction){}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state)const override
    {
        attenuation = color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ior) : ior;

        vec3 unit_direction = unit_vector(r_in.direction());
        float cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        vec3 direction;
        if (cannot_refract || schlick(cos_theta, refraction_ratio) > curand_uniform(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = ray(rec.p, direction);
        return true;
        
    }
};