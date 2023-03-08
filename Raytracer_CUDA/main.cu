#include "main.cuh"
constexpr char CHANNEL_NUM = 3;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func <<"\n" << cudaGetErrorString(result) << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}



__device__ color ray_color(const ray& r, hittable **world, curandState *rand_state)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0f,1.0f,1.0f);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) 
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray,rec, attenuation, scattered, rand_state))
            {
                cur_attenuation = cur_attenuation * attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            color c = (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
       
    return color(0.0f,0.0f,0.0f);
   
}


__global__ void render(vec3* fb, int img_w, int img_h,int sample_size, camera **cam, hittable **world)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    if ((i >= img_w) || (j >= img_h)) return;

    int pixel_index = j * img_w + i;
    
    curandState rand_state;
    curand_init(1984, pixel_index, 0, &rand_state);

    color col(0, 0, 0);
    
    for (int s = 0; s < sample_size; s++)
    {
        float u = float(i + curand_uniform(&rand_state)) / float(img_w);
        float v = float(j + curand_uniform(&rand_state)) / float(img_h);
        ray r = (*cam)->get_ray(u, v);
        col += ray_color(r, world, &rand_state);
    }
    col /= float(sample_size);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
 
}


__global__ void create_world(hittable** list, hittable** world, camera **cam, int img_width, int img_height)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        list[0] = new sphere(vec3(0, 0, -1), 0.5, new lambertian(vec3(0.8, 0.3, 0.3)));
        list[1] = new sphere(vec3(0, -100.5, -1), 100, new lambertian(vec3(0.8, 0.8, 0.0)));
        list[2] = new sphere(vec3(1, 0, -1), 0.5, new metal(vec3(0.8, 0.6, 0.2), 1.0));
        list[3] = new sphere(vec3(-1, 0, -1), 0.5, new dielectric(1.5));
        list[4] = new sphere(vec3(-1, 0, -1), -0.4, new dielectric(1.5));
        *world = new hittable_list(list, 4);
        *cam = new camera(vec3(-2, 2, 1),
            vec3(0, 0, -1),
            vec3(0, 1, 0),
            20.0,
            float(img_width) / float(img_height));
    }
}

__global__ void free_world(hittable** list, hittable** world, camera** cam)
{
    for (int i = 0; i < 5; i++) {
        delete ((sphere*)list[i])->mat_ptr;
        delete list[i];
    }
    delete* world;
    delete* cam;
}


int main()
{
    const double aspect_ratio = 16.0 / 9.0;
    const uint32_t image_width = 1920;
    const uint32_t image_height = 1080;
    const uint32_t sample_size = 100;
 


    size_t frame_buffer_size = image_width * image_height * sizeof(vec3);

    vec3* frame_buffer;

    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&frame_buffer), frame_buffer_size));


    hittable** list;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&list), 2 * sizeof(hittable*)));
    hittable** world;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&world), sizeof(hittable*)));
    camera** cam;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&cam), sizeof(camera*)));
    create_world << <1, 1 >> > (list, world,cam, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());





    int thread_x = 8;
    int thread_y = 8;

    dim3 blocks(image_width / thread_x + 1, image_height / thread_y + 1);
    dim3 threads(thread_x, thread_y);
    
    //uint8_t* pixel_array = new uint8_t [CHANNEL_NUM * image_width * image_height];

    std::unique_ptr<uint8_t[]>pixel_array = std::make_unique<uint8_t[]>(CHANNEL_NUM * image_width * image_height);

    render << <blocks,threads >> > (frame_buffer, image_width, image_height, sample_size, cam ,world);

   

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    int index = 0;
    for (int j = image_height - 1; j >= 0; j--) {

        for (int i = 0; i < image_width; i++) {

            int pixel_index = j*image_width+i;

            int ir = int(255.99 * frame_buffer[pixel_index].x());
            int ig = int(255.99 * frame_buffer[pixel_index].y());
            int ib = int(255.99 * frame_buffer[pixel_index].z());

            pixel_array[index++] = ir;
            pixel_array[index++] = ig;
            pixel_array[index++] = ib;
        }
    }
   
    int val = stbi_write_png("image.png", image_width, image_height, CHANNEL_NUM,pixel_array.get(), image_width * CHANNEL_NUM);
    
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (list, world,cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(list));
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(frame_buffer));
    cudaDeviceReset();

    return 0;
}

