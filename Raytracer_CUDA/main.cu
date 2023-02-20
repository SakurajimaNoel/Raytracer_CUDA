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


__device__ color ray_color(const ray& r)
{
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}



__global__ void render(vec3* fb, int img_w, int img_h, vec3 btm_lft_crnr, vec3 horizontal, vec3 vertical, vec3 origin)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);

    if ((i >= img_w) || (j >= img_h)) return;

    int pixel_index = j * img_w + i;
    
    float u = float(i) / float(img_w);
    float v = float(j) / float(img_h);

    ray r(origin, btm_lft_crnr + u * horizontal + v * vertical);

    fb[pixel_index] = ray_color(r);
}







int main()
{
    const int image_width = 256;
    const int image_height = 256;
    const double aspect_ratio = 16.0 / 9.0;
  
    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    float focal_length = 1.0;

    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(viewport_width, 0, 0);
    vec3 vertical = vec3(0, viewport_height, 0);
    point3 lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);


    size_t frame_buffer_size = image_width * image_height * sizeof(vec3);

    vec3* frame_buffer;

    checkCudaErrors(cudaMallocManaged(reinterpret_cast<void**>(&frame_buffer), frame_buffer_size));

    int thread_x = 8;
    int thread_y = 8;

    dim3 blocks(image_width / thread_x + 1, image_height / thread_y + 1);
    dim3 threads(thread_x, thread_y);
    
    uint8_t pixel_array[CHANNEL_NUM * image_width * image_height]{};

    //auto res = std::make_unique<uint8_t[]>(3 * image_width * image_height);
    render << <blocks,threads >> > (frame_buffer, image_width, image_height, lower_left_corner, horizontal, vertical, origin);

   

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
    checkCudaErrors(cudaFree(frame_buffer));


    int val = stbi_write_png("image.png", image_width, image_height, CHANNEL_NUM,&pixel_array, image_width * CHANNEL_NUM);
   
    return 0;
}

