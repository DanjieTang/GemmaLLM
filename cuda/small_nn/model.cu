#include <cuda_runtime.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

constexpr int TILE_SIZE = 16;


#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t error = (call);                                   \
        if (error != cudaSuccess) {                                   \
            std::cerr                                                 \
                << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                << ": " << cudaGetErrorString(error) << '\n';         \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

std::vector<float> load_float32_file(
    const std::string& filename,
    std::size_t expected_elements)
{
    std::ifstream file(
        filename,
        std::ios::binary | std::ios::ate
    );

    if (!file) {
        throw std::runtime_error(
            "Could not open file: " + filename
        );
    }

    const std::streamsize byte_count = file.tellg();
    const std::streamsize expected_bytes =
        static_cast<std::streamsize>(
            expected_elements * sizeof(float)
        );

    if (byte_count != expected_bytes) {
        throw std::runtime_error(
            filename + " has " +
            std::to_string(byte_count) +
            " bytes; expected " +
            std::to_string(expected_bytes)
        );
    }

    file.seekg(0, std::ios::beg);

    std::vector<float> values(expected_elements);

    if (!file.read(
            reinterpret_cast<char*>(values.data()),
            expected_bytes))
    {
        throw std::runtime_error(
            "Could not read file: " + filename
        );
    }

    return values;
}

__global__ void linear(float* input_tensor, float* weights, float* bias, float* output_tensor, int batch_size, int input_dim, int output_dim){
    int local_x_index = threadIdx.x;
    int local_y_index = threadIdx.y;
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float tile_matrix[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_input[TILE_SIZE][TILE_SIZE];

    int K = input_dim / TILE_SIZE;

    float sum = bias[global_y_index];
    for (int i = 0; i < K; i++){
        // Load into memory
        tile_matrix[local_y_index][local_x_index] = weights[global_y_index * input_dim + i * TILE_SIZE + local_x_index];
        tile_input[local_y_index][local_x_index] = input_tensor[(local_y_index + i * TILE_SIZE) * batch_size + global_x_index];

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++){
            sum += tile_matrix[local_y_index][j] * tile_input[j][local_x_index];
        }
        __syncthreads();
    }

    output_tensor[global_y_index * batch_size + global_x_index] = sum;
}

__global__ void relu(float* tensor, int width){
    int global_x_index = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y_index = blockIdx.y * blockDim.y + threadIdx.y;

    if (tensor[global_y_index * width + global_x_index] < 0){
        tensor[global_y_index * width + global_x_index] = 0;
    }
}

int main(){
    int batch_size = 32;
    int input_dim = 4096;
    int hidden_dim = 256;
    int output_dim = 16;

    size_t input_size = batch_size * input_dim * sizeof(float);
    size_t layer1_weight_size = input_dim * hidden_dim * sizeof(float);
    size_t layer1_bias_size = hidden_dim * sizeof(float);
    size_t layer2_weight_size = hidden_dim * output_dim * sizeof(float);
    size_t layer2_bias_size = output_dim * sizeof(float);
    size_t output_size = batch_size * output_dim * sizeof(float);
    size_t hidden_size = batch_size * hidden_dim * sizeof(float);

    const std::vector<float> h_input_tensor = load_float32_file("tensors/input.bin", batch_size * input_dim);
    const std::vector<float> h_layer1_weights = load_float32_file("tensors/layer1_weights.bin", input_dim * hidden_dim);
    const std::vector<float> h_layer1_bias = load_float32_file("tensors/layer1_bias.bin", hidden_dim);
    const std::vector<float> h_layer2_weights = load_float32_file("tensors/layer2_weights.bin", hidden_dim * output_dim);
    const std::vector<float> h_layer2_bias = load_float32_file("tensors/layer2_bias.bin", output_dim);
    std::vector<float> h_output_tensor = load_float32_file("tensors/output.bin", batch_size * output_dim);

    float *d_input_tensor, *d_layer1_weights, *d_layer1_bias, *d_layer2_weights, *d_layer2_bias, *d_output_tensor, *d_hidden_tensor;

    CUDA_CHECK(cudaMalloc(&d_input_tensor, input_size));
    CUDA_CHECK(cudaMalloc(&d_layer1_weights, layer1_weight_size));
    CUDA_CHECK(cudaMalloc(&d_layer1_bias, layer1_bias_size));
    CUDA_CHECK(cudaMalloc(&d_layer2_weights, layer2_weight_size));
    CUDA_CHECK(cudaMalloc(&d_layer2_bias, layer2_bias_size));
    CUDA_CHECK(cudaMalloc(&d_output_tensor, output_size));
    CUDA_CHECK(cudaMalloc(&d_hidden_tensor, hidden_size));

    CUDA_CHECK(cudaMemcpy(d_input_tensor, h_input_tensor.data(), input_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layer1_weights, h_layer1_weights.data(), layer1_weight_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layer1_bias, h_layer1_bias.data(), layer1_bias_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layer2_weights, h_layer2_weights.data(), layer2_weight_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_layer2_bias, h_layer2_bias.data(), layer2_bias_size, cudaMemcpyHostToDevice));

    dim3 block_size_hid(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_hid(batch_size / TILE_SIZE, hidden_dim / TILE_SIZE);

    linear<<<grid_size_hid, block_size_hid>>>(d_input_tensor, d_layer1_weights, d_layer1_bias, d_hidden_tensor, batch_size, input_dim, hidden_dim);
    CUDA_CHECK(cudaGetLastError());
    relu<<<grid_size_hid, block_size_hid>>>(d_hidden_tensor, batch_size);
    CUDA_CHECK(cudaGetLastError());

    dim3 block_size_out(TILE_SIZE, TILE_SIZE);
    dim3 grid_size_out(batch_size / TILE_SIZE, output_dim / TILE_SIZE);

    linear<<<grid_size_out, block_size_out>>>(d_hidden_tensor, d_layer2_weights, d_layer2_bias, d_output_tensor, batch_size, hidden_dim, output_dim);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> h_computed_output(batch_size * output_dim);
    CUDA_CHECK(cudaMemcpy(h_computed_output.data(), d_output_tensor, output_size, cudaMemcpyDeviceToHost));

    // h_output_tensor still holds the expected values from tensors/output.bin;
    // dump the GPU result separately so validate.py can compare the two.
    std::ofstream computed_file("tensors/computed_output.bin", std::ios::binary);
    if (!computed_file) {
        std::cerr << "Could not open tensors/computed_output.bin for writing\n";
        return EXIT_FAILURE;
    }
    computed_file.write(
        reinterpret_cast<const char*>(h_computed_output.data()),
        static_cast<std::streamsize>(output_size)
    );

    std::cout << h_computed_output[2];
}
