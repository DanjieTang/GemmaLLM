#include <cuda_runtime.h>
#include <stdio.h>

__global__ void print_kernel(int num){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // print from every thread - output order is not guaranteed
    printf("thread %d: hello from GPU\n", tid);

    // printing only from a few threads avoids flooding the console
    if (tid < 8){
        printf("thread %d: blockIdx=%d threadIdx=%d\n", tid, blockIdx.x, threadIdx.x);
    }
}

int main(){
    print_kernel<<<2, 8>>>(16);

    cudaDeviceSynchronize();
    return 0;
}
