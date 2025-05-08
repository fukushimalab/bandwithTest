#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <type_traits>

#define CHECK(cmd) do { \
    cudaError_t e = cmd; \
    if (e != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(e) << std::endl; \
        return; \
    } \
} while(0)

// ===================================
// Unified kernel (float / float4)
// ===================================
template <typename T>
__global__ void shared_to_global_kernel(T* __restrict__ dst, int count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = threadIdx.x;

    if constexpr (std::is_same_v<T, float>) {
        extern __shared__ float smem[];
        if (tid < count) {
            smem[lid] = static_cast<float>(lid);
            __syncthreads();
            dst[tid] = smem[lid];
        }
    } else if constexpr (std::is_same_v<T, float4>) {
        extern __shared__ __align__(16) float4 smem4[];
        if (tid < count) {
            smem4[lid] = make_float4(1.0f, 2.0f, 3.0f, 4.0f);
            __syncthreads();
            dst[tid] = smem4[lid];
        }
    }
}

// ===================================
// Benchmark runner
// ===================================
template <typename T>
void run_benchmark(std::ofstream& out, const std::string& label, const std::vector<size_t>& sizes) {
    const int blockSize = 256;
    for (auto size : sizes) {
        int numElems = size / sizeof(T);
        T* d_dst;
        CHECK(cudaMalloc(&d_dst, size));
        CHECK(cudaMemset(d_dst, 0, size));

        int gridSize = (numElems + blockSize - 1) / blockSize;
        size_t smemSize = blockSize * sizeof(T);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Warm-up
        shared_to_global_kernel<T><<<gridSize, blockSize, smemSize>>>(d_dst, numElems);
        cudaDeviceSynchronize();

        // Benchmark
        cudaEventRecord(start);
        shared_to_global_kernel<T><<<gridSize, blockSize, smemSize>>>(d_dst, numElems);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        double gb = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = gb / (ms / 1000.0);

        out << (size / 1024) << "," << label << "," << bandwidth << std::endl;

        cudaFree(d_dst);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    std::ofstream out("shared_to_global.csv");
    out << "SizeKB,Type,BandwidthGBps\n";

    std::vector<size_t> sizes = {
        64 << 10, 128 << 10, 256 << 10, 512 << 10,
        1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20
    };

    run_benchmark<float>(out, "float_sync", sizes);
    run_benchmark<float4>(out, "float4_sync", sizes);

    out.close();
    return 0;
}

