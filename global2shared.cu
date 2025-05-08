#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <type_traits>

#define CHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(e) << std::endl; return; \
  } \
} while(0)

// =========================
// Unified Benchmark Kernel
// =========================
template <typename T>
__global__ void global_to_shared_kernel(const T* __restrict__ src, int count, bool use_async) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lid = threadIdx.x;

    if constexpr (std::is_same_v<T, float>) {
        extern __shared__ __align__(32) float smem[];
        if (tid < count) {
#if __CUDA_ARCH__ >= 800
            if (use_async) {
                void* smem_ptr = static_cast<void*>(&smem[lid]);
                const void* src_ptr = static_cast<const void*>(&src[tid]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::
                             "r"(reinterpret_cast<int>(smem_ptr)), "r"(reinterpret_cast<int>(src_ptr)), "n"(sizeof(float)));
            } else {
                smem[lid] = src[tid];
            }
#else
            smem[lid] = src[tid]; // fallback
#endif
        }
        __syncthreads();

    } else if constexpr (std::is_same_v<T, float4>) {
        extern __shared__ float4 smem4[];
        if (tid < count) {
#if __CUDA_ARCH__ >= 800
            if (use_async) {
                void* smem_ptr = static_cast<void*>(&smem4[lid]);
                const void* src_ptr = static_cast<const void*>(&src[tid]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], %2;" ::
                             "r"(reinterpret_cast<int>(smem_ptr)), "r"(reinterpret_cast<int>(src_ptr)), "n"(sizeof(float4)));
            } else {
                smem4[lid] = src[tid];
            }
#else
            smem4[lid] = src[tid]; // fallback
#endif
        }
	asm volatile("cp.async.commit_group;");
	asm volatile("cp.async.wait_group 0;");
        __syncthreads();
    }
}

// =========================
// Benchmark Runner
// =========================
template <typename T>
void run_benchmark(std::ofstream& out, const std::string& label, const std::vector<size_t>& sizes, bool use_async) {
    const int blockSize = 256;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    for (auto size : sizes) {
        int numElems = size / sizeof(T);
        T* d_src;
        CHECK(cudaMalloc(&d_src, size));
        CHECK(cudaMemset(d_src, 1, size));

        int gridSize = (numElems + blockSize - 1) / blockSize;
        size_t smemSize = blockSize * sizeof(T);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        
        if (prop.major < 8 && use_async) {
            std::cout << "Skipping async on pre-Ampere device\n";
            continue;
        }

        global_to_shared_kernel<T><<<gridSize, blockSize, smemSize>>>(d_src, numElems, use_async);
        cudaDeviceSynchronize();

        cudaEventRecord(start);
        global_to_shared_kernel<T><<<gridSize, blockSize, smemSize>>>(d_src, numElems, use_async);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        double gb = static_cast<double>(size) / (1024.0 * 1024.0 * 1024.0);
        double bandwidth = gb / (ms / 1000.0);

        out << (size / 1024) << "," << label << "," << bandwidth << std::endl;

        cudaFree(d_src);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
}

int main() {
    std::ofstream out("global_to_shared_async_constexpr.csv");
    out << "SizeKB,Type,BandwidthGBps\n";

    std::vector<size_t> sizes = {
        64 << 10, 128 << 10, 256 << 10, 512 << 10,
        1 << 20, 2 << 20, 4 << 20, 8 << 20, 16 << 20, 32 << 20, 64 << 20
    };

    run_benchmark<float>(out, "float_sync", sizes, false);
    run_benchmark<float4>(out, "float4_sync", sizes, false);
    //run_benchmark<float>(out, "float_async", sizes, true);
    //run_benchmark<float4>(out, "float4_async", sizes, true);

    out.close();
    return 0;
}

