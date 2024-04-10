#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

#define BLOCKSIZE 32

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__  void kernNaiveScan(int N, int two_d_1, int *input, int *output) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            if (index >= two_d_1) {
                output[index] = input[index - two_d_1] + input[index];
            }
            else {
                output[index] = input[index];
            }
        }

        __global__ void kernShiftArray(int N, int* input, int* output) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= N) return;
            output[index] = index == 0 ? 0 : input[index - 1];
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
           
            // TODO
            int* g_idata, *g_odata;
            cudaMalloc((void **)&g_idata, n * sizeof(int));
            cudaMalloc((void **)&g_odata, n * sizeof(int));
            cudaMemcpy((void *)g_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            //nvtxRangePushA("Naive scan");
            timer().startGpuTimer();
            int max_d = ilog2ceil(n);
            for (int i = 1; i <= max_d; i++) {
                kernNaiveScan <<<(n + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >>> (n, 1 << (i-1), g_idata, g_odata);
                std::swap(g_odata, g_idata);
            }
            kernShiftArray <<<(n + BLOCKSIZE - 1) / BLOCKSIZE, BLOCKSIZE >>> (n, g_idata, g_odata);
            timer().endGpuTimer();
            cudaMemcpy(odata, g_odata, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(g_idata);
            cudaFree(g_odata);
        }
    }
}
