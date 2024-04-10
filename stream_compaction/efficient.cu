#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "efficient.h"
#include <iostream>
#define DEBUG_PRINT 0
#define BLOCK_SIZE 128
// For BCAO
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5

#define ZERO_BANK_CONFLICTS 0
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) ( ((n) >> LOG_NUM_BANKS) + ((n) >> (2 * LOG_NUM_BANKS)) )
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif
namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int *data, int n, int stride) {
            int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
            index *= stride;
            if (index > n) return;
            int real_idx = index - 1;
            //printf("real_idx: %d: before idx: %d \n", real_idx,  real_idx - stride / 2);
            //printf("real_val: %d before value: %d\n", data[real_idx], data[real_idx - stride / 2]);
            data[real_idx] += data[real_idx - stride / 2];
        }

        __global__ void kernDownSweep(int* data, int n, int stride) {
            int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
            index = index * stride;
            if (index > n) return;
            int right_child_idx = index - 1;
            int left_child_idx = right_child_idx - stride / 2;
            int tmp = data[left_child_idx];
            data[left_child_idx] = data[right_child_idx];
            data[right_child_idx] += tmp;
        }

        __global__ void blockSharedScan(int* g_idata, int* g_odata, int n, int* SUM) {
            __shared__ int temp[BLOCK_SIZE << 1];
            int thid = threadIdx.x;
            //printf("thid %d\n", thid);
            int offset = 1;
            int blockOffset = BLOCK_SIZE * blockIdx.x * 2;
            if (blockOffset + thid * 2 < n) {
                temp[thid * 2] = g_idata[blockOffset + thid * 2];
            }

            if (blockOffset + thid * 2 + 1 < n) {
                temp[thid * 2 + 1] = g_idata[blockOffset + thid * 2 + 1];
            }

            for (int d = BLOCK_SIZE; d > 0; d >>= 1) {
     
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (thid * 2 + 1) - 1;
                    //printf("ai: %d ", ai);
                    int bi = offset * (thid * 2 + 2) - 1;
                   /* printf("ai: %d bi: %d \n", ai, bi);*/
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }
            
            if (thid == 0) {
                if (SUM != NULL) {
                    SUM[blockIdx.x] = temp[(BLOCK_SIZE << 1) - 1];
                }
                temp[(BLOCK_SIZE <<1) - 1] = 0;
                //temp[1] = 0;
            }
           
            for (int d = 1; d < BLOCK_SIZE << 1; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * ((thid * 2) + 1) - 1;
                    int bi = offset * ((thid * 2) + 2) - 1;

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                    //printf("temp bi offset %d %d %d\n", offset, bi, temp[bi]);
                }
            //    //break;
            }

            __syncthreads();
            if (blockOffset + (thid * 2) < n) {
                g_odata[blockOffset + (thid * 2)] = temp[(thid * 2)]; // write results to device memory
            }
            if (blockOffset + (thid * 2) + 1 < n) {
                g_odata[blockOffset + ((thid * 2) + 1)] = temp[(thid * 2) + 1];
            }
        }

        __global__ void BACO_blockSharedScan(int* g_idata, int* g_odata, int n, int* SUM) {
            __shared__ int temp[(BLOCK_SIZE << 1) + BLOCK_SIZE];
            int thid = threadIdx.x;
            //printf("thid %d\n", thid);
            int offset = 1;
            int blockOffset = BLOCK_SIZE * blockIdx.x * 2;
            int ai = threadIdx.x;
            int bi = threadIdx.x + BLOCK_SIZE;
            int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            if (blockOffset + ai < n) {
                temp[ai + bankOffsetA] = g_idata[blockOffset + ai];
            }

            if (blockOffset + bi < n) {
                temp[bi + bankOffsetB] = g_idata[blockOffset + bi];
            }

            for (int d = BLOCK_SIZE; d > 0; d >>= 1) {

                __syncthreads();
                if (thid < d) {
                    int ai = offset * (thid * 2 + 1) - 1;
                    //printf("ai: %d ", ai);
                    int bi = offset * (thid * 2 + 2) - 1;
                    /* printf("ai: %d bi: %d \n", ai, bi);*/
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
                    temp[bi] += temp[ai];
                }
                offset <<= 1;
            }

            if (thid == 0) {
                if (SUM != NULL) {
                    SUM[blockIdx.x] = temp[(BLOCK_SIZE << 1) - 1 +  CONFLICT_FREE_OFFSET((BLOCK_SIZE << 1) - 1)];
                }
                temp[(BLOCK_SIZE << 1) - 1 + CONFLICT_FREE_OFFSET((BLOCK_SIZE << 1) - 1)] = 0;
                //temp[1] = 0;
            }

            for (int d = 1; d < BLOCK_SIZE << 1; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * ((thid * 2) + 1) - 1;
                    int bi = offset * ((thid * 2) + 2) - 1;
                    ai += CONFLICT_FREE_OFFSET(ai);
                    bi += CONFLICT_FREE_OFFSET(bi);
                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                    //printf("temp bi offset %d %d %d\n", offset, bi, temp[bi]);
                }
                //    //break;
            }

            __syncthreads();
            if (blockOffset + ai < n) {
                g_odata[blockOffset + ai] = temp[ai + bankOffsetA]; // write results to device memory
            }
            if (blockOffset + bi < n) {
                g_odata[blockOffset + bi] = temp[bi + bankOffsetB];
            }
        }

        __global__ void uniformAdd(int n, int *g_out, int *INCR) {
            int index = threadIdx.x + BLOCK_SIZE * 2 * blockIdx.x;
            int valueToAdd = INCR[blockIdx.x];
            if (index < n) {
                g_out[index] += valueToAdd;
            } 
            if (index + BLOCK_SIZE < n) {
                g_out[index + BLOCK_SIZE] += valueToAdd;
            }
        }


        void printHostData(int* d_data, int size) {
            int* h_data = new int[size];
            cudaMemcpy(h_data, d_data, size * sizeof(int), cudaMemcpyDeviceToHost);

            // Print the data
            for (int i = 0; i < size; ++i) {
                std::cout << h_data[i] << " ";
            }
            std::cout << std::endl;

            delete[] h_data; // Free host memory
        }

        void workEfficientScanInPlace(int *d_data, int size) {
            if (size != ceilPow2(size)) {
                throw std::runtime_error("workEfficientInPlace: not power of 2");
            }
            for (int stride = 2; stride <= size; stride <<= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernUpSweep <<<blockNum, blockSize >>> (d_data, size, stride);
                cudaDeviceSynchronize();
#if DEBUG_PRINT
                printf("\n================================\n");
                printf("UP SWEEP\n");
                printf("================================\n\n");
                printHostData(d_data, size);
               
#endif
            }
            cudaMemset(d_data + size - 1, 0, sizeof(int));
            for (int stride = size; stride >= 2; stride >>= 1) {
                int num = size / stride;
                int blockSize = Common::getDynamicBlockSizeEXT(num);
                int blockNum = ceilDiv(num, blockSize);
                kernDownSweep <<< blockNum, blockSize >>> (d_data, size, stride);
            }
        }

        void sharedScan(int n, int* odata, const int* idata) {
            int size = ceilPow2(n);
            int level = 0;
            std::vector<int*> d_sum_buffers;
            std::vector<int*> d_incr_buffers;
            std::vector<int> buffer_sizes;
            int tmp = size / (BLOCK_SIZE * 2);
            while (tmp > 1) {
                //todo£º add tmp = 0 contidion
                int* d_SUMS, * d_INCRS;
                d_SUMS = NULL;
                d_INCRS = NULL;
                cudaMalloc((void**)&d_SUMS, tmp * sizeof(int));
                cudaMalloc((void**)&d_INCRS, tmp * sizeof(int));
                d_sum_buffers.push_back(d_SUMS);
                d_incr_buffers.push_back(d_INCRS);
                buffer_sizes.push_back(tmp);
                tmp = tmp / (BLOCK_SIZE * 2);
                level += 1;
            }
            int* d_odata, * d_idata;
            cudaMalloc((void**)&d_odata, size * sizeof(int));
            cudaMalloc((void**)&d_idata, size * sizeof(int));
            cudaMemcpy((void*)d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int blocksPerGrid = size / (BLOCK_SIZE * 2);
            if (d_sum_buffers.size() > 0) {
#if ZERO_BANK_CONFLICTS
                BACO_blockSharedScan << <blocksPerGrid, BLOCK_SIZE >> > (d_idata, d_odata, size, d_sum_buffers[0]);
#else
                blockSharedScan <<<blocksPerGrid, BLOCK_SIZE >> > (d_idata, d_odata, size, d_sum_buffers[0]);
#endif
#if DEBUG_PRINT
                printf("\n================================\n");
                printf("block shared \n");
                printf("================================\n\n");
                printHostData(d_odata, size);
                printHostData(d_sum_buffers[0], buffer_sizes[0]);

#endif
                if (level == 1) {
                    timer().startGpuTimer();
                    //std::cout << "level: " << level << " buffersize " << buffer_sizes[0] << std::endl;
#if DEBUG_PRINT
                    std::cout << "level: " << level << " buffersize " << buffer_sizes[0] << std::endl;
#endif
                    int gridNum = 1 + (buffer_sizes[0] - 1) / BLOCK_SIZE;
#if ZERO_BANK_CONFLICTS
                    BACO_blockSharedScan <<<gridNum, BLOCK_SIZE >>> (d_sum_buffers[0], d_incr_buffers[0], buffer_sizes[0], NULL);
#if DEBUG_PRINT
                    printf("\n==============\n");
                    printf("after uniform\n");
                    printHostData(d_incr_buffers[0], buffer_sizes[0]);
#endif
#else
                    blockSharedScan << <gridNum, BLOCK_SIZE >> > (d_sum_buffers[0], d_incr_buffers[0], buffer_sizes[0], NULL);
#endif
                    uniformAdd <<<blocksPerGrid, BLOCK_SIZE >>> (size, d_odata, d_incr_buffers[0]);
                    timer().endGpuTimer();
                    cudaMemcpy(odata, d_odata, size * sizeof(int), cudaMemcpyDeviceToHost);

                    cudaFree(d_sum_buffers[0]);
                    cudaFree(d_incr_buffers[0]);
                    cudaFree(d_odata);
                    cudaFree(d_idata);
                }
                else {
#if DEBUG_PRINT
                    std::cout << "level: " << level << " buffersize " << buffer_sizes[0] << std::endl;
#endif
                    timer().startGpuTimer();
                    for (int i = 0; i < level-1; i++) {
                        int gridNum = 1 + (buffer_sizes[i] - 1) / BLOCK_SIZE;
#if ZERO_BANK_CONFLICTS
                        BACO_blockSharedScan << <gridNum, BLOCK_SIZE >> > (d_idata, d_odata, size, d_sum_buffers[0]);
#else
                        blockSharedScan <<<gridNum, BLOCK_SIZE >>> (d_sum_buffers[i], d_incr_buffers[i], buffer_sizes[i], (d_sum_buffers[i+1]));
#endif
                    }
                    int gridNum = 1 + (buffer_sizes[level - 1] - 1) / BLOCK_SIZE;
#if ZERO_BANK_CONFLICTS
                    BACO_blockSharedScan << <gridNum, BLOCK_SIZE >> > (d_idata, d_odata, size, d_sum_buffers[0]);
#else
                    blockSharedScan << <gridNum, BLOCK_SIZE >> > (d_sum_buffers[level-1], d_incr_buffers[level-1], buffer_sizes[level-1], NULL);
#endif
                    for (int i = level - 1; i > 0; i--) {
                        int gridNum = 1 + (buffer_sizes[i-1] - 1) / BLOCK_SIZE;
                        uniformAdd <<<gridNum, BLOCK_SIZE >>> (buffer_sizes[i-1], d_incr_buffers[i - 1], d_incr_buffers[i]);
                    }
                    uniformAdd << <blocksPerGrid, BLOCK_SIZE >> > (size, d_odata, d_incr_buffers[0]);
                    timer().endGpuTimer();
#if DEBUG_PRINT
                    printf("\n==============\n");
                    printf("after uniform\n");
                    printHostData(d_incr_buffers[0], buffer_sizes[0]);
                    printHostData(d_odata, size);
                    printf("\n==============\n");

#endif
                    cudaMemcpy(odata, d_odata, size * sizeof(int), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < level; i++) {
                        cudaFree(d_sum_buffers[i]);
                        cudaFree(d_incr_buffers[i]);
                    }
                }
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            
            // TODO
            int size = ceilPow2(n);
            int* g_data;
            cudaMalloc((void **)&g_data, size * sizeof(int));
            cudaMemcpy((void*)g_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            timer().startGpuTimer();
            //sharedScan(n, odata, idata);
            workEfficientScanInPlace(g_data, size);
            timer().endGpuTimer();
            cudaMemcpy(odata, g_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(g_data);
           
        }

      

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int* d_in, * d_out;
            cudaMalloc((void**)&d_in, n * sizeof(int));
            cudaMalloc((void**)&d_out, n * sizeof(int));
            cudaMemcpy((void*)d_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int* d_indices;
            int size = ceilPow2(n);
            int blockSize = Common::getDynamicBlockSizeEXT(n);
            int blockNum = ceilDiv(n, blockSize);
            cudaMalloc((void **)&d_indices, size * sizeof(int));
            Common::kernMapToBoolean<<<blockNum, blockSize>>>(n, d_indices, d_in);
            workEfficientScanInPlace(d_indices, size);
            Common::kernScatter<<<blockNum, blockSize >>>(n, d_out, d_in, d_in, d_indices);

            int compactedSize;
            cudaMemcpy(&compactedSize, d_indices + n -1, sizeof(int), cudaMemcpyDeviceToHost);
            compactedSize += (idata[n - 1] != 0);
            cudaMemcpy(odata, d_out, compactedSize * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(d_in);
            cudaFree(d_out);
            timer().endGpuTimer();
            return compactedSize;
        }
    }
}
