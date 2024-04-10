#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            thrust::host_vector<int> in(idata, idata + n);
            thrust::device_vector<int> device_in = in;
            thrust::device_vector<int> device_out(n);
            timer().startGpuTimer();
            // TODO use `thrust::exclusive_scan`
            // example: for device_vectors dv_in and dv_out:
            auto last = thrust::exclusive_scan(device_in.begin(), device_in.end(), device_out.begin());
            timer().endGpuTimer();
            int size = last - device_out.begin();
            thrust::host_vector<int> out = device_out;
            memcpy(odata, out.data(), size * sizeof(int));
        }
    }
}
