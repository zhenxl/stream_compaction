#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            odata[0] = 0;
            for (int i = 1; i < n; i++) {
                odata[i] = odata[i - 1] + idata[i-1];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int j = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[j++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return j;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int* tmp1 = new int[n], * tmp2 = new int[n];
            for (int i = 0; i < n; i++) {
                tmp1[i] = !!idata[i];
            }
            tmp2[0] = 0;
            for (int i = 1; i < n; i++) {
                tmp2[i] = tmp2[i - 1] + tmp1[i-1];
            }
            int cnt = tmp2[n - 1] + tmp1[n - 1];
            for (int i = 0; i < n; i++) {
                if (idata[i]) {
                    odata[tmp2[i]] = idata[i];
                }
            }
            timer().endCpuTimer();
            delete[] tmp1;
            delete[] tmp2;
            return cnt;
        }
    }
}
