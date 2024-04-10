CUDA Stream Compaction
======================
* Tested on personal pc:
  - Windows 11
  - i7-12700 @ 4.90GHz with 16GB RAM
  - RTX 4070 Ti Laptop 12GB

### feature

## Features

This project works on implementation and optimization of several parallelized scan-relevant algorithms with CUDA. A brief introduction of what these algorithms do and what implementations I have done:

- Scan: calculate the prefix sum (using arbitrary operator) of an array

  - [1] CPU scan with/without simulating parallelized scan 

  - [2] GPU naive scan 

  - [3] GPU work-efficient scan 

  - [4] GPU work-efficient scan with shared memory optimization & bank conflict reduction

  - [5] GPU scan using `thrust::exclusive_scan` 

- Stream compaction: remove elements that unmeet specific conditions from an array, and keep the rest compact in memory

  - [6] CPU stream compaction with/without CPU scan

  - [7] GPU stream compaction using [3] work-efficient 

  - [8] GPU stream compaction using [4] optimized work-efficient scan 



