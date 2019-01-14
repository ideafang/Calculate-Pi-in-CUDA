# CUDA for Monte Carlo estimation of Pi
## English
CUDA code for Monte Carlo estimation of Pi(see https://en.wikipedia.org/wiki/Monte_Carlo_method)

The goal here was not to set a new record for the number of decimal places, but rather to demonstrate the benefits of GPU computing, by using CUDA(see https://en.wikipedia.org/wiki/CUDA)to parallelize a traditional serial algorithm used to estimate Pi.

To run you must have an NVIDIA GPU capable of compiling and running CUDA.

Using Monte Carlo with 2^25 random points to calculate estimation of Pi.the code is using 512 threads per block, 128 blocks and each threads will process 512 points.

The code includes CPU serial computing and four layer-by-layer optimized versions of the GPU. Use clock() to record the calculation time, and finally output the calculation results and time consumption of each version, thus reflecting the advantages of CUDA parallel computing.

The four optimizations of the GPU are used (2. shared memory, 3. memory coalesce + shared memory, 4. memory coalesce). The final result varies with machine performance, but it is nearly 200 times faster than the CPU, and the effect is remarkable.

## 中文

CUDA使用蒙特卡洛法计算Pi值（方法见：https://zh.wikipedia.org/wiki/%E8%92%99%E5%9C%B0%E5%8D%A1%E7%BE%85%E6%96%B9%E6%B3%95 ）

这里的目标不是计算更多的小数位数，而是通过使用CUDA（ https://zh.wikipedia.org/wiki/CUDA ）来证明GPU计算的好处

运行环境：一台安装好CUDA并具有NVIDIA GPU显卡的电脑，CUDA的具体配置方法自行百度。

使用2^25规模的随机点通过蒙特卡洛方法来计算Pi值。代码中使用了128个线程块（Blocks）,每个线程块中包括512个线程（threadsPerBlock），每个线程计算512个随机点。

代码包括CPU串行计算和GPU的4个逐层优化版本。使用clock()分别对计算时间计时,最后输出各个版本的计算结果和耗时，从而体现出CUDA并行计算的优势。

GPU的四次优化分别使用了（2. 共享内存, 3. 共享内存+合并访问, 4. 合并访问）最终优化结果因机器性能而异，但比CPU要快上近200倍，效果显著。