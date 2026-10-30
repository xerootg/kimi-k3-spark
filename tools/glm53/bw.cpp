// bw: iGPU streaming-read bandwidth (sum of uint4 loads over N MB, best of R runs), the iGPU roofline
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
int main(int argc, char ** argv) {
    size_t mb = argc > 1 ? atoi(argv[1]) : 1024; size_t n = mb << 16; // uint4 count (16 B each)
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    auto * a = sycl::malloc_device<sycl::uint4>(n, q); auto * o = sycl::malloc_device<unsigned>(1 << 20, q);
    q.fill(a, sycl::uint4{1, 2, 3, 4}, n).wait();
    for (int wg : {256, 512}) for (int per : {4, 16, 64}) {
        size_t items = n / per; double best = 1e9;
        for (int r = 0; r < 6; r++) {
            auto t0 = std::chrono::steady_clock::now();
            q.parallel_for(sycl::nd_range<1>(items, wg), [=](sycl::nd_item<1> it) {
                size_t g = it.get_group(0), l = it.get_local_id(0); unsigned s = 0;
                for (int i = 0; i < per; i++) { sycl::uint4 v = a[(g * per + i) * wg + l]; s += v.x() ^ v.y() ^ v.z() ^ v.w(); }
                if (s == 12345) o[0] = s;
            }).wait();
            best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        printf("wg %d per %d: %.1f GB/s\n", wg, per, n * 16 / best / 1e9);
    }
}
