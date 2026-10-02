/* cuda_pricing_kernel.cu -- the same pool pricer as a raw CUDA kernel.
 *
 * vectorized_pricing.py fuses the path loop but keeps the month loop in
 * Python, so PyTorch issues one kernel launch per month -- 360 launches no
 * matter how many paths are in flight, which is why the GPU loses to the
 * vectorized CPU below a few thousand paths.
 *
 * A raw kernel does not have that problem: one thread prices one path, and
 * the month loop runs *inside* the thread, so the whole 360-month recursion
 * is one kernel launch regardless of path count. This is the CUDA-core
 * shape the chapter describes -- an elementwise, branch-heavy scalar loop,
 * not a matrix multiply, so Tensor Cores have nothing to do here.
 *
 * Build and run (requires the CUDA toolkit):
 *
 *     nvcc -O3 -arch=native cuda_pricing_kernel.cu -o cuda_pricing_kernel
 *     ./cuda_pricing_kernel
 */

#include <cstdio>
#include <cmath>
#include <curand_kernel.h>

__global__ void price_paths_kernel(unsigned long long seed,
                                    double r0, double a, double theta, double sigma,
                                    double note_rate, double balance,
                                    double base_smm, double sensitivity, double spread,
                                    int months, int n_paths, double *prices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_paths) return;

    /* Each thread owns one path's rate draws -- no cross-thread state. */
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, idx, 0, &rng);

    const double dt = 1.0 / 12.0;
    const double i_rate = note_rate / 12.0;
    const double pmt = balance * i_rate / (1.0 - pow(1.0 + i_rate, -(double)months));

    double r_t = r0, outstanding = balance, price = 0.0, log_discount = 0.0;

    for (int k = 1; k <= months; ++k) {
        double dW = curand_normal_double(&rng) * sqrt(dt);
        r_t = r_t + a * (theta - r_t) * dt + sigma * dW;

        log_discount -= (r_t + spread) * dt;
        double discount = exp(log_discount);

        double interest = outstanding * i_rate;
        double scheduled = fmax(fmin(pmt - interest, outstanding), 0.0);
        double incentive = fmax(note_rate - r_t, 0.0);
        double smm = base_smm + sensitivity * incentive * incentive;
        double prepay = fmax(outstanding - scheduled, 0.0) * smm;
        double principal = scheduled + prepay;

        price += (interest + principal) * discount;
        outstanding = fmax(outstanding - principal, 0.0);
    }

    prices[idx] = price;
}

int main() {
    const double R0 = 0.045, A = 0.5, THETA = 0.045, SIGMA = 0.025;
    const double NOTE_RATE = 0.065, BALANCE = 100.0, SPREAD = 0.0;
    const double BASE_SMM = 0.004, SENSITIVITY = 5.0;
    const int MONTHS = 360, N_PATHS = 400000;

    double *d_prices;
    cudaMalloc(&d_prices, N_PATHS * sizeof(double));

    const int threads = 256;
    const int blocks = (N_PATHS + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Warmup: pays for context creation and kernel compilation once, same
     * reason _time() in vectorized_pricing.py discards a first call. */
    price_paths_kernel<<<blocks, threads>>>(0, R0, A, THETA, SIGMA, NOTE_RATE,
        BALANCE, BASE_SMM, SENSITIVITY, SPREAD, MONTHS, N_PATHS, d_prices);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    price_paths_kernel<<<blocks, threads>>>(1, R0, A, THETA, SIGMA, NOTE_RATE,
        BALANCE, BASE_SMM, SENSITIVITY, SPREAD, MONTHS, N_PATHS, d_prices);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double *h_prices = new double[N_PATHS];
    cudaMemcpy(h_prices, d_prices, N_PATHS * sizeof(double), cudaMemcpyDeviceToHost);

    double mean = 0.0;
    for (int i = 0; i < N_PATHS; ++i) mean += h_prices[i];
    mean /= N_PATHS;

    printf("%d paths x %d months, one kernel launch\n", N_PATHS, MONTHS);
    printf("  fused kernel (GPU)   %9.1f ms\n", (double)ms);
    printf("  mean pool price      %9.4f\n", mean);

    delete[] h_prices;
    cudaFree(d_prices);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
