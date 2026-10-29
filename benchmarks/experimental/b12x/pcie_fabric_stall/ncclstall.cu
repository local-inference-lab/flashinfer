// ncclstall: per-operation timing of NCCL all-reduce on several GPUs from one process (diagnostic tool).
// Each GPU replays a CUDA graph of K all-reduces; a one-thread kernel stamps %globaltimer before and after
// every all-reduce, so stalls of single operations are visible. Optional spin kernel emulates compute gaps.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <unistd.h>

#define CK(x)                                                                                     \
  do {                                                                                            \
    cudaError_t e_ = (x);                                                                         \
    if (e_ != cudaSuccess) {                                                                      \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));          \
      exit(1);                                                                                    \
    }                                                                                             \
  } while (0)
#define NK(x)                                                                                     \
  do {                                                                                            \
    ncclResult_t r_ = (x);                                                                        \
    if (r_ != ncclSuccess) {                                                                      \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, ncclGetErrorString(r_));          \
      exit(1);                                                                                    \
    }                                                                                             \
  } while (0)

__global__ void stamp(unsigned long long* buf, unsigned* cnt) {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  buf[(*cnt)++] = t;
}
__global__ void spin(unsigned long long ns) {
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
__global__ void fill(__nv_bfloat16* p, long n, float v) {
  for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += (long)gridDim.x * blockDim.x)
    p[i] = __float2bfloat16(v);
}

static double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  return v[(size_t)std::min<double>(v.size() - 1, q * (v.size() - 1) + 0.5)];
}

int main(int argc, char** argv) {
  std::vector<int> gpus = {0, 1, 2, 3};
  int kib = 80, K = 64, reps = 2000;
  double gap_us = 0;
  std::string dump;
  for (int i = 1; i < argc; ++i) {
    std::string k = argv[i];
    if (k == "--gpus") {
      gpus.clear();
      std::string s = argv[++i];
      size_t p = 0;
      while (p < s.size()) { size_t q = s.find(',', p); if (q == std::string::npos) q = s.size(); gpus.push_back(atoi(s.substr(p, q - p).c_str())); p = q + 1; }
    } else if (k == "--kib") kib = atoi(argv[++i]);
    else if (k == "--ops-per-graph") K = atoi(argv[++i]);
    else if (k == "--reps") reps = atoi(argv[++i]);
    else if (k == "--gap-us") gap_us = atof(argv[++i]);
    else if (k == "--dump") dump = argv[++i];
    else { fprintf(stderr, "unknown %s\n", k.c_str()); return 2; }
  }
  const int R = gpus.size();
  const long n = (long)kib * 1024 / 2;
  std::vector<ncclComm_t> comms(R);
  NK(ncclCommInitAll(comms.data(), R, gpus.data()));
  int ver = 0;
  ncclGetVersion(&ver);
  std::vector<cudaStream_t> st(R);
  std::vector<__nv_bfloat16*> buf(R);
  std::vector<unsigned long long*> ts(R);
  std::vector<unsigned*> cnt(R);
  const long nst = 2L * K * reps;
  for (int r = 0; r < R; ++r) {
    CK(cudaSetDevice(gpus[r]));
    CK(cudaStreamCreateWithFlags(&st[r], cudaStreamNonBlocking));
    CK(cudaMalloc(&buf[r], n * 2));
    CK(cudaMalloc(&ts[r], nst * 8));
    CK(cudaMalloc(&cnt[r], 4));
    CK(cudaMemset(cnt[r], 0, 4));
    fill<<<64, 256, 0, st[r]>>>(buf[r], n, 0.0f);
    CK(cudaStreamSynchronize(st[r]));
  }
  // Warm up outside capture (connection setup).
  for (int w = 0; w < 20; ++w) {
    NK(ncclGroupStart());
    for (int r = 0; r < R; ++r) NK(ncclAllReduce(buf[r], buf[r], n, ncclBfloat16, ncclSum, comms[r], st[r]));
    NK(ncclGroupEnd());
  }
  for (int r = 0; r < R; ++r) { CK(cudaSetDevice(gpus[r])); CK(cudaStreamSynchronize(st[r])); }
  std::vector<cudaGraph_t> graph(R);
  std::vector<cudaGraphExec_t> exec(R);
  for (int r = 0; r < R; ++r) { CK(cudaSetDevice(gpus[r])); CK(cudaStreamBeginCapture(st[r], cudaStreamCaptureModeRelaxed)); }
  for (int op = 0; op < K; ++op) {
    for (int r = 0; r < R; ++r) {
      CK(cudaSetDevice(gpus[r]));
      if (gap_us > 0) spin<<<1, 1, 0, st[r]>>>((unsigned long long)(gap_us * 1000));
      stamp<<<1, 1, 0, st[r]>>>(ts[r], cnt[r]);
    }
    NK(ncclGroupStart());
    for (int r = 0; r < R; ++r) NK(ncclAllReduce(buf[r], buf[r], n, ncclBfloat16, ncclSum, comms[r], st[r]));
    NK(ncclGroupEnd());
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(gpus[r])); stamp<<<1, 1, 0, st[r]>>>(ts[r], cnt[r]); }
  }
  for (int r = 0; r < R; ++r) {
    CK(cudaSetDevice(gpus[r]));
    CK(cudaStreamEndCapture(st[r], &graph[r]));
    CK(cudaGraphInstantiate(&exec[r], graph[r], 0));
  }
  for (int rep = 0; rep < reps; ++rep)
    for (int r = 0; r < R; ++r) { CK(cudaSetDevice(gpus[r])); CK(cudaGraphLaunch(exec[r], st[r])); }
  for (int r = 0; r < R; ++r) { CK(cudaSetDevice(gpus[r])); CK(cudaStreamSynchronize(st[r])); }
  std::vector<std::vector<unsigned long long>> h(R, std::vector<unsigned long long>(nst));
  for (int r = 0; r < R; ++r) { CK(cudaSetDevice(gpus[r])); CK(cudaMemcpy(h[r].data(), ts[r], nst * 8, cudaMemcpyDeviceToHost)); }
  const long ops = (long)K * reps;
  printf("# nccl %d gpus", ver);
  for (int v : gpus) printf(" %d", v);
  printf(" %d KiB ops %ld gap %.1f us\n", kib, ops, gap_us);
  std::vector<double> all;
  for (int r = 0; r < R; ++r) {
    std::vector<double> d(ops);
    int n100 = 0, n1m = 0;
    for (long o = 0; o < ops; ++o) {
      d[o] = (h[r][2 * o + 1] - h[r][2 * o]) / 1000.0;
      if (o >= K && d[o] > 100) ++n100;
      if (o >= K && d[o] > 1000) ++n1m;
    }
    std::vector<double> dd(d.begin() + K, d.end());
    double mean = 0;
    for (double x : dd) mean += x;
    mean /= dd.size();
    printf("   gpu %2d op p50 %7.2f p99 %7.2f p99.9 %8.1f max %8.1f mean %7.2f us | >100us %d >1ms %d\n", gpus[r],
           pct(dd, .5), pct(dd, .99), pct(dd, .999), pct(dd, 1), mean, n100, n1m);
  }
  if (!dump.empty()) {
    FILE* f = fopen(dump.c_str(), "wb");
    int hdr[3] = {R, K, reps};
    fwrite(hdr, 4, 3, f);
    fwrite(gpus.data(), 4, R, f);
    for (int r = 0; r < R; ++r) fwrite(h[r].data(), 8, nst, f);
    fclose(f);
  }
  fflush(stdout);
  _exit(0);  // skip communicator teardown (single-thread multi-GPU teardown can block)
}
