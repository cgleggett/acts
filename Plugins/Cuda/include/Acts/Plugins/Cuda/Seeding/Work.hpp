#ifndef CUDA_WORK_H
#define CUDA_WORK_H 1


#include <cuda_runtime.h>
#include <string>

struct Work {
  Work(){};
  Work(int cid, int it, int nthr, bool dc, bool dg, bool ag, bool q,
       int sk, int ng, int nt, int na, cudaDeviceProp p, const std::string& f):
    cudaDeviceID(cid), iThread(it), nThreads(nthr), do_cpu(dc), do_gpu(dg), allgroup(ag), quiet(q),
    skip(sk), nGroupToIterate(ng), nTrplPerSpBLimit(nt), nAvgTrplPerSpBLimit(na),
    cudaProp(p), file(f) {
  };
  
  int cudaDeviceID{0};
  int iThread {0};
  int nThreads {0};

  bool do_cpu{true};
  bool do_gpu{true};
  bool allgroup{false};
  bool quiet{true};

  int skip{0};
  int nGroupToIterate{500};
  int nTrplPerSpBLimit{100};
  int nAvgTrplPerSpBLimit{2};

  cudaDeviceProp cudaProp{};
  cudaStream_t stream{};

  bool doCudaTiming{false};
  
  float timeDoubletCuda_ms{0.0};
  float timeTransformCuda_ms{0.0};
  float timeTripletCuda_ms{0.0};
  
  std::string file{"sp.txt"};
};

#endif
