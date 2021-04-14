// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Cuda/Utilities/CudaScalar.cu"
#include "Acts/Plugins/Cuda/Utilities/CUDACore/allocate_host.h"


namespace Acts {

template <typename var_t>
class CudaScalar;

template <typename var_t>
class CpuScalar {
 public:
  CpuScalar(CudaScalar<var_t>* cuScalar, cudaStream_t* s):m_stream(s) {
    m_hostPtr = (var_t*) cms::cuda::allocate_host(sizeof(var_t), *m_stream);
    ACTS_CUDA_ERROR_CHECK(cudaMemcpyAsync(m_hostPtr, cuScalar->get(), sizeof(var_t),
                                          cudaMemcpyDeviceToHost,*m_stream));
  }

  CpuScalar(var_t* cuScalar, cudaStream_t* s):m_stream(s) {
    m_hostPtr = (var_t*) cms::cuda::allocate_host(sizeof(var_t), *m_stream);
    ACTS_CUDA_ERROR_CHECK(cudaMemcpyAsync(m_hostPtr, cuScalar, sizeof(var_t),
                                          cudaMemcpyDeviceToHost, *m_stream));
  }

  ~CpuScalar() {
    cms::cuda::free_host(m_hostPtr);
  }

  var_t* get() { return m_hostPtr; }

  void Set(var_t val) { m_hostPtr[0] = val; }

 private:
  var_t* m_hostPtr {nullptr};
  size_t m_size;
  cudaStream_t* m_stream{nullptr};
};

}  // namespace Acts
