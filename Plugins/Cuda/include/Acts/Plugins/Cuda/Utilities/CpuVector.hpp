// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Cuda/Utilities/CudaVector.cu"
#include "Acts/Plugins/Cuda/Utilities/CUDACore/allocate_host.h"

#include <cstring>

namespace Acts {

template <typename var_t>
class CudaVector;

template <typename var_t>
class CpuVector {
 public:
  CpuVector() = delete;
  CpuVector(size_t size, cudaStream_t*s):m_stream(s) {
    m_size = size;
    // m_pinned = pinned;
    // if (pinned == 0) {
    //   m_hostPtr = new var_t[m_size];
    // } else if (pinned == 1) {
    //   cudaMallocHost(&m_hostPtr, m_size * sizeof(var_t));
    // }
    m_hostPtr = (var_t*) cms::cuda::allocate_host(m_size*sizeof(var_t), *m_stream);
  }

  CpuVector(size_t size, CudaVector<var_t>* cuVec, cudaStream_t* s):m_stream(s) {
    m_size = size;
    // m_pinned = pinned;
    // if (pinned == 0) {
    //   m_hostPtr = new var_t[m_size];
    // } else if (pinned == 1) {
    //   cudaMallocHost(&m_hostPtr, m_size * sizeof(var_t));
    // }
    m_hostPtr = (var_t*) cms::cuda::allocate_host(m_size*sizeof(var_t), *m_stream);
    cudaMemcpyAsync(m_hostPtr, cuVec->get(), m_size * sizeof(var_t),
                    cudaMemcpyDeviceToHost, *m_stream);
  }

  ~CpuVector() {
    // if (!m_pinned) {
    //   delete m_hostPtr;
    // } else if (m_pinned && m_hostPtr) {
    //   cudaFreeHost(m_hostPtr);
    // }
    cms::cuda::free_host(m_hostPtr);    
  }

  var_t* get(size_t offset = 0) { return m_hostPtr + offset; }

  void set(size_t offset, var_t val) { m_hostPtr[offset] = val; }

  void copyD2H(var_t* devPtr, size_t len, size_t offset) {
    cudaMemcpyAsync(m_hostPtr + offset, devPtr, len * sizeof(var_t),
                    cudaMemcpyDeviceToHost, *m_stream);
  }

  void copyD2H(var_t* devPtr, size_t len, size_t offset, cudaStream_t* stream) {
    cudaMemcpyAsync(m_hostPtr + offset, devPtr, len * sizeof(var_t),
                    cudaMemcpyDeviceToHost, *stream);
  }

  void zeros() { memset(m_hostPtr, 0, m_size * sizeof(var_t)); }

 private:
  cudaStream_t* m_stream {nullptr};
  var_t* m_hostPtr = nullptr;
  size_t m_size;
  bool m_pinned;
};

}  // namespace Acts
