// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Cuda/Utilities/CpuScalar.hpp"

#include <iostream>
#include <memory>

#include "CudaUtils.cu"
#include "cuda.h"
#include "cuda_runtime.h"
#include <cxxabi.h>
#include <thread>


namespace Acts {

template <typename var_t>
class CpuScalar;

template <typename var_t>
class CudaScalar {
 public:
  CudaScalar() {
    ACTS_CUDA_ERROR_CHECK(cudaMalloc((var_t**)&m_devPtr, sizeof(var_t)));
  }

  CudaScalar(cudaStream_t *s, int dev=0): m_stream(s),m_devID(dev) {
    ACTS_CUDA_ERROR_CHECK(cudaMallocAsync((var_t**)&m_devPtr, sizeof(var_t), *m_stream));
  }

  
  CudaScalar(var_t* scalar) {
    ACTS_CUDA_ERROR_CHECK(cudaMalloc((var_t**)&m_devPtr, sizeof(var_t)));
    ACTS_CUDA_ERROR_CHECK(
        cudaMemcpy(m_devPtr, scalar, sizeof(var_t), cudaMemcpyHostToDevice));
  }

  CudaScalar(var_t* scalar, cudaStream_t *s, int dev=0):m_stream(s),m_devID(dev) {
    ACTS_CUDA_ERROR_CHECK(cudaMallocAsync((var_t**)&m_devPtr, sizeof(var_t),*m_stream));
    ACTS_CUDA_ERROR_CHECK(
                          cudaMemcpyAsync(m_devPtr, scalar, sizeof(var_t),
                                          cudaMemcpyHostToDevice,*m_stream));
  }

  CudaScalar(const var_t* scalar) {
    ACTS_CUDA_ERROR_CHECK(cudaMalloc((var_t**)&m_devPtr, sizeof(var_t)));
    if (m_devPtr == nullptr) {
      throw std::bad_alloc();
    }
    ACTS_CUDA_ERROR_CHECK(
        cudaMemcpy(m_devPtr, scalar, sizeof(var_t), cudaMemcpyHostToDevice));
  }

  CudaScalar(const var_t* scalar, cudaStream_t* s, int dev=0):m_stream(s),m_devID(dev) {
    ACTS_CUDA_ERROR_CHECK(cudaMallocAsync((var_t**)&m_devPtr, sizeof(var_t),*m_stream));
    if (m_devPtr == nullptr) {
      throw std::bad_alloc();
    }
    ACTS_CUDA_ERROR_CHECK(
                          cudaMemcpyAsync(m_devPtr, scalar, sizeof(var_t),
                                          cudaMemcpyHostToDevice, *m_stream));
  }

  ~CudaScalar() {
    if (m_stream) {
      ACTS_CUDA_ERROR_CHECK(cudaFreeAsync(m_devPtr,*m_stream));
    } else {
      ACTS_CUDA_ERROR_CHECK(cudaFree(m_devPtr));
    }
  }

  var_t* get() { return m_devPtr; }

  void zeros() { cudaMemset(m_devPtr, 0, sizeof(var_t)); }

 private:
  var_t* m_devPtr{nullptr};
  cudaStream_t* m_stream{nullptr};
  int m_devID {0};
};
}  // namespace Acts
