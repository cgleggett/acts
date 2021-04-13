// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Cuda/Utilities/CpuMatrix.hpp"
#include "Acts/Plugins/Cuda/Utilities/CUDACore/allocate_device.h"


#include <iostream>
#include <memory>

#include "CudaUtils.cu"
#include "cuda.h"
#include "cuda_runtime.h"

namespace Acts {

template <typename var_t>
class CpuMatrix;

template <typename var_t>
class CudaMatrix {
 public:
  CudaMatrix() = delete;
  CudaMatrix(size_t nRows, size_t nCols) {
    m_setSize(nRows, nCols);
    ACTS_CUDA_ERROR_CHECK(
        cudaMalloc((var_t**)&m_devPtr, m_size * sizeof(var_t)));
    //    zeros();
  }
  CudaMatrix(size_t nRows, size_t nCols, cudaStream_t* s, int dev=0):m_stream(s),m_devID(dev) {
    m_setSize(nRows, nCols);
    //    ACTS_CUDA_ERROR_CHECK(cudaMallocAsync((var_t**)&m_devPtr, m_size * sizeof(var_t), *m_stream));
    m_devPtr = (var_t*) cms::cuda::allocate_device(m_devID, m_size*sizeof(var_t), *m_stream);

    zeros();
  }

  CudaMatrix(size_t nRows, size_t nCols, var_t* mat) {
    m_setSize(nRows, nCols);
    ACTS_CUDA_ERROR_CHECK(
        cudaMalloc((var_t**)&m_devPtr, m_size * sizeof(var_t)));
    copyH2D(mat, m_size, 0);
  }
  CudaMatrix(size_t nRows, size_t nCols, var_t* mat, cudaStream_t* s, int dev=0):m_stream(s),m_devID(dev) {
    m_setSize(nRows, nCols);
    //    ACTS_CUDA_ERROR_CHECK(
    //                          cudaMallocAsync((var_t**)&m_devPtr, m_size * sizeof(var_t), *m_stream));
    m_devPtr = (var_t*) cms::cuda::allocate_device(m_devID, m_size*sizeof(var_t), *m_stream);
    copyH2D(mat, m_size, 0);
  }

  CudaMatrix(size_t nRows, size_t nCols, CpuMatrix<var_t>* mat) {
    m_setSize(nRows, nCols);
    ACTS_CUDA_ERROR_CHECK(
        cudaMalloc((var_t**)&m_devPtr, m_size * sizeof(var_t)));
    copyH2D(mat->get(0, 0), m_size, 0);
  }
  CudaMatrix(size_t nRows, size_t nCols, CpuMatrix<var_t>* mat, cudaStream_t* s, int dev=0):m_stream(s),m_devID(dev) {
    m_setSize(nRows, nCols);
    //    ACTS_CUDA_ERROR_CHECK(
    //                          cudaMallocAsync((var_t**)&m_devPtr, m_size * sizeof(var_t), *m_stream));
    m_devPtr = (var_t*) cms::cuda::allocate_device(m_devID, m_size*sizeof(var_t), *m_stream);

    copyH2D(mat->get(0, 0), m_size, 0);
  }

  CudaMatrix(size_t nRows, size_t nCols, var_t* mat, size_t len,
             size_t offset) {
    m_setSize(nRows, nCols);
    ACTS_CUDA_ERROR_CHECK(
        cudaMalloc((var_t**)&m_devPtr, m_size * sizeof(var_t)));
    copyH2D(mat, len, offset);
  }
  CudaMatrix(size_t nRows, size_t nCols, var_t* mat, size_t len,
             size_t offset, cudaStream_t* s, int dev=0):m_stream(s),m_devID(dev) {
    m_setSize(nRows, nCols);
    // ACTS_CUDA_ERROR_CHECK(
    //                       cudaMallocAsync((var_t**)&m_devPtr, m_size * sizeof(var_t),*m_stream));
    m_devPtr = (var_t*) cms::cuda::allocate_device(m_devID, m_size*sizeof(var_t), *m_stream);

    copyH2D(mat, len, offset);
  }

  CudaMatrix(size_t nRows, size_t nCols, CpuMatrix<var_t>* mat, size_t len,
             size_t offset) {
    m_setSize(nRows, nCols);
    ACTS_CUDA_ERROR_CHECK(
        cudaMalloc((var_t**)&m_devPtr, m_size * sizeof(var_t)));
    copyH2D(mat->get(0, 0), len, offset);
  }
  CudaMatrix(size_t nRows, size_t nCols, CpuMatrix<var_t>* mat, size_t len,
             size_t offset, cudaStream_t* s, int dev=0):m_stream(s),m_devID(dev) {
    m_setSize(nRows, nCols);
    // ACTS_CUDA_ERROR_CHECK(
    //                       cudaMallocAsync((var_t**)&m_devPtr, m_size * sizeof(var_t),*m_stream));
    m_devPtr = (var_t*) cms::cuda::allocate_device(m_devID, m_size*sizeof(var_t), *m_stream);

    copyH2D(mat->get(0, 0), len, offset);
  }

  ~CudaMatrix() {
    if (m_devPtr) {
      if (m_stream) {
        //        cudaFreeAsync(m_devPtr,*m_stream);
        cms::cuda::free_device(m_devID, m_devPtr, *m_stream);
      } else {
        cudaFree(m_devPtr);
      }
    }
  }

  var_t* get(size_t row = 0, size_t col = 0) {
    int offset = row + col * m_nRows;
    return m_devPtr + offset;
  }

  void copyH2D(var_t* matrix, size_t len, size_t offset = 0) {
    if (m_stream) {
      ACTS_CUDA_ERROR_CHECK(cudaMemcpyAsync(m_devPtr + offset, matrix,
                                            len * sizeof(var_t),
                                            cudaMemcpyHostToDevice,
                                            *m_stream));
    } else {
      ACTS_CUDA_ERROR_CHECK(cudaMemcpy(m_devPtr + offset, matrix,
                                       len * sizeof(var_t),
                                       cudaMemcpyHostToDevice));
    }
  }

  void copyH2D(const var_t* matrix, size_t len, size_t offset = 0) {
    if (m_stream) {
      ACTS_CUDA_ERROR_CHECK(cudaMemcpyAsync(m_devPtr + offset, matrix,
                                            len * sizeof(var_t),
                                            cudaMemcpyHostToDevice,
                                            *m_stream));
    } else {
      ACTS_CUDA_ERROR_CHECK(cudaMemcpy(m_devPtr + offset, matrix,
                                       len * sizeof(var_t),
                                       cudaMemcpyHostToDevice));
    }
  }

  void zeros() {
    if (m_stream) {
      ACTS_CUDA_ERROR_CHECK(cudaMemsetAsync(m_devPtr, 0, m_size * sizeof(var_t),*m_stream));
    } else {
      ACTS_CUDA_ERROR_CHECK(cudaMemset(m_devPtr, 0, m_size * sizeof(var_t)));
    }
  }

 private:
  var_t* m_devPtr{nullptr};
  size_t m_nCols;
  size_t m_nRows;
  size_t m_size;
  cudaStream_t* m_stream{nullptr};
  int m_devID {0};

  void m_setSize(size_t row, size_t col) {
    m_nRows = row;
    m_nCols = col;
    m_size = m_nRows * m_nCols;
  }
};

}  // namespace Acts
