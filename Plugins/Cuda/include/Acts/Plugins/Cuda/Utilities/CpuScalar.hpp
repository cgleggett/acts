// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Plugins/Cuda/Utilities/CudaScalar.cu"

namespace Acts {

template <typename var_t>
class CudaScalar;

template <typename var_t>
class CpuScalar {
 public:
  CpuScalar(bool pin=false):m_pinned(pin) {
    if (!m_pinned) {
      m_hostPtr = new var_t[1];
    } else {
      ACTS_CUDA_ERROR_CHECK(cudaMallocHost(&m_hostPtr, sizeof(var_t)));
    }
  }

  CpuScalar(CudaScalar<var_t>* cuScalar, bool pin=false): m_pinned(pin) {
    if (!m_pinned) {
      m_hostPtr = new var_t[1];
    } else {
      cudaMallocHost(&m_hostPtr, sizeof(var_t));
    }
    ACTS_CUDA_ERROR_CHECK(cudaMemcpy(m_hostPtr, cuScalar->get(), sizeof(var_t),
                                     cudaMemcpyDeviceToHost));
  }

  CpuScalar(CudaScalar<var_t>* cuScalar, cudaStream_t* s, bool pin=false):m_pinned(pin),m_stream(s) {
    if (! m_pinned) {
      m_hostPtr = new var_t[1];
    } else {
      cudaMallocHost(&m_hostPtr, sizeof(var_t));
    }
    ACTS_CUDA_ERROR_CHECK(cudaMemcpyAsync(m_hostPtr, cuScalar->get(), sizeof(var_t),
                                          cudaMemcpyDeviceToHost, *m_stream));

  }
  CpuScalar(var_t* cuScalar, cudaStream_t* s, bool p=false):m_pinned(p),m_stream(s) {
    if (m_pinned) {
      cudaMallocHost(&m_hostPtr, sizeof(var_t));
    } else {
      m_hostPtr = new var_t[1];
    }
    ACTS_CUDA_ERROR_CHECK(cudaMemcpyAsync(m_hostPtr, cuScalar, sizeof(var_t),
                                          cudaMemcpyDeviceToHost, *m_stream));
  }


  ~CpuScalar() {
    if (!m_pinned) {
      delete m_hostPtr;
    } else if (m_pinned && m_hostPtr) {
      cudaFreeHost(m_hostPtr);
    }
  }

  var_t* get() { return m_hostPtr; }

  void Set(var_t val) { m_hostPtr[0] = val; }

 private:
  var_t* m_hostPtr = nullptr;
  size_t m_size;
  bool m_pinned {false};
  cudaStream_t* m_stream{nullptr};
};

}  // namespace Acts
