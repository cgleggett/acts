// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include "Acts/Plugins/Cuda/Seeding/Work.hpp"
#include "Acts/Plugins/Cuda/Seeding/Structs.hpp"


namespace Acts {

template <typename external_spacepoint_t>
Seedfinder<external_spacepoint_t, Acts::Cuda>::Seedfinder(
    Acts::SeedfinderConfig<external_spacepoint_t> config)
    : m_config(std::move(config)) {
  // calculation of scattering using the highland formula
  // convert pT to p once theta angle is known
  m_config.highland = 13.6 * std::sqrt(m_config.radLengthPerSeed) *
                      (1 + 0.038 * std::log(m_config.radLengthPerSeed));
  float maxScatteringAngle = m_config.highland / m_config.minPt;
  m_config.maxScatteringAngle2 = maxScatteringAngle * maxScatteringAngle;

  // helix radius in homogeneous magnetic field. Units are Kilotesla, MeV and
  // millimeter
  // TODO: change using ACTS units
  m_config.pTPerHelixRadius = 300. * m_config.bFieldInZ;
  m_config.minHelixDiameter2 =
      std::pow(m_config.minPt * 2 / m_config.pTPerHelixRadius, 2);
  m_config.pT2perRadius =
      std::pow(m_config.highland / m_config.pTPerHelixRadius, 2);
}

// CUDA seed finding
template <typename external_spacepoint_t>
template <typename sp_range_t>
std::vector<Seed<external_spacepoint_t>>
Seedfinder<external_spacepoint_t, Acts::Cuda>::createSeedsForGroup(
    sp_range_t bottomSPs, sp_range_t middleSPs, sp_range_t topSPs,
    Work& w, GPUStructs::Config* scd) const {
  std::vector<Seed<external_spacepoint_t>> outputVec;

  // Get SeedfinderConfig values
  auto compatSeedLimit_cpu = m_config.seedFilter->getSeedFilterConfig().compatSeedLimit;

  //---------------------------------
  // Algorithm 0. Matrix Flattening
  //---------------------------------

  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      middleSPvec;
  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*>
      bottomSPvec;
  std::vector<const Acts::InternalSpacePoint<external_spacepoint_t>*> topSPvec;

  // Get the size of spacepoints
  int nSpM(0);
  int nSpB(0);
  int nSpT(0);

  for (auto sp : middleSPs) {
    nSpM++;
    middleSPvec.push_back(sp);
  }
  for (auto sp : bottomSPs) {
    nSpB++;
    bottomSPvec.push_back(sp);
  }
  for (auto sp : topSPs) {
    nSpT++;
    topSPvec.push_back(sp);
  }

  CudaScalar<int> nSpM_cuda(&nSpM, &w.stream);
  CudaScalar<int> nSpB_cuda(&nSpB, &w.stream);
  CudaScalar<int> nSpT_cuda(&nSpT, &w.stream);

  if (nSpM == 0 || nSpB == 0 || nSpT == 0)
    return outputVec;

  // Matrix flattening
  CpuMatrix<float> spMmat_cpu(nSpM, 6, &w.stream);  // x y z r varR varZ
  CpuMatrix<float> spBmat_cpu(nSpB, 6, &w.stream);
  CpuMatrix<float> spTmat_cpu(nSpT, 6, &w.stream);

  auto fillMatrix = [](auto& mat, auto& id, auto sp) {
    mat.set(id, 0, sp->x());
    mat.set(id, 1, sp->y());
    mat.set(id, 2, sp->z());
    mat.set(id, 3, sp->radius());
    mat.set(id, 4, sp->varianceR());
    mat.set(id, 5, sp->varianceZ());
    id++;
  };

  int mIdx(0);
  for (auto sp : middleSPs) {
    fillMatrix(spMmat_cpu, mIdx, sp);
  }
  int bIdx(0);
  for (auto sp : bottomSPs) {
    fillMatrix(spBmat_cpu, bIdx, sp);
  }
  int tIdx(0);
  for (auto sp : topSPs) {
    fillMatrix(spTmat_cpu, tIdx, sp);
  }

  CudaMatrix<float> spMmat_cuda(nSpM, 6, &spMmat_cpu, &w.stream);
  CudaMatrix<float> spBmat_cuda(nSpB, 6, &spBmat_cpu, &w.stream);
  CudaMatrix<float> spTmat_cuda(nSpT, 6, &spTmat_cpu, &w.stream);
  //------------------------------------
  //  Algorithm 1. Doublet Search (DS)
  //------------------------------------

  CudaScalar<int> nSpMcomp_cuda(new int(0), &w.stream);
  CudaScalar<int> nSpBcompPerSpMMax_cuda(new int(0), &w.stream);
  CudaScalar<int> nSpTcompPerSpMMax_cuda(new int(0), &w.stream);
  CudaVector<int> nSpBcompPerSpM_cuda(nSpM, &w.stream);
  nSpBcompPerSpM_cuda.zeros();
  CudaVector<int> nSpTcompPerSpM_cuda(nSpM, &w.stream);
  nSpTcompPerSpM_cuda.zeros();
  CudaVector<int> McompIndex_cuda(nSpM, &w.stream);
  CudaMatrix<int> BcompIndex_cuda(nSpB, nSpM, &w.stream);
  CudaMatrix<int> TcompIndex_cuda(nSpT, nSpM, &w.stream);
  CudaMatrix<int> tmpBcompIndex_cuda(nSpB, nSpM, &w.stream);
  CudaMatrix<int> tmpTcompIndex_cuda(nSpT, nSpM, &w.stream);

  dim3 DS_BlockSize = m_config.maxBlockSize;
  dim3 DS_GridSize(nSpM, 1, 1);

  searchDoublet(DS_GridSize, DS_BlockSize, nSpM_cuda.get(), spMmat_cuda.get(),
                nSpB_cuda.get(), spBmat_cuda.get(), nSpT_cuda.get(),
                spTmat_cuda.get(), &scd->deltaRMin, &scd->deltaRMax,
                &scd->cotThetaMax, &scd->collisionRegionMin,
                &scd->collisionRegionMax, nSpMcomp_cuda.get(),
                nSpBcompPerSpMMax_cuda.get(), nSpTcompPerSpMMax_cuda.get(),
                nSpBcompPerSpM_cuda.get(), nSpTcompPerSpM_cuda.get(),
                McompIndex_cuda.get(), BcompIndex_cuda.get(),
                tmpBcompIndex_cuda.get(), TcompIndex_cuda.get(),
                tmpTcompIndex_cuda.get(), w);

  CpuScalar<int> nSpMcomp_cpu(&nSpMcomp_cuda, &w.stream);
  CpuScalar<int> nSpBcompPerSpMMax_cpu(&nSpBcompPerSpMMax_cuda, &w.stream);
  CpuScalar<int> nSpTcompPerSpMMax_cpu(&nSpTcompPerSpMMax_cuda, &w.stream);
  CpuVector<int> nSpBcompPerSpM_cpu(nSpM, &nSpBcompPerSpM_cuda, &w.stream);
  CpuVector<int> nSpTcompPerSpM_cpu(nSpM, &nSpTcompPerSpM_cuda, &w.stream);
  CpuVector<int> McompIndex_cpu(nSpM, &McompIndex_cuda);

  //--------------------------------------
  //  Algorithm 2. Transform coordinate
  //--------------------------------------

  CudaMatrix<float> spMcompMat_cuda(*nSpMcomp_cpu.get(), 6, &w.stream);
  CudaMatrix<float> spBcompMatPerSpM_cuda(*nSpBcompPerSpMMax_cpu.get(),
                                          (*nSpMcomp_cpu.get()) * 6, &w.stream);
  CudaMatrix<float> spTcompMatPerSpM_cuda(*nSpTcompPerSpMMax_cpu.get(),
                                          (*nSpMcomp_cpu.get()) * 6, &w.stream);
  CudaMatrix<float> circBcompMatPerSpM_cuda(*nSpBcompPerSpMMax_cpu.get(),
                                            (*nSpMcomp_cpu.get()) * 6, &w.stream);
  CudaMatrix<float> circTcompMatPerSpM_cuda(*nSpTcompPerSpMMax_cpu.get(),
                                            (*nSpMcomp_cpu.get()) * 6, &w.stream);

  dim3 TC_GridSize(*nSpMcomp_cpu.get(), 1, 1);
  dim3 TC_BlockSize = m_config.maxBlockSize;

  transformCoordinate(
      TC_GridSize, TC_BlockSize, nSpM_cuda.get(), spMmat_cuda.get(),
      McompIndex_cuda.get(), nSpB_cuda.get(), spBmat_cuda.get(),
      nSpBcompPerSpMMax_cuda.get(), BcompIndex_cuda.get(), nSpT_cuda.get(),
      spTmat_cuda.get(), nSpTcompPerSpMMax_cuda.get(), TcompIndex_cuda.get(),
      spMcompMat_cuda.get(), spBcompMatPerSpM_cuda.get(),
      circBcompMatPerSpM_cuda.get(), spTcompMatPerSpM_cuda.get(),
      circTcompMatPerSpM_cuda.get(), w);

  //------------------------------------------------------
  //  Algorithm 3. Triplet Search (TS) & Seed filtering
  //------------------------------------------------------

  const int nTrplPerSpMLimit =
      m_config.nAvgTrplPerSpBLimit * (*nSpBcompPerSpMMax_cpu.get());
  CudaScalar<int> nTrplPerSpMLimit_cuda(&nTrplPerSpMLimit, &w.stream);

  CudaScalar<int> nTrplPerSpBLimit_cuda(&m_config.nTrplPerSpBLimit, &w.stream);
  CpuScalar<int> nTrplPerSpBLimit_cpu(
      &nTrplPerSpBLimit_cuda, &w.stream);  // need to be USM

  CudaVector<int> nTrplPerSpM_cuda(*nSpMcomp_cpu.get(), &w.stream);
  nTrplPerSpM_cuda.zeros();
  CudaMatrix<Triplet> TripletsPerSpM_cuda(nTrplPerSpMLimit,
                                          *nSpMcomp_cpu.get(), &w.stream);
  CpuVector<int> nTrplPerSpM_cpu(*nSpMcomp_cpu.get(), &w.stream, true);
  nTrplPerSpM_cpu.zeros();
  CpuMatrix<Triplet> TripletsPerSpM_cpu(nTrplPerSpMLimit, *nSpMcomp_cpu.get(),
                                        &w.stream, true);
  for (int i_m = 0; i_m <= *nSpMcomp_cpu.get(); i_m++) {
    cudaStreamSynchronize(w.stream);

    // Search Triplet
    if (i_m < *nSpMcomp_cpu.get()) {
      int mIndex = *McompIndex_cpu.get(i_m);
      int* nSpBcompPerSpM = nSpBcompPerSpM_cpu.get(mIndex);
      int* nSpTcompPerSpM = nSpTcompPerSpM_cpu.get(mIndex);

      dim3 TS_GridSize(*nSpBcompPerSpM, 1, 1);
      dim3 TS_BlockSize =
          dim3(fmin(m_config.maxBlockSize, *nSpTcompPerSpM), 1, 1);

      searchTriplet(
          TS_GridSize, TS_BlockSize, nSpTcompPerSpM_cpu.get(mIndex),
          nSpTcompPerSpM_cuda.get(mIndex), nSpMcomp_cuda.get(),
          spMcompMat_cuda.get(i_m, 0), nSpBcompPerSpMMax_cuda.get(),
          BcompIndex_cuda.get(0, i_m), circBcompMatPerSpM_cuda.get(0, 6 * i_m),
          nSpTcompPerSpMMax_cuda.get(), TcompIndex_cuda.get(0, i_m),
          spTcompMatPerSpM_cuda.get(0, 6 * i_m),
          circTcompMatPerSpM_cuda.get(0, 6 * i_m),
          // Seed finder config
          &scd->maxScatteringAngle2, &scd->sigmaScattering,
          &scd->minHelixDiameter2, &scd->pT2perRadius,
          &scd->impactMax, nTrplPerSpMLimit_cuda.get(),
          nTrplPerSpBLimit_cpu.get(), nTrplPerSpBLimit_cuda.get(),
          &scd->deltaInvHelixDiameter, &scd->impactWeightFactor,
          &scd->filterDeltaRMin, &scd->compatSeedWeight,
          &compatSeedLimit_cpu, &scd->compatSeedLimit,
          // output
          nTrplPerSpM_cuda.get(i_m), TripletsPerSpM_cuda.get(0, i_m),
          w);
      nTrplPerSpM_cpu.copyD2H(nTrplPerSpM_cuda.get(i_m), 1, i_m, &w.stream);

      TripletsPerSpM_cpu.copyD2H(TripletsPerSpM_cuda.get(0, i_m),
                                 nTrplPerSpMLimit, nTrplPerSpMLimit * i_m,
                                 &w.stream);
    }

    if (i_m > 0) {
      const auto m_experimentCuts = m_config.seedFilter->getExperimentCuts();
      std::vector<std::pair<
          float, std::unique_ptr<const InternalSeed<external_spacepoint_t>>>>
          seedsPerSpM;

      for (int i = 0; i < *nTrplPerSpM_cpu.get(i_m - 1); i++) {
        auto& triplet = *TripletsPerSpM_cpu.get(i, i_m - 1);
        int mIndex = *McompIndex_cpu.get(i_m - 1);
        int bIndex = triplet.bIndex;
        int tIndex = triplet.tIndex;

        auto& bottomSP = *bottomSPvec[bIndex];
        auto& middleSP = *middleSPvec[mIndex];
        auto& topSP = *topSPvec[tIndex];
        if (m_experimentCuts != nullptr) {
          // add detector specific considerations on the seed weight
          triplet.weight +=
              m_experimentCuts->seedWeight(bottomSP, middleSP, topSP);
          // discard seeds according to detector specific cuts (e.g.: weight)
          if (!m_experimentCuts->singleSeedCut(triplet.weight, bottomSP,
                                               middleSP, topSP)) {
            continue;
          }
        }

        float Zob = 0;  // It is not used in the seed filter but needs to be
                        // fixed anyway...

        seedsPerSpM.push_back(std::make_pair(
            triplet.weight,
            std::make_unique<const InternalSeed<external_spacepoint_t>>(
                bottomSP, middleSP, topSP, Zob)));
      }

      m_config.seedFilter->filterSeeds_1SpFixed(seedsPerSpM, outputVec);
    }
  }
  return outputVec;
}
}  // namespace Acts
