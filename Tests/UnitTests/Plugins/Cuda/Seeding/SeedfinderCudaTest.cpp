// This file is part of the Acts project.
//
// Copyright (C) 2020 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "Acts/Plugins/Cuda/Seeding/Seedfinder.hpp"
#include "Acts/Plugins/Cuda/Seeding/Work.hpp"
#include "Acts/Seeding/BinFinder.hpp"
#include "Acts/Seeding/BinnedSPGroup.hpp"
#include "Acts/Seeding/InternalSeed.hpp"
#include "Acts/Seeding/InternalSpacePoint.hpp"
#include "Acts/Seeding/Seed.hpp"
#include "Acts/Seeding/SeedFilter.hpp"
#include "Acts/Seeding/Seedfinder.hpp"
#include "Acts/Seeding/SpacePointGrid.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>
#include <thread>
#include <mutex>
#include <queue>
#include <filesystem>

#include <boost/type_erasure/any_cast.hpp>
#include <cuda_profiler_api.h>

#include "ATLASCuts.hpp"
#include "SpacePoint.hpp"
  

int proc(Work &w);

class sdf_task {
public:

  sdf_task(Work &w, std::queue<std::string>* q=nullptr, std::mutex* m=nullptr) : m_work(w), m_que(q), m_mut(m) {};
  
  void operator() () const {
    if (m_que != 0) {
      while (true) {
        {
          std::lock_guard<std::mutex> lk(*m_mut);
          if (m_que->size() == 0) {
            return;
          }          
          m_work.file = m_que->front();
          m_que->pop();
        }
        proc(m_work);
      }
    } else {
      proc(m_work);
    }
  }

  Work &m_work;
  std::queue<std::string>* m_que {nullptr};
  std::mutex* m_mut{nullptr};
  
  
  
};
  


std::vector<const SpacePoint*> readFile(std::string filename) {
  std::string line;
  int layer;
  std::vector<const SpacePoint*> readSP;

  std::ifstream spFile(filename);
  if (spFile.is_open()) {
    while (!spFile.eof()) {
      std::getline(spFile, line);
      std::stringstream ss(line);
      std::string linetype;
      ss >> linetype;
      float x, y, z, r, varianceR, varianceZ;
      if (linetype == "lxyz") {
        ss >> layer >> x >> y >> z >> varianceR >> varianceZ;
        r = std::sqrt(x * x + y * y);
        float f22 = varianceR;
        float wid = varianceZ;
        float cov = wid * wid * .08333;
        if (cov < f22)
          cov = f22;
        if (std::abs(z) > 450.) {
          varianceZ = 9. * cov;
          varianceR = .06;
        } else {
          varianceR = 9. * cov;
          varianceZ = .06;
        }
        SpacePoint* sp =
            new SpacePoint{x, y, z, r, layer, varianceR, varianceZ};
        //     if(r < 200.){
        //       sp->setClusterList(1,0);
        //     }
        readSP.push_back(sp);
      }
    }
  }
  return readSP;
}

int main(int argc, char** argv) {

  std::string file{"sp.txt"};
  bool help(false);
  bool quiet(false);
  bool allgroup(false);
  bool do_cpu(false);
  bool do_gpu(false);
  bool do_timing(false);
  std::string fdir{};
  int nThreads = 1;
  int nGroupToIterate = 500;
  int skip = 0;
  int deviceID = 0;
  int nTrplPerSpBLimit = 100;
  int nAvgTrplPerSpBLimit = 2;
  size_t nFiles = 0;

  int opt;
  while ((opt = getopt(argc, argv, "haf:n:s:d:l:m:N:F:qCGTD:")) != -1) {
    switch (opt) {
      case 'a':
        allgroup = true;
        break;
      case 'f':
        file = optarg;
        break;
      case 'D':
        fdir = optarg;
        break;
      case 'n':
        nGroupToIterate = atoi(optarg);
        break;
      case 's':
        skip = atoi(optarg);
        break;
      case 'd':
        deviceID = atoi(optarg);
        break;
      case 'l':
        nAvgTrplPerSpBLimit = atoi(optarg);
        break;
      case 'm':
        nTrplPerSpBLimit = atoi(optarg);
        break;
      case 'N':
        nThreads = atoi(optarg);
        break;
      case 'F':
        nFiles = atoi(optarg);
        break;
      case 'q':
        quiet = true;
        break;
      case 'C':
        do_cpu = true;
        break;
      case 'G':
        do_gpu = true;
        break;
      case 'T':
        do_timing = true;
        break;
      case 'h':
        help = true;
        [[fallthrough]];
      default: /* '?' */
        std::cerr << "Usage: " << argv[0] << " [-hq] [-f FILENAME]\n";
        if (help) {
          std::cout << "      -h : this help" << std::endl;
          std::cout << "      -a ALL   : analyze all groups. Default is \""
                    << allgroup << "\"" << std::endl;
          std::cout
              << "      -f FILE  : read spacepoints from FILE. Default is \""
              << file << "\"" << std::endl;
          std::cout << "      -n NUM   : Number of groups to iterate in seed "
                       "finding. Default is "
                    << nGroupToIterate << std::endl;
          std::cout << "      -s SKIP  : Number of groups to skip in seed "
                       "finding. Default is "
                    << skip << std::endl;
          std::cout << "      -d DEVID : NVIDIA GPU device ID. Default is "
                    << deviceID << std::endl;
          std::cout << "      -l : A limit on the average number of triplets "
                       "per bottom spacepoint: this is used for determining "
                       "matrix size for triplets per middle space point"
                    << nAvgTrplPerSpBLimit << std::endl;
          std::cout << "      -m : A limit on the number of triplets per "
                       "bottom spacepoint: users do not have to touch this for "
                       "# spacepoints < ~200k"
                    << nTrplPerSpBLimit << std::endl;
          std::cout << "      -q : don't print out all found seeds"
                    << std::endl;
          std::cout << "      -G : only run on GPU, not CPU" << std::endl;
        }

        exit(EXIT_FAILURE);
    }
  }

  std::queue<std::string>* fque(nullptr);
  std::mutex* fmut(nullptr);
  if (fdir != "") {
    fque = new std::queue<std::string>;
    fmut = new std::mutex;
    
    for (const auto& ff : std::filesystem::directory_iterator(fdir))  {
      std::string fps = ff.path();
      //      std::cout << ff.path() << std::endl;
      std::ifstream ifs(ff.path());
      if (ifs.is_open()) {
        fque->push(ff.path());
        std::cout << "--> will proc " << fps << std::endl;
      }
      if (fque->size() >= nFiles) {
        break;
      }
    }
  }
  
  std::string devName;
  ACTS_CUDA_ERROR_CHECK(cudaSetDevice(deviceID));

  cudaDeviceProp prop;
  ACTS_CUDA_ERROR_CHECK(cudaGetDeviceProperties(&prop, deviceID));
  printf("\n GPU Device %d: \"%s\" with compute capability %d.%d\n\n", deviceID,
         prop.name, prop.major, prop.minor);

  std::vector<Work> vw;
  for (int i=0; i<nThreads; ++i) {
    vw.push_back( Work(deviceID, i, nThreads, do_cpu, do_gpu, allgroup, quiet, skip,
                       nGroupToIterate, nTrplPerSpBLimit, nAvgTrplPerSpBLimit, prop, file) );
    ACTS_CUDA_ERROR_CHECK(cudaStreamCreate(& vw.back().stream));
    vw.back().doCudaTiming = do_timing;
  }
  
  std::vector<std::thread> tv;
  for (int i=0; i<nThreads; ++i) {
    tv.push_back( std::thread{ sdf_task(vw[i], fque, fmut) } );
  }

  for (auto &v: tv) {
    v.join();
  }

  if (do_timing) {
    std::cout << "kernel timing in ms\n";
    int i=0;
    std::cout << "     doublet  transf   triplet  cudaSF\n";
    float t1{0}, t2{0}, t3{0}, t4{0};
    for (auto &v: vw) {
      ++i;
      ACTS_CUDA_ERROR_CHECK(cudaStreamDestroy( v.stream ));
      printf("%3d:  %6.3f  %6.3f  %6.3f   %6.3f\n",i,v.timeDoubletCuda_ms, v.timeTransformCuda_ms,
             v.timeTripletCuda_ms, v.timeSeedfinder/v.nCalls);
      t1 += v.timeDoubletCuda_ms;
      t2 += v.timeTransformCuda_ms;
      t3 += v.timeTripletCuda_ms;
      t4 += v.timeSeedfinder/v.nCalls;
    }
    printf("avg:  %6.3f  %6.3f  %6.3f   %6.3f\n",t1/nThreads,t2/nThreads,t3/nThreads, t4/nThreads);
  }

  
  return 0;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

int proc(Work &w) {

  const std::string& file = w.file;
  //  int deviceID = w.cudaDeviceID;
  
  bool allgroup = w.allgroup;
  bool do_cpu = w.do_cpu;
  bool do_gpu = w.do_gpu;
  bool quiet = w.quiet;
  int skip = w.skip;
  int nGroupToIterate = w.nGroupToIterate;
  int nTrplPerSpBLimit = w.nTrplPerSpBLimit;
  int nAvgTrplPerSpBLimit = w.nAvgTrplPerSpBLimit;

  ++ w.nCalls;
  
  auto start_pre = std::chrono::system_clock::now();
  
  std::ifstream f(file);
  if (!f.good()) {
    std::cerr << "input file \"" << file << "\" does not exist\n";
    exit(EXIT_FAILURE);
  }

  std::vector<const SpacePoint*> spVec = readFile(file);

  std::cout << "read " << spVec.size() << " SP from file " << file << std::endl;
  //  MSG( "read " << spVec.size() << " SP from file " << file );

  // Set seed finder configuration
  Acts::SeedfinderConfig<SpacePoint> config;
  // silicon detector max
  config.rMax = 160.;
  config.deltaRMin = 5.;
  config.deltaRMax = 160.;
  config.collisionRegionMin = -250.;
  config.collisionRegionMax = 250.;
  config.zMin = -2800.;
  config.zMax = 2800.;
  config.maxSeedsPerSpM = 5;
  // 2.7 eta
  config.cotThetaMax = 7.40627;
  config.sigmaScattering = 1.00000;

  config.minPt = 500.;
  config.bFieldInZ = 0.00199724;

  config.beamPos = {-.5, -.5};
  config.impactMax = 10.;

  // cuda
  cudaDeviceProp prop = w.cudaProp;

  config.maxBlockSize = prop.maxThreadsPerBlock / w.nThreads;
  config.nTrplPerSpBLimit = nTrplPerSpBLimit;
  config.nAvgTrplPerSpBLimit = nAvgTrplPerSpBLimit;

  // binfinder
  auto bottomBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
      Acts::BinFinder<SpacePoint>());
  auto topBinFinder = std::make_shared<Acts::BinFinder<SpacePoint>>(
      Acts::BinFinder<SpacePoint>());
  Acts::SeedFilterConfig sfconf;
  Acts::ATLASCuts<SpacePoint> atlasCuts = Acts::ATLASCuts<SpacePoint>();
  config.seedFilter = std::make_unique<Acts::SeedFilter<SpacePoint>>(
      Acts::SeedFilter<SpacePoint>(sfconf, &atlasCuts));
  Acts::Seedfinder<SpacePoint> seedfinder_cpu(config);
  Acts::Seedfinder<SpacePoint, Acts::Cuda> seedfinder_cuda(config);

  // covariance tool, sets covariances per spacepoint as required
  auto ct = [=](const SpacePoint& sp, float, float, float) -> Acts::Vector2 {
    return {sp.varianceR, sp.varianceZ};
  };

  // setup spacepoint grid config
  Acts::SpacePointGridConfig gridConf;
  gridConf.bFieldInZ = config.bFieldInZ;
  gridConf.minPt = config.minPt;
  gridConf.rMax = config.rMax;
  gridConf.zMax = config.zMax;
  gridConf.zMin = config.zMin;
  gridConf.deltaRMax = config.deltaRMax;
  gridConf.cotThetaMax = config.cotThetaMax;
  // create grid with bin sizes according to the configured geometry
  std::unique_ptr<Acts::SpacePointGrid<SpacePoint>> grid =
      Acts::SpacePointGridCreator::createGrid<SpacePoint>(gridConf);
  auto spGroup = Acts::BinnedSPGroup<SpacePoint>(spVec.begin(), spVec.end(), ct,
                                                 bottomBinFinder, topBinFinder,
                                                 std::move(grid), config);

  auto end_pre = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsec_pre = end_pre - start_pre;
  double preprocessTime = elapsec_pre.count();
  std::cout << "Preprocess Time: " << preprocessTime << std::endl;

  //--------------------------------------------------------------------//
  //                        Begin Seed finding                          //
  //--------------------------------------------------------------------//

  auto start_cpu = std::chrono::system_clock::now();

  int group_count;
  auto groupIt = spGroup.begin();

  //----------- CPU ----------//
  group_count = 0;
  std::vector<std::vector<Acts::Seed<SpacePoint>>> seedVector_cpu;
  groupIt = spGroup.begin();

  if (do_cpu) {
    for (int i_s = 0; i_s < skip; i_s++)
      ++groupIt;
    for (; !(groupIt == spGroup.end()); ++groupIt) {
      seedVector_cpu.push_back(seedfinder_cpu.createSeedsForGroup(
          groupIt.bottom(), groupIt.middle(), groupIt.top()));
      group_count++;
      if (allgroup == false) {
        if (group_count >= nGroupToIterate)
          break;
      }
    }
    // auto timeMetric_cpu = seedfinder_cpu.getTimeMetric();
    std::cout << "Analyzed " << group_count << " groups for CPU" << std::endl;
  }

  auto end_cpu = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsec_cpu = end_cpu - start_cpu;
  double cpuTime = elapsec_cpu.count();

  //----------- CUDA ----------//

  cudaProfilerStart();
  //  auto start_cuda = std::chrono::system_clock::now();
  auto start_cuda = std::chrono::high_resolution_clock::now();

  group_count = 0;
  std::vector<std::vector<Acts::Seed<SpacePoint>>> seedVector_cuda;
  groupIt = spGroup.begin();

  if (do_gpu) {
    for (int i_s = 0; i_s < skip; i_s++)
      ++groupIt;
    for (; !(groupIt == spGroup.end()); ++groupIt) {
      seedVector_cuda.push_back(seedfinder_cuda.createSeedsForGroup(
                                                                    groupIt.bottom(), groupIt.middle(), groupIt.top(),w));
      group_count++;
      if (allgroup == false) {
        if (group_count >= nGroupToIterate)
          break;
      }
    }
  }
  //  auto end_cuda = std::chrono::system_clock::now();
  auto end_cuda = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsec_cuda = end_cuda - start_cuda;
  double cudaTime = elapsec_cuda.count();
  w.timeSeedfinder += cudaTime;

  cudaProfilerStop();
  std::cout << "Analyzed " << group_count << " groups for CUDA" << std::endl;

  std::cout << std::endl;
  std::cout << "----------------------- Time Metric -----------------------"
            << std::endl;
  std::cout << "                       " << (do_cpu ? "CPU" : "   ")
            << "          CUDA        " << (do_cpu ? "Speedup " : "")
            << std::endl;
  std::cout << "Seedfinding_Time  " << std::setw(11)
            << (do_cpu ? std::to_string(cpuTime) : "") << "  " << std::setw(11)
            << (do_gpu ? std::to_string(cudaTime) : "") << "  " << std::setw(11)
    //            << cudaTime << "  " << std::setw(11)
            << ((do_cpu&&do_gpu) ? std::to_string(cpuTime / cudaTime) : "") << std::endl;
  double wallTime_cpu = cpuTime + preprocessTime;
  double wallTime_cuda = cudaTime + preprocessTime;
  std::cout << "Wall_time         " << std::setw(11)
            << (do_cpu ? std::to_string(wallTime_cpu) : "") << "  " << std::setw(11) 
            << (do_gpu ? std::to_string(wallTime_cuda) : "") << "  " << std::setw(11) 
    //            << std::setw(11) << wallTime_cuda << "  " << std::setw(11)
            << ( (do_cpu&&do_gpu) ? std::to_string(wallTime_cpu / wallTime_cuda) : "")
            << std::endl;
  std::cout << "-----------------------------------------------------------"
            << std::endl;
  std::cout << std::endl;

  int nSeed_cpu = 0;
  for (auto& outVec : seedVector_cpu) {
    nSeed_cpu += outVec.size();
  }

  int nSeed_cuda = 0;
  for (auto& outVec : seedVector_cuda) {
    nSeed_cuda += outVec.size();
  }

  std::cout << "Number of Seeds (CPU | CUDA): " << nSeed_cpu << " | "
            << nSeed_cuda << std::endl;

  if (! (do_gpu&&do_cpu) ) {
    return(0);
  }
 
  
  int nMatch = 0;

  for (size_t i = 0; i < seedVector_cpu.size(); i++) {
    auto regionVec_cpu = seedVector_cpu[i];
    auto regionVec_cuda = seedVector_cuda[i];

    std::vector<std::vector<SpacePoint>> seeds_cpu;
    std::vector<std::vector<SpacePoint>> seeds_cuda;

    // for (size_t i_cpu = 0; i_cpu < regionVec_cpu.size(); i_cpu++) {
    for (auto sd : regionVec_cpu) {
      std::vector<SpacePoint> seed_cpu;
      seed_cpu.push_back(*(sd.sp()[0]));
      seed_cpu.push_back(*(sd.sp()[1]));
      seed_cpu.push_back(*(sd.sp()[2]));

      seeds_cpu.push_back(seed_cpu);
    }

    for (auto sd : regionVec_cuda) {
      std::vector<SpacePoint> seed_cuda;
      seed_cuda.push_back(*(sd.sp()[0]));
      seed_cuda.push_back(*(sd.sp()[1]));
      seed_cuda.push_back(*(sd.sp()[2]));

      seeds_cuda.push_back(seed_cuda);
    }

    if (do_cpu && do_gpu) {
      for (auto seed : seeds_cpu) {
        for (auto other : seeds_cuda) {
          if (seed[0] == other[0] && seed[1] == other[1] && seed[2] == other[2]) {
            nMatch++;
            break;
          }
        }
      }
    }
  }

  if (do_cpu && do_gpu) {
    std::cout << nMatch << " seeds are matched" << std::endl;
    std::cout << "Matching rate: " << float(nMatch) / nSeed_cpu * 100 << "%"
              << std::endl;
  }

  if (!quiet) {
    if (do_cpu) {
      std::cout << "CPU Seed result:" << std::endl;

      for (auto& regionVec : seedVector_cpu) {
        for (size_t i = 0; i < regionVec.size(); i++) {
          const Acts::Seed<SpacePoint>* seed = &regionVec[i];
          const SpacePoint* sp = seed->sp()[0];
          std::cout << " (" << sp->x() << ", " << sp->y() << ", " << sp->z()
                    << ") ";
          sp = seed->sp()[1];
          std::cout << sp->surface << " (" << sp->x() << ", " << sp->y() << ", "
                    << sp->z() << ") ";
          sp = seed->sp()[2];
          std::cout << sp->surface << " (" << sp->x() << ", " << sp->y() << ", "
                    << sp->z() << ") ";
          std::cout << std::endl;
        }
      }

      std::cout << std::endl;
    }
    if (do_gpu) {
      std::cout << "CUDA Seed result:" << std::endl;

      for (auto& regionVec : seedVector_cuda) {
        for (size_t i = 0; i < regionVec.size(); i++) {
          const Acts::Seed<SpacePoint>* seed = &regionVec[i];
          const SpacePoint* sp = seed->sp()[0];
          std::cout << " (" << sp->x() << ", " << sp->y() << ", " << sp->z()
                    << ") ";
          sp = seed->sp()[1];
          std::cout << sp->surface << " (" << sp->x() << ", " << sp->y() << ", "
                    << sp->z() << ") ";
          sp = seed->sp()[2];
          std::cout << sp->surface << " (" << sp->x() << ", " << sp->y() << ", "
                    << sp->z() << ") ";
          std::cout << std::endl;
        }
      }
    }
  }

  std::cout << std::endl;
  std::cout << std::endl;

  return 0;
}
