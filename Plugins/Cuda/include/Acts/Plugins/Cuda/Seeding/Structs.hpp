#ifndef SEEDFINDER_STRUCTS_H
#define SEEDFINDER_STRUCTS_H 1

// #define MAX_NSPM 2048
// #define MAX_NSPB 9216
// #define MAX_NSPT 9216

#define MAX_NSPM 1024
#define MAX_NSPB 4092
#define MAX_NSPT 4092

namespace GPUStructs {
struct Config {
  float deltaRMin{0.};
  float deltaRMax{0.};
  float cotThetaMax{0.};
  float collisionRegionMin{0.};
  float collisionRegionMax{0.};
  float maxScatteringAngle2{0.};
  float sigmaScattering{0.};
  float minHelixDiameter2{0.};
  float pT2perRadius{0.};
  float impactMax{0.};
  float deltaInvHelixDiameter{0.};
  float impactWeightFactor{0.};
  float filterDeltaRMin{0.};
  float compatSeedWeight{0.};
  size_t compatSeedLimit{0};
};

struct Flatten {
  int nSpM{0};
  int nSpB{0};
  int nSpT{0};

  float spMmat[MAX_NSPM*6];
  float spBmat[MAX_NSPB*6];
  float spTmat[MAX_NSPT*6];
  // float spMmat[MAX_NSPM][6];
  // float spBmat[MAX_NSPB][6];
  // float spTmat[MAX_NSPT][6];
};

struct Doublet {

  int nSpMcomp{0};
  int nSpBcompPerSpMMax{0};
  int nSpTcompPerSpMMax{0};
  int nSpBcompPerSpM[MAX_NSPM];
  int McompIndex[MAX_NSPM];
  int BcompIndex[MAX_NSPB][MAX_NSPM];
  int TcompIndex[MAX_NSPT][MAX_NSPM];
  int tmpBcompIndex[MAX_NSPB][MAX_NSPM];
  int tmpTcompIndex[MAX_NSPT][MAX_NSPM];
};

// struct Transform {

// };

// struct Triplet {
// };

}
#endif
