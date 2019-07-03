// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

// clang-format off
#define BOOST_TEST_MODULE KalmanVertexUpdator Tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/output_test_stream.hpp>
// clang-format on

#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/MagneticField/ConstantBField.hpp"
#include "Acts/Tests/CommonHelpers/FloatComparisons.hpp"
#include "Acts/Vertexing/KalmanVertexUpdator.hpp"
#include "Acts/Vertexing/TrackAtVertex.hpp"
#include "Acts/Utilities/Units.hpp"
#include "Acts/Utilities/Definitions.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/EigenStepper.hpp"
#include "Acts/Vertexing/LinearizedTrackFactory.hpp"

namespace bdata = boost::unit_test::data;
using namespace Acts::UnitLiterals;

namespace Acts {
namespace Test {

using Covariance = BoundSymMatrix;

// Create a test context
GeometryContext tgContext = GeometryContext();
MagneticFieldContext mfContext = MagneticFieldContext();

// Vertex x/y position distribution
std::uniform_real_distribution<> vXYDist(-0.1_mm, 0.1_mm);
// Vertex z position distribution
std::uniform_real_distribution<> vZDist(-20_mm, 20_mm);
// Track d0 distribution
std::uniform_real_distribution<> d0Dist(-0.01_mm, 0.01_mm);
// Track z0 distribution
std::uniform_real_distribution<> z0Dist(-0.2_mm, 0.2_mm);
// Track pT distribution
std::uniform_real_distribution<> pTDist(0.4_GeV, 10_GeV);
// Track phi distribution
std::uniform_real_distribution<> phiDist(-M_PI, M_PI);
// Track theta distribution
std::uniform_real_distribution<> thetaDist(1.0, M_PI - 1.0);
// Track charge helper distribution
std::uniform_real_distribution<> qDist(-1, 1);
// Track IP resolution distribution
std::uniform_real_distribution<> resIPDist(0., 100_um);
// Track angular distribution
std::uniform_real_distribution<> resAngDist(0., 0.1);
// Track q/p resolution distribution
std::uniform_real_distribution<> resQoPDist(-0.01, 0.01);
// Number of vertices per test event distribution

///
/// @brief Unit test for KalmanVertexUpdator
///
BOOST_AUTO_TEST_CASE(Kalman_Vertex_Updator) {
  bool debug = true;

  // Number of tests
  unsigned int nTests = 10;

  // Set up RNG
  int mySeed = 31415;
  std::mt19937 gen(mySeed);

  // Set up constant B-Field
  ConstantBField bField(0.0, 0.0, 1_T);

  // Set up Eigenstepper
  EigenStepper<ConstantBField> stepper(bField);

  // Set up propagator with void navigator
  Propagator<EigenStepper<ConstantBField>> propagator(stepper);

  // Set up LinearizedTrackFactory, needed for linearizing the tracks
  LinearizedTrackFactory<ConstantBField,
                         Propagator<EigenStepper<ConstantBField>>>::Config
      ltConfig(bField);
  LinearizedTrackFactory<ConstantBField,
                         Propagator<EigenStepper<ConstantBField>>>
      linFactory(ltConfig);

  // The track updator to be tested
  KalmanVertexUpdator<BoundParameters> updator;

  // Create perigee surface at origin
  std::shared_ptr<PerigeeSurface> perigeeSurface =
      Surface::makeShared<PerigeeSurface>(Vector3D(0., 0., 0.));

  // Creates a random tracks around origin and a random vertex.
  // VertexUpdator adds track to vertex and updates the position
  // which should afterwards be closer to the origin/track
  for (unsigned int i = 0; i < nTests; ++i) {
    if (debug) {
      std::cout << "Test " << i + 1 << std::endl;
    }
    // Construct positive or negative charge randomly
    double q = qDist(gen) < 0 ? -1. : 1.;

    // Construct random track parameters around origin
    TrackParametersBase::ParVector_t paramVec;

    paramVec << d0Dist(gen), z0Dist(gen), phiDist(gen), thetaDist(gen),
        q / pTDist(gen), 0.;

    if (debug) {
      std::cout << "Creating track parameters: " << paramVec << std::endl;
    }

    // Fill vector of track objects with simple covariance matrix
    std::unique_ptr<Covariance> covMat = std::make_unique<Covariance>();

    // Resolutions
    double res_d0 = resIPDist(gen);
    double res_z0 = resIPDist(gen);
    double res_ph = resAngDist(gen);
    double res_th = resAngDist(gen);
    double res_qp = resQoPDist(gen);

    (*covMat) << res_d0 * res_d0, 0., 0., 0., 0., 0., 0., res_z0 * res_z0, 0.,
        0., 0., 0., 0., 0., res_ph * res_ph, 0., 0., 0., 0., 0., 0.,
        res_th * res_th, 0., 0., 0., 0., 0., 0., res_qp * res_qp, 0., 0., 0.,
        0., 0., 0., 1.;
    BoundParameters params(tgContext, std::move(covMat), paramVec,
                           perigeeSurface);

    // Linearized state of the track
    LinearizedTrack linTrack =
        linFactory
            .linearizeTrack(tgContext, mfContext, &params,
                            SpacePointVector::Zero(), propagator)
            .value();

    // Create TrackAtVertex
    TrackAtVertex<BoundParameters> trkAtVtx(0., params, params);

    // Set linearized state of trackAtVertex
    trkAtVtx.linearizedState = linTrack;

    // Create a vertex
    Vector3D vtxPos(vXYDist(gen), vXYDist(gen), vZDist(gen));
    Vertex<BoundParameters> vtx(vtxPos);
    vtx.setFullCovariance(SpacePointSymMatrix::Identity());

    // Update trkAtVertex with assumption of originating from vtx
    updator.addAndUpdate(&vtx, trkAtVtx);

    if (debug) {
      std::cout << "Old vertex position: " << vtxPos << std::endl;
      std::cout << "New vertex position: " << vtx.position() << std::endl;
    }

    double oldDistance = vtxPos.norm();
    double newDistance = vtx.position().norm();

    if (debug) {
      std::cout << "Old distance: " << oldDistance << std::endl;
      std::cout << "New distance: " << newDistance << std::endl;
    }

    // After update, vertex should be closer to the track
    BOOST_CHECK(newDistance < oldDistance);

    // Track should have been added to the vertex
    BOOST_CHECK(vtx.tracks().size() > 0);

  }  // end for loop

}  // end test case

}  // namespace Test
}  // namespace Acts