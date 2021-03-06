// This file is part of the Acts project.
//
// Copyright (C) 2019 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/Utilities/Definitions.hpp"
#include "Acts/Visualization/IVisualization.hpp"

#include <utility>
#include <vector>

namespace Acts {

/// This helper produces output in the OBJ format. Note that colors are not
/// supported in this implementation.
///
template <typename T = double>
class ObjHelper : public IVisualization {
 public:
  static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                "Use either double or float");

  /// Stored value type, should be double or float
  using ValueType = T;

  /// Type of a vertex based on the value type
  using VertexType = ActsVector<ValueType, 3>;

  /// Type of a line
  using LineType = std::pair<size_t, size_t>;

  /// @copydoc Acts::IVisualization::vertex()
  void vertex(const Vector3D& vtx,
              IVisualization::ColorType color = {0, 0, 0}) override {
    (void)color;  // suppress unused warning
    m_vertices.push_back(vtx.template cast<ValueType>());
  }

  /// @copydoc Acts::IVisualization::line()
  void line(const Vector3D& a, const Vector3D& b,
            IVisualization::ColorType color = {0, 0, 0}) override {
    (void)color;  // suppress unused warning
    // not implemented
    vertex(a);
    vertex(b);
    m_lines.push_back({m_vertices.size() - 2, m_vertices.size() - 1});
  }

  /// @copydoc Acts::IVisualization::face()
  void face(const std::vector<Vector3D>& vtxs,
            IVisualization::ColorType color = {0, 0, 0}) override {
    (void)color;  // suppress unused warning
    FaceType idxs;
    idxs.reserve(vtxs.size());
    for (const auto& vtx : vtxs) {
      vertex(vtx);
      idxs.push_back(m_vertices.size() - 1);
    }
    m_faces.push_back(std::move(idxs));
  }

  /// @copydoc Acts::IVisualization::faces()
  void faces(const std::vector<Vector3D>& vtxs,
             const std::vector<FaceType>& faces,
             ColorType color = {120, 120, 120}) {
    // No faces given - call the face() method
    if (faces.empty()) {
      face(vtxs, color);
    } else {
      auto vtxoffs = m_vertices.size();
      m_vertices.insert(m_vertices.end(), vtxs.begin(), vtxs.end());
      for (const auto& face : faces) {
        if (face.size() == 2) {
          m_lines.push_back({face[0] + vtxoffs, face[2] + vtxoffs});
        } else {
          FaceType rawFace = face;
          std::transform(rawFace.begin(), rawFace.end(), rawFace.begin(),
                         [&](size_t& iv) { return (iv + vtxoffs); });
          m_faces.push_back(rawFace);
        }
      }
    }
  }

  /// @copydoc Acts::IVisualization::write()
  void write(std::ostream& os) const override {
    for (const VertexType& vtx : m_vertices) {
      os << "v " << vtx.x() << " " << vtx.y() << " " << vtx.z() << "\n";
    }

    for (const LineType& ln : m_lines) {
      os << "l " << ln.first + 1 << " " << ln.second + 1 << "\n";
    }

    for (const FaceType& fc : m_faces) {
      os << "f";
      for (size_t i = 0; i < fc.size(); i++) {
        os << " " << fc[i] + 1;
      }
      os << "\n";
    }
  }

  ///  @copydoc Acts::IVisualization::clear()
  void clear() override {
    m_vertices.clear();
    m_faces.clear();
    m_lines.clear();
  }

 private:
  std::vector<VertexType> m_vertices;
  std::vector<FaceType> m_faces;
  std::vector<LineType> m_lines;
};
}  // namespace Acts
