/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_linalg_matrix_utils_vmatrix_defs_hpp_
#define _aer_framework_linalg_matrix_utils_vmatrix_defs_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/linalg/matrix_utils/matrix_defs.hpp"
#include "framework/types.hpp"
#include "framework/utils.hpp"

namespace AER {
namespace Linalg {

//------------------------------------------------------------------------------
// Static column-vectorized matrices
//------------------------------------------------------------------------------

class VMatrix {
 public:
  // Single-qubit gates
  const static cvector_t I;    // name: "id"
  const static cvector_t X;    // name: "x"
  const static cvector_t Y;    // name: "y"
  const static cvector_t Z;    // name: "z"
  const static cvector_t H;    // name: "h"
  const static cvector_t S;    // name: "s"
  const static cvector_t SDG;  // name: "sdg"
  const static cvector_t T;    // name: "t"
  const static cvector_t TDG;  // name: "tdg"
  const static cvector_t X90;  // name: "x90"

  // Two-qubit gates
  const static cvector_t CX;    // name: "cx"
  const static cvector_t CZ;    // name: "cz"
  const static cvector_t SWAP;  // name: "swap"

  // Identity Matrix
  static cvector_t identity(size_t dim);

  // Single-qubit waltz gates
  static cvector_t u1(double lam);
  static cvector_t u2(double phi, double lam);
  static cvector_t u3(double theta, double phi, double lam);

  // Single-qubit rotation gates
  static cvector_t r(double phi, double lam);
  static cvector_t rx(double theta);
  static cvector_t ry(double theta);
  static cvector_t rz(double theta);

  // Two-qubit rotation gates
  static cvector_t rxx(double theta);
  static cvector_t rzz(double theta);

  // Complex arguments are implemented by taking std::real
  // of the input
  static cvector_t u1(complex_t lam) { return u1(std::real(lam)); }
  static cvector_t u2(complex_t phi, complex_t lam) {
    return u2(std::real(phi), std::real(lam));
  }
  static cvector_t u3(complex_t theta, complex_t phi, complex_t lam) {
    return u3(std::real(theta), std::real(phi), std::real(lam));
  };
  static cvector_t r(complex_t theta, complex_t phi) {
    return r(std::real(theta), std::real(phi));
  }
  static cvector_t rx(complex_t theta) { return rx(std::real(theta)); }
  static cvector_t ry(complex_t theta) { return ry(std::real(theta)); }
  static cvector_t rz(complex_t theta) { return rz(std::real(theta)); }
  static cvector_t rxx(complex_t theta) { return rxx(std::real(theta)); }
  static cvector_t rzz(complex_t theta) { return rzz(std::real(theta)); }

  // Return the matrix for a named matrix string
  // Allowed names correspond to all the const static single-qubit
  // and two-qubit gate members
  static const cvector_t from_name(const std::string &name) {
    return *label_map_.at(name);
  }

  // Check if the input name string is allowed
  static bool allowed_name(const std::string &name) {
    return (label_map_.find(name) != label_map_.end());
  }

 private:
  // Lookup table that returns a pointer to the static data member
  const static stringmap_t<const cvector_t *> label_map_;
};

//==============================================================================
// Implementations
//==============================================================================

const cvector_t VMatrix::I = Utils::vectorize_matrix(Matrix::I);

const cvector_t VMatrix::X = Utils::vectorize_matrix(Matrix::X);

const cvector_t VMatrix::Y = Utils::vectorize_matrix(Matrix::Y);

const cvector_t VMatrix::Z = Utils::vectorize_matrix(Matrix::Z);

const cvector_t VMatrix::S = Utils::vectorize_matrix(Matrix::S);

const cvector_t VMatrix::SDG = Utils::vectorize_matrix(Matrix::SDG);

const cvector_t VMatrix::T = Utils::vectorize_matrix(Matrix::T);

const cvector_t VMatrix::TDG = Utils::vectorize_matrix(Matrix::TDG);

const cvector_t VMatrix::H = Utils::vectorize_matrix(Matrix::H);

const cvector_t VMatrix::X90 = Utils::vectorize_matrix(Matrix::X90);

const cvector_t VMatrix::CX = Utils::vectorize_matrix(Matrix::CX);

const cvector_t VMatrix::CZ = Utils::vectorize_matrix(Matrix::CZ);

const cvector_t VMatrix::SWAP = Utils::vectorize_matrix(Matrix::SWAP);

// Lookup table
const stringmap_t<const cvector_t *> VMatrix::label_map_ = {
    {"id", &VMatrix::I},     {"x", &VMatrix::X},   {"y", &VMatrix::Y},
    {"z", &VMatrix::Z},      {"h", &VMatrix::H},   {"s", &VMatrix::S},
    {"sdg", &VMatrix::SDG},  {"t", &VMatrix::T},   {"tdg", &VMatrix::TDG},
    {"x90", &VMatrix::X90},  {"cx", &VMatrix::CX}, {"cz", &VMatrix::CZ},
    {"swap", &VMatrix::SWAP}};

cvector_t VMatrix::identity(size_t dim) {
  cvector_t mat(dim * dim);
  for (size_t j = 0; j < dim; j++) mat[j + j * dim] = {1.0, 0.0};
  return mat;
}

cvector_t VMatrix::u1(double lambda) {
  cvector_t mat(2 * 2);
  mat[0 + 0 * 2] = {1., 0.};
  mat[1 + 1 * 2] = std::exp(complex_t(0., lambda));
  return mat;
}

cvector_t VMatrix::u2(double phi, double lambda) {
  cvector_t mat(2 * 2);
  const complex_t i(0., 1.);
  const complex_t invsqrt2(1. / std::sqrt(2), 0.);
  mat[0 + 0 * 2] = invsqrt2;
  mat[0 + 1 * 2] = -std::exp(i * lambda) * invsqrt2;
  mat[1 + 0 * 2] = std::exp(i * phi) * invsqrt2;
  mat[1 + 1 * 2] = std::exp(i * (phi + lambda)) * invsqrt2;
  return mat;
}

cvector_t VMatrix::u3(double theta, double phi, double lambda) {
  cvector_t mat(2 * 2);
  const complex_t i(0., 1.);
  mat[0 + 0 * 2] = std::cos(theta / 2.);
  mat[0 + 1 * 2] = -std::exp(i * lambda) * std::sin(theta / 2.);
  mat[1 + 0 * 2] = std::exp(i * phi) * std::sin(theta / 2.);
  mat[1 + 1 * 2] = std::exp(i * (phi + lambda)) * std::cos(theta / 2.);
  return mat;
}

cvector_t VMatrix::r(double theta, double phi) {
  cvector_t mat(2 * 2);
  const complex_t i(0., 1.);
  mat[0 + 0 * 2] = std::cos(0.5 * theta);
  mat[0 + 1 * 2] = -i * std::exp(-i * phi) * std::sin(0.5 * theta);
  mat[1 + 0 * 2] = -i * std::exp(i * phi) * std::sin(0.5 * theta);
  mat[1 + 1 * 2] = std::cos(0.5 * theta);
  return mat;
}

cvector_t VMatrix::rx(double theta) {
  cvector_t mat(2 * 2);
  const complex_t i(0., 1.);
  mat[0 + 0 * 2] = std::cos(0.5 * theta);
  mat[0 + 1 * 2] = -i * std::sin(0.5 * theta);
  mat[1 + 0 * 2] = mat[0 + 1 * 2];
  mat[1 + 1 * 2] = mat[0 + 0 * 2];
  return mat;
}

cvector_t VMatrix::ry(double theta) {
  cvector_t mat(2 * 2);
  mat[0 + 0 * 2] = std::cos(0.5 * theta);
  mat[0 + 1 * 2] = -1.0 * std::sin(0.5 * theta);
  mat[1 + 0 * 2] = -mat[0 + 1 * 2];
  mat[1 + 1 * 2] = mat[0 + 0 * 2];
  return mat;
}

cvector_t VMatrix::rz(double theta) {
  cvector_t mat(2 * 2);
  const complex_t i(0., 1.);
  mat[0 + 0 * 2] = std::exp(-i * 0.5 * theta);
  mat[1 + 1 * 2] = std::exp(i * 0.5 * theta);
  return mat;
}

cvector_t VMatrix::rxx(double theta) {
  cvector_t mat(4 * 4);
  const complex_t i(0., 1.);
  const double cost = std::cos(0.5 * theta);
  const double sint = std::sin(0.5 * theta);
  mat[0 + 0 * 4] = cost;
  mat[0 + 3 * 4] = -i * sint;
  mat[1 + 1 * 4] = cost;
  mat[1 + 2 * 4] = -i * sint;
  mat[2 + 1 * 4] = -i * sint;
  mat[2 + 2 * 4] = cost;
  mat[3 + 0 * 4] = -i * sint;
  mat[3 + 3 * 4] = cost;
  return mat;
}

cvector_t VMatrix::rzz(double theta) {
  cvector_t mat(4 * 4);
  const complex_t i(0., 1.);
  const complex_t exp_p = std::exp(i * 0.5 * theta);
  const complex_t exp_m = std::exp(-i * 0.5 * theta);
  mat[0 + 0 * 4] = exp_m;
  mat[1 + 1 * 4] = exp_p;
  mat[2 + 2 * 4] = exp_p;
  mat[3 + 3 * 4] = exp_m;
  return mat;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif