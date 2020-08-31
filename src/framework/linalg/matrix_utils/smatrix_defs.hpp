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

#ifndef _aer_framework_linalg_matrix_utils_smatrix_defs_hpp_
#define _aer_framework_linalg_matrix_utils_smatrix_defs_hpp_

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
// Static Superoperator Matrices
//------------------------------------------------------------------------------

class SMatrix {
public:
  // Single-qubit gates
  const static cmatrix_t I;   // name: "id"
  const static cmatrix_t X;   // name: "x"
  const static cmatrix_t Y;   // name: "y"
  const static cmatrix_t Z;   // name: "z"
  const static cmatrix_t H;   // name: "h"
  const static cmatrix_t S;   // name: "s"
  const static cmatrix_t SDG; // name: "sdg"
  const static cmatrix_t T;   // name: "t"
  const static cmatrix_t TDG; // name: "tdg"
  const static cmatrix_t X90; // name: "x90"

  // Two-qubit gates
  const static cmatrix_t CX;   // name: "cx"
  const static cmatrix_t CZ;   // name: "cz"
  const static cmatrix_t SWAP; // name: "swap"

  // Identity Matrix
  static cmatrix_t identity(size_t dim);

  // Single-qubit waltz gates
  static cmatrix_t u1(double lam);
  static cmatrix_t u2(double phi, double lam);
  static cmatrix_t u3(double theta, double phi, double lam);

  // Complex arguments are implemented by taking std::real
  // of the input
  static cmatrix_t u1(complex_t lam) { return u1(std::real(lam)); }
  static cmatrix_t u2(complex_t phi, complex_t lam) {
    return u2(std::real(phi), std::real(lam));
  }
  static cmatrix_t u3(complex_t theta, complex_t phi, complex_t lam) {
    return u3(std::real(theta), std::real(phi), std::real(lam));
  };

  // Return superoperator matrix for reset instruction
  // on specified dim statespace.
  // The returned matrix is (dim * dim, dim * dim).
  static cmatrix_t reset(size_t dim);

  // Return the matrix for a named matrix string
  // Allowed names correspond to all the const static single-qubit
  // and two-qubit gate members
  static const cmatrix_t from_name(const std::string &name) {
    return *label_map_.at(name);
  }

  // Check if the input name string is allowed
  static bool allowed_name(const std::string &name) {
    return (label_map_.find(name) != label_map_.end());
  }

private:
  // Lookup table that returns a pointer to the static data member
  const static stringmap_t<const cmatrix_t *> label_map_;
};

//==============================================================================
// Implementations
//==============================================================================

const cmatrix_t SMatrix::I = Utils::unitary_superop(Matrix::I);

const cmatrix_t SMatrix::X = Utils::unitary_superop(Matrix::X);

const cmatrix_t SMatrix::Y = Utils::unitary_superop(Matrix::Y);

const cmatrix_t SMatrix::Z = Utils::unitary_superop(Matrix::Z);

const cmatrix_t SMatrix::S = Utils::unitary_superop(Matrix::S);

const cmatrix_t SMatrix::SDG = Utils::unitary_superop(Matrix::SDG);

const cmatrix_t SMatrix::T = Utils::unitary_superop(Matrix::T);

const cmatrix_t SMatrix::TDG = Utils::unitary_superop(Matrix::TDG);

const cmatrix_t SMatrix::H = Utils::unitary_superop(Matrix::H);

const cmatrix_t SMatrix::X90 = Utils::unitary_superop(Matrix::X90);

const cmatrix_t SMatrix::CX = Utils::unitary_superop(Matrix::CX);

const cmatrix_t SMatrix::CZ = Utils::unitary_superop(Matrix::CZ);

const cmatrix_t SMatrix::SWAP = Utils::unitary_superop(Matrix::SWAP);

// Lookup table
const stringmap_t<const cmatrix_t *> SMatrix::label_map_ = {
    {"id", &SMatrix::I},     {"x", &SMatrix::X},   {"y", &SMatrix::Y},
    {"z", &SMatrix::Z},      {"h", &SMatrix::H},   {"s", &SMatrix::S},
    {"sdg", &SMatrix::SDG},  {"t", &SMatrix::T},   {"tdg", &SMatrix::TDG},
    {"x90", &SMatrix::X90},  {"cx", &SMatrix::CX}, {"cz", &SMatrix::CZ},
    {"swap", &SMatrix::SWAP}};

cmatrix_t SMatrix::identity(size_t dim) { return Matrix::identity(dim * dim); }

cmatrix_t SMatrix::u1(double lambda) {
  cmatrix_t mat(4, 4);
  mat(0, 0) = {1., 0.};
  mat(1, 1) = std::exp(complex_t(0., lambda));
  mat(2, 2) = std::exp(complex_t(0., -lambda));
  mat(3, 3) = {1., 0.};
  return mat;
}

cmatrix_t SMatrix::u2(double phi, double lambda) {
  return Utils::tensor_product(Matrix::u2(-phi, -lambda),
                               Matrix::u2(phi, lambda));
}

cmatrix_t SMatrix::u3(double theta, double phi, double lambda) {
  return Utils::tensor_product(Matrix::u3(theta, -phi, -lambda),
                               Matrix::u3(theta, phi, lambda));
}

//------------------------------------------------------------------------------
} // end namespace Linalg
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif