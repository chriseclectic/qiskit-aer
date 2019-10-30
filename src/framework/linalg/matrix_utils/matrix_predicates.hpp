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

#ifndef _aer_framework_linalg_matrix_predicates_hpp_
#define _aer_framework_linalg_matrix_predicates_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/linalg/enable_if_numeric.hpp"
#include "framework/linalg/matrix_utils/matrix_functions.hpp"
#include "framework/types.hpp"

namespace AER {
namespace Linalg {

//------------------------------------------------------------------------------
// Matrix predicates
//------------------------------------------------------------------------------

// Return true if the input matrix is square
template <class T>
bool is_square(const matrix<T> &mat);

// Return true if the input matrix is diagonal
template <class T>
bool is_diagonal(const matrix<T> &mat);

// Return true if the input matrix is diagonal
template <class T>
bool is_diagonal(const matrix<T> &mat, double threshold);

// Return true if two matrices are elementwise equal
template <class T>
bool is_equal(const matrix<T> &mat1, const matrix<T> &mat2, double threshold);

// Return true if the matrix is scalar multiple of the identity
template <class T>
std::pair<bool, double> is_identity_phase(const matrix<T> &mat,
                                          double threshold);

// Return true if the input matrix is an identity matrix
template <class T>
bool is_identity(const matrix<T> &mat, double threshold);

// Return true if the input matrix is row matrix of identity
// mat = [[1, ..., 1]]
template <class T>
bool is_diagonal_identity(const matrix<T> &mat, double threshold);

// Return true if the input matrix is unitary
template <class T>
bool is_unitary(const matrix<T> &mat, double threshold);

// Return true if the input matrix is Hermitian
template <class T>
bool is_hermitian(const matrix<T> &mat, double threshold);

// Return true if the input matrix is symmetric
template <class T>
bool is_symmetrix(const matrix<T> &mat, double threshold);

// Return true if the set of matrices satisfy CPTP Kraus condition
template <class T>
bool is_cptp_kraus(const std::vector<matrix<T>> &kraus, double threshold);

//==============================================================================
// Implementations
//==============================================================================

template <class T>
bool is_square(const matrix<T> &mat) {
  if (mat.GetRows() != mat.GetColumns()) return false;
  return true;
}

template <class T>
bool is_diagonal(const matrix<T> &mat) {
  // Check if row-matrix for diagonal
  if (mat.GetRows() == 1 && mat.GetColumns() > 0) return true;
  return false;
}

template <class T>
bool is_equal(const matrix<T> &mat1, const matrix<T> &mat2, double threshold) {
  // Check matrices are same shape
  const auto nrows = mat1.GetRows();
  const auto ncols = mat1.GetColumns();
  if (nrows != mat2.GetRows() || ncols != mat2.GetColumns()) return false;

  // Check matrices are equal on an entry by entry basis
  double delta = 0;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      delta += std::real(std::abs(mat1(i, j) - mat2(i, j)));
    }
  }
  return (delta < threshold);
}

template <class T>
bool is_diagonal(const matrix<T> &mat, double threshold) {
  // Check U matrix is identity
  const auto nrows = mat.GetRows();
  const auto ncols = mat.GetColumns();
  if (nrows != ncols) return false;
  for (size_t i = 0; i < nrows; i++)
    for (size_t j = 0; j < ncols; j++)
      if (i != j && std::real(std::abs(mat(i, j))) > threshold) return false;
  return true;
}

template <class T>
std::pair<bool, double> is_identity_phase(const matrix<T> &mat,
                                          double threshold) {
  // To check if identity we first check we check that:
  // 1. U(0,0) = exp(i * theta)
  // 2. U(i, i) = U(0, 0)
  // 3. U(i, j) = 0 for j != i
  auto failed = std::make_pair(false, 0.0);

  // Check condition 1.
  const auto u00 = mat(0, 0);
  // if (std::norm(std::abs(u00) - 1.0) > threshold)
  //  return failed;
  if (std::norm(std::abs(u00) - 1.0) > threshold) {
    return failed;
  }
  const auto theta = std::arg(u00);

  // Check conditions 2 and 3
  double delta = 0.;
  const auto nrows = mat.GetRows();
  const auto ncols = mat.GetColumns();
  if (nrows != ncols) return failed;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      auto val = (i == j) ? std::norm(mat(i, j) - u00) : std::norm(mat(i, j));
      if (val > threshold) {
        return failed;  // fail fast if single entry differs
      } else
        delta += val;  // accumulate difference
    }
  }
  // Check small errors didn't accumulate
  if (delta > threshold) {
    return failed;
  }
  // Otherwise we pass
  return std::make_pair(true, theta);
}

template <class T>
bool is_identity(const matrix<T> &mat, double threshold) {
  // Check mat(0, 0) == 1
  if (std::norm(mat(0, 0) - T(1)) > threshold) return false;
  // If this passes now use is_identity_phase (and we know
  // phase will be zero).
  return is_identity_phase(mat, threshold).first;
}

template <class T>
bool is_diagonal_identity(const matrix<T> &mat, double threshold) {
  // Check U matrix is identity
  if (is_diagonal(mat, threshold) == false) return false;
  double delta = 0.;
  const auto ncols = mat.GetColumns();
  for (size_t j = 0; j < ncols; j++) {
    delta += std::real(std::abs(mat(0, j) - 1.0));
  }
  return (delta < threshold);
}

template <class T>
bool is_unitary(const matrix<T> &mat, double threshold) {
  size_t nrows = mat.GetRows();
  size_t ncols = mat.GetColumns();
  // Check if diagonal row-matrix
  if (nrows == 1) {
    for (size_t j = 0; j < ncols; j++) {
      double delta = std::abs(1.0 - std::real(std::abs(mat(0, j))));
      if (delta > threshold) return false;
    }
    return true;
  }
  // Check U matrix is square
  if (nrows != ncols) return false;
  // Check U matrix is unitary
  const matrix<T> check = mat * dagger(mat);
  return is_identity(check, threshold);
}

template <class T>
bool is_hermitian_matrix(const matrix<T> &mat, double threshold) {
  return is_equal(mat, dagger(mat), threshold);
}

template <class T>
bool is_symmetrix(const matrix<T> &mat, double threshold) {
  return is_equal(mat, transpose(mat), threshold);
}

template <class T>
bool is_cptp_kraus(const std::vector<matrix<T>> &mats, double threshold) {
  matrix<T> cptp(mats[0].size());
  for (const auto &mat : mats) {
    cptp = cptp + dagger(mat) * mat;
  }
  return is_identity(cptp, threshold);
}

//------------------------------------------------------------------------------
}  // namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif