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

#ifndef _aer_framework_linalg_matrix_functions_hpp_
#define _aer_framework_linalg_matrix_functions_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/linalg/almost_equal.hpp"
#include "framework/linalg/enable_if_numeric.hpp"
#include "framework/linalg/matrix.hpp"
#include "framework/types.hpp"

namespace AER {
namespace Linalg {
namespace Predicates {

//------------------------------------------------------------------------------
// Matrix Functions
//------------------------------------------------------------------------------

// Return True if the input matrix is square
template <class T, typename = enable_if_numeric<T>>
bool is_square(const matrix<T> &mat);

// Return true if the input matrix is a matrix diagonal
// [[m[0, 0], m[1, 1], ..., m[n-1, n-1]]]
template <class T, typename = enable_if_numeric<T>>
bool is_diagonal(const matrix<T> &mat);

// Return true if two matrices are approximatly equal
template <class T1, class T2, typename = enable_if_numeric<T1>,
          typename = enable_if_numeric<T2>>
bool is_equal(const matrix<T1> &mat1, const matrix<T2> &mat2,
              T1 max_diff = std::numeric_limits<T1>::epsilon(),
              T1 max_relative_diff = std::numeric_limits<T1>::epsilon());

// Return true if the matrix is approximately diagonal
template <class T, typename = enable_if_numeric<T>>
bool is_diagonal(const matrix<T> &mat,
                 T max_diff = std::numeric_limits<T>::epsilon(),
                 T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the input matrix is an identity matrix
template <class T, typename = enable_if_numeric<T>>
bool is_identity(const matrix<T> &mat,
                 T max_diff = std::numeric_limits<T>::epsilon(),
                 T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the input matrix is exp(i * theta) * identity
template <class T, typename = enable_if_numeric<T>>
std::pair<bool, double> is_identity_phase(
    const matrix<T> &mat, T max_diff = std::numeric_limits<T>::epsilon(),
    T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the input matrix is the diagonal of an identity matrix
// This is equivlaent to being a column matrix [[1, ..., 1]]
template <class T, typename = enable_if_numeric<T>>
bool is_diagonal_identity(
    const matrix<T> &mat, T max_diff = std::numeric_limits<T>::epsilon(),
    T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the input matrix is unitary
template <class T, typename = enable_if_numeric<T>>
bool is_unitary(const matrix<T> &mat,
                T max_diff = std::numeric_limits<T>::epsilon(),
                T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the input matrix is Hermitian
template <class T, typename = enable_if_numeric<T>>
bool is_hermitian(const matrix<T> &mat,
                  T max_diff = std::numeric_limits<T>::epsilon(),
                  T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the input matrix is symmetric
template <class T, typename = enable_if_numeric<T>>
bool is_symmetrix(const matrix<T> &mat,
                  T max_diff = std::numeric_limits<T>::epsilon(),
                  T max_relative_diff = std::numeric_limits<T>::epsilon());

// Return true if the list of Kraus matrices is CPTP.
template <class T, typename = enable_if_numeric<T>>
bool is_cptp_kraus(const std::vector<matrix<T>> &kraus,
                   T max_diff = std::numeric_limits<T>::epsilon(),
                   T max_relative_diff = std::numeric_limits<T>::epsilon());

//==============================================================================
// Implementations: Matrix functions
//==============================================================================

template <class T, typename = enable_if_numeric<T>>
bool is_square(const matrix<T> &mat) {
  if (mat.GetRows() != mat.GetColumns()) return false;
  return true;
}

template <class T, typename = enable_if_numeric<T>>
bool is_diagonal(const matrix<T> &mat) {
  // Check if row-matrix for diagonal
  if (mat.GetRows() == 1 && mat.GetColumns() > 0) return true;
  return false;
}

template <class T1, class T2, typename = enable_if_numeric<T1>,
          typename = enable_if_numeric<T2>>
bool is_equal(const matrix<T1> &mat1, const matrix<T2> &mat2, T1 max_diff,
              T1 max_relative_diff) {
  // Check matrices are same shape
  const auto nrows = mat1.GetRows();
  const auto ncols = mat1.GetColumns();
  if (nrows != mat2.GetRows() || ncols != mat2.GetColumns()) return false;

  // Check matrices are equal on an entry by entry basis
  double delta = 0;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      if (!approx_equal(mat1(i, j), mat2(i, j), max_diff, max_relative_diff))
        return false;
    }
  }
  return true;
}

template <class T, typename = enable_if_numeric<T>>
bool is_diagonal(const matrix<T> &mat, T max_diff, T max_relative_diff) {
  // Check U matrix is identity
  const auto nrows = mat.GetRows();
  const auto ncols = mat.GetColumns();
  if (nrows != ncols) return false;
  for (size_t i = 0; i < nrows; i++)
    for (size_t j = 0; j < ncols; j++)
      if (i != j && !almost_equal(mat(i, j), T(0), max_diff, max_relative_diff))
        return false;
  return true;
}

template <class T, typename = enable_if_numeric<T>>
std::pair<bool, double> is_identity_phase(const matrix<T> &mat, T max_diff,
                                          T max_relative_diff) {
  // To check if identity we first check we check that:
  // 1. U(0,0) = exp(i * theta)
  // 2. U(i, i) = U(0, 0)
  // 3. U(i, j) = 0 for j != i
  auto failed = std::make_pair(false, 0.0);

  // Check condition 1
  const auto u00 = mat(0, 0);
  if (!almost_equal(std::abs(u00), T(1))) {
    return failed;
  }
  const auto theta = std::arg(u00);

  // Check conditions 2 and 3
  const auto nrows = mat.GetRows();
  const auto ncols = mat.GetColumns();
  if (nrows != ncols) return failed;
  for (size_t i = 0; i < nrows; i++) {
    for (size_t j = 0; j < ncols; j++) {
      auto val = (i == j) ? u00 : T(0);
      if (!almost_equal(m(i, j), val, max_diff, max_relative_diff))
        return failed;
    }
  }
  // Otherwise we pass
  return std::make_pair(true, theta);
}

template <class T, typename = enable_if_numeric<T>>
bool is_identity(const matrix<T> &mat, T max_diff, T max_relative_diff) {
  // Check mat(0, 0) == 1
  if (!almost_equal(mat(0, 0), T(1), max_diff, max_relative_diff)) return false;
  // If this passes now use is_identity_phase (and we know
  // phase will be zero).
  return is_identity_phase(mat, max_diff, max_relative_diff).first;
}

template <class T, typename = enable_if_numeric<T>>
bool is_diagonal_identity(const matrix<T> &mat, T max_diff,
                          T max_relative_diff) {
  // Check U matrix is identity
  if (!is_diagonal(mat, max_diff, max_relative_diff)) return false;
  const auto ncols = mat.GetColumns();
  for (size_t j = 0; j < ncols; j++) {
    if (!almost_equal(mat(0, j), T(1), max_diff, max_relative_diff))
      return false;
  }
  return true;
}

template <class T, typename = enable_if_numeric<T>>
bool is_unitary(const matrix<T> &mat, T max_diff, T max_relative_diff) {
  size_t nrows = mat.GetRows();
  size_t ncols = mat.GetColumns();
  // Check if diagonal row-matrix
  if (nrows == 1) {
    for (size_t j = 0; j < ncols; j++) {
      if (!almost_equal(std::real(std::abs(mat(0, j))), T(1), max_diff,
                        T max_relative_diff))
        return false;
    }
    return true;
  }
  // Check U matrix is square
  if (nrows != ncols) return false;
  // Check U matrix is unitary
  const matrix<T> check = mat * dagger(mat);
  return is_identity(check, max_diff, max_relative_diff);
}

template <class T, typename = enable_if_numeric<T>>
bool is_hermitian_matrix(const matrix<T> &mat, T max_diff,
                         T max_relative_diff) {
  return is_equal(mat, dagger(mat), max_diff, max_relative_diff);
}

template <class T, typename = enable_if_numeric<T>>
bool is_symmetrix(const matrix<T> &mat, T max_diff, T max_relative_diff) {
  return is_equal(mat, transpose(mat), max_diff, max_relative_diff);
}

template <class T, typename = enable_if_numeric<T>>
bool is_cptp_kraus(const std::vector<matrix<T>> &mats, T max_diff,
                   T max_relative_diff) {
  matrix<T> cptp(mats[0].size());
  for (const auto &mat : mats) {
    cptp = cptp + dagger(mat) * mat;
  }
  return is_identity(cptp, max_diff, max_relative_diff);
}

//------------------------------------------------------------------------------
}  // namespace Predicates
}  // namespace Linalg
}  // namespace AER
//------------------------------------------------------------------------------
#endif