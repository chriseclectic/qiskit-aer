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



#ifndef _tensor_kernel_hpp_
#define _tensor_kernel_hpp_

#include <array>
#include <cstdint>
#include <vector>


namespace AER {
namespace Tensor {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using dynamic_reg_t = std::vector<uint_t>;
template <size_t N>
using static_reg_t = std::array<uint_t, N>;
using indexes_t = std::unique_ptr<uint_t[]>;


// STATIC CASE
template <typename T, typename A, typename M, size_t N>
void matrix_kernel(A& data,
                   const static_reg_t<N> inds,
                   const M& mat) {
  // Write block to temp and clear input array
  std::array<T, N> tmp;
  for (size_t i = 0; i < N; i++) {
    const auto ii = inds[i];
    tmp[i] = data[ii];
    data[ii] = 0.;
  }
  // Update the array with block matrix multiplication output
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      data[inds[i]] += mat[i + N * j] * tmp[j];
}


// DYNAMIC CASE
template <typename T, typename A, typename M>
void matrix_kernel(A& data, const dynamic_reg_t inds,
                   const M& mat, size_t N) {
  // Write block to temp and clear input array
  std::vector<T> tmp(N);
  for (size_t i = 0; i < N; i++) {
    const auto ii = inds[i];
    tmp[i] = data[ii];
    data[ii] = 0.;
  }
  // Update the array with block matrix multiplication output
  for (size_t i = 0; i < N; i++)
    for (size_t j = 0; j < N; j++)
      data[inds[i]] += mat[i + N * j] * tmp[j];
}

//------------------------------------------------------------------------------
} // namespace Tensor
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif // end module
