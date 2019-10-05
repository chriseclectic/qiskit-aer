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



#ifndef _aer_tesnor_matmul_hpp_
#define _aer_tesnor_matmul_hpp_

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "block_updater.hpp"
#include "kernel.hpp"


namespace AER {
namespace Tensor {
namespace Ops {

// Type aliases

using dynamic_reg_t = std::vector<uint_t>;
template <size_t N> using static_reg_t = std::array<uint_t, N>;

using indexes_t = std::unique_ptr<uint_t[]>;
using complex_t = std::complex<double>;
using cvector_t = std::vector<complex_t>;
using rvector_t = std::vector<double>;

//-----------------------------------------------------------------------
// Apply Matrices
//-----------------------------------------------------------------------

// Apply a N-qubit matrix to the state vector.
// The matrix is input as vector of the column-major vectorized N-qubit matrix.
template<typename T1, typename T2>
void apply_matrix(T1& data, const dynamic_reg_t &qubits, const T2 &vmat);

/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template<typename T1, typename T2>
void apply_matrix(T1& data, const dynamic_reg_t &qubits, const T2 &vmat) {

  // Calculate number of qubits
  const size_t nq = static_cast<size_t>(std::log2(data.size()));

  // Initialize block updater
  // TODO: OpenMP config
  BlockUpdater updater;

  // Static array optimized lambda functions
  const size_t N = qubits.size();
  switch (N) {
    case 1: {
      // TODO: Use optimized single-qubit case of multi-controlled gate
      //apply_mc1q(qubits, mat);
  
      static_reg_t<1> qs({qubits[0]});
      std::copy_n(qubits.begin(), 1, qs.begin());
      updater.apply_lambda(data, nq, matrix_kernel, qs, vmat);
      return;
    }
    case 2: {
      // TODO: Use optimized two-qubit case of multi-controlled gate
      //apply_mc2q(qubits, mat);
    
      static_reg_t<2> qs({qubits[0], qubits[1]});
      updater.apply_lambda(data, nq, matrix_kernel, qs, vmat);
      return;
    }
    case 3: {
      // Lambda function for 3-qubit matrix multiplication
      static_reg_t<3> qs;
      std::copy_n(qubits.begin(), 3, qs.begin());
      updater.apply_lambda(data, nq, matrix_kernel, qs, vmat);
      return;
    }
    case 4: {
      // Lambda function for 4-qubit matrix multiplication
      static_reg_t<3> qs;
      std::copy_n(qubits.begin(), 4, qs.begin());
      updater.apply_lambda(data, nq, matrix_kernel, qs, vmat);
      return;
    }
    default: {
      // Slower dynamic update function
      auto lambda = [=](T1& _data, const dynamic_reg_t &inds, const T2 &_mat)->void {
        matrix_kernel(_data, inds, _mat, N);
      };
      updater.apply_lambda(data, nq, lambda, qubits, vmat);
    }
  } // end switch
}

//------------------------------------------------------------------------------
} // namespace Ops
//------------------------------------------------------------------------------
} // namespace Tensor
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif // end module
