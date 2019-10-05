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

#ifndef _aer_tensor_block_update_
#define _aer_tensor_block_update_

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <vector>

#include "indexing.hpp"

namespace AER {
namespace Tensor {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using dynamic_reg_t = std::vector<uint_t>;
template <size_t N> using static_reg_t = std::array<uint_t, N>;

class BlockUpdater {
public:
  //-----------------------------------------------------------------------
  // Constructor
  //-----------------------------------------------------------------------

  BlockUpdater() = default;
  BlockUpdater(int nthreads, int nq_threshold)
      : omp_threads_(nthreads), omp_nq_threshold_(nq_threshold){};

  //-----------------------------------------------------------------------
  // Statevector update with Lambda function
  //-----------------------------------------------------------------------
  // Apply a lambda function to all entries of the statevector.
  // The function signature should be:
  //
  // [&](const int_t k)->void
  //
  // where k is the index of the vector
  template <typename T, typename Lambda>
  void apply_lambda(T &data, const size_t nq, Lambda &&func) const;

  //-----------------------------------------------------------------------
  // Statevector block update with Lambda function
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a lambda function to each block specified by the qubits argument.
  //
  // NOTE: The lambda functions can use the dynamic or static indexes
  // signature however if N is known at compile time the static case should
  // be preferred as it is significantly faster.

  // Apply a N-qubit lambda function to all blocks of the statevector
  // for the given qubits. The function signature should be either:
  //
  // (Static): [&](const static_reg_t<1ULL<<N> &inds)->void
  // (Dynamic): [&](const dynamic_reg_t &inds)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function.
  template <typename T, typename Lambda, typename R>
  void apply_lambda(T &data, const size_t nq, Lambda &&func,
                    const R &func_qubits) const;

  // Apply an N-qubit parameterized lambda function to all blocks of the
  // statevector for the given qubits. The function signature should be:
  //
  // (Static): [&](const static_reg_t<1ULL<<N> &inds, const param_t
  // &params)->void (Dynamic): [&](const dynamic_reg_t &inds, const param_t
  // &params)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function and `param` is a templated parameter class.
  // (typically a complex vector).
  template <typename T, typename Lambda, typename R, typename P>
  void apply_lambda(T &data, const size_t nq, Lambda &&func,
                    const R &func_qubits, const P &params) const;

  //-----------------------------------------------------------------------
  // Controlled Lambdas
  //-----------------------------------------------------------------------
  template <typename T, typename Lambda, typename R1, typename R2>
  void apply_controlled_lambda(T &data, const size_t nq, Lambda &&func,
                               const R1 &func_target_qubits,
                               const R2 &func_control_qubits) const;

  template <typename T, typename Lambda, typename R1, typename R2,
            typename param_t>
  void apply_controlled_lambda(T &data, const size_t nq, Lambda &&func,
                               const R1 &func_target_qubits,
                               const R2 &func_control_qubits,
                               const param_t &params) const;

  //-----------------------------------------------------------------------
  // State reduction with Lambda functions
  //-----------------------------------------------------------------------
  // Apply a complex reduction lambda function to all entries of the
  // statevector and return the complex result.
  // The function signature should be:
  //
  // [&](const int_t k, double &val_re, double &val_im)->void
  //
  // where k is the index of the vector, val_re and val_im are the doubles
  // to store the reduction.
  // Returns std::complex<double>(val_re, val_im)
  template <typename T, typename Lambda>
  std::complex<double> apply_reduction_lambda(T &data, const size_t nq,
                                              Lambda &&func) const;

  //-----------------------------------------------------------------------
  // Statevector block reduction with Lambda function
  //-----------------------------------------------------------------------
  // These functions loop through the indexes of the qubitvector data and
  // apply a reduction lambda function to each block specified by the qubits
  // argument. The reduction lambda stores the reduction in two doubles
  // (val_re, val_im) and returns the complex result
  // std::complex<double>(val_re, val_im)
  //
  // NOTE: The lambda functions can use the dynamic or static indexes
  // signature however if N is known at compile time the static case should
  // be preferred as it is significantly faster.

  // Apply a N-qubit complex matrix reduction lambda function to all blocks
  // of the statevector for the given qubits.
  // The lambda function signature should be:
  //
  // (Static): [&](const areg_t<1ULL<<N> &inds, const param_t &mat,
  //               double &val_re, double &val_im)->void
  // (Dynamic): [&](const indexes_t &inds, const param_t &mat,
  //                double &val_re, double &val_im)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function, `val_re` and `val_im` are the doubles to
  // store the reduction returned as std::complex<double>(val_re, val_im).
  template <typename T, typename Lambda, typename R>
  std::complex<double> apply_reduction_lambda(T &data, const size_t nq,
                                              Lambda &&func,
                                              const R &func_qubits) const;

  // Apply a N-qubit complex matrix reduction lambda function to all blocks
  // of the statevector for the given qubits.
  // The lambda function signature should be:
  //
  // (Static): [&](const areg_t<1ULL<<N> &inds, const param_t &parms,
  //               double &val_re, double &val_im)->void
  // (Dynamic): [&](const indexes_t &inds, const param_t &params,
  //                double &val_re, double &val_im)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexe`s function, `params` is a templated parameter class
  // (typically a complex vector), `val_re` and `val_im` are the doubles to
  // store the reduction returned as std::complex<double>(val_re, val_im).
  template <typename T, typename Lambda, typename R, typename P>
  std::complex<double>
  apply_reduction_lambda(T &data, const size_t nq, Lambda &&func,
                         const R &func_qubits, const P &params) const;

protected:
  //-----------------------------------------------------------------------
  // OpenMP Settings
  //-----------------------------------------------------------------------

  // Number of threads to use for OpenMP update
  int omp_threads_ = 1;

  // Qubit threshold to activate OpenMP parallelization
  int omp_nq_threshold_ = 14;
};

/*******************************************************************************
 *
 * LAMBDA FUNCTION TEMPLATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// State update
//------------------------------------------------------------------------------

template <typename T, typename Lambda>
void BlockUpdater::apply_lambda(T &data, const size_t nq, Lambda &&func) const {
  const int_t END = 1LL << nq;
#pragma omp parallel if (nq > omp_nq_threshold_ && omp_threads_ > 1)           \
    num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      std::forward<Lambda>(func)(data, k);
    }
  }
}

template <typename T, typename Lambda, typename R>
void BlockUpdater::apply_lambda(T &data, const size_t nq, Lambda &&func,
                                const R &func_qubits) const {
  const int_t END = Indexing::BITS[nq] >> func_qubits.size();
  // Sort qubits
  auto qubits_sorted = func_qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
#pragma omp parallel if (nq > omp_nq_threshold_ && omp_threads_ > 1)           \
    num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      // store entries touched by U
      const auto inds = Indexing::indexes(func_qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(data, inds);
    }
  }
}

template <typename T, typename Lambda, typename R, typename P>
void BlockUpdater::apply_lambda(T &data, const size_t nq, Lambda &&func,
                                const R &func_qubits, const P &params) const {
  const int_t END = Indexing::BITS[nq] >> func_qubits.size();
  auto qubits_sorted = func_qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

#pragma omp parallel if (nq > omp_nq_threshold_ && omp_threads_ > 1)           \
    num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = Indexing::indexes(func_qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(data, inds, params);
    }
  }
}

//------------------------------------------------------------------------------
// Control-blocked Lambda
//------------------------------------------------------------------------------

template <typename T, typename Lambda, typename R1, typename R2>
void BlockUpdater::apply_controlled_lambda(
    T &data, const size_t nq, Lambda &&func, const R1 &func_target_qubits,
    const R2 &func_control_qubits) const {
  // TODO: We should be able to have this function call the parameterized
  // apply_controlled_lambda using nullptr as param template type
  // but we would need to "pad" the Lambda func to take the second nullptr
  // argument in a way that doesn't effect performance in inner OpenMP loop

  // Check if control_qubits are empty, if so apply non-controlled lambda
  if (func_control_qubits.empty()) {
    apply_lambda(data, nq, func, func_target_qubits);
    return;
  }

  // Get sorted list of control + target qubits
  const auto N_TARG = func_target_qubits.size();
  const auto N_CTRL = func_control_qubits.size();
  dynamic_reg_t qubits_sorted;
  qubits_sorted.reserve(N_TARG + N_CTRL);
  qubits_sorted.insert(qubits_sorted.begin(), func_control_qubits.begin(),
                       func_control_qubits.end());
  qubits_sorted.insert(qubits_sorted.begin(), func_target_qubits.begin(),
                       func_target_qubits.end());
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Get look size and block size
  const int_t END = Indexing::BITS[nq] >> (N_TARG + N_CTRL);
  const int_t CONTROL_VAL = Indexing::MASKS[N_CTRL];
#pragma omp parallel if (nq > omp_nq_threshold_ && omp_threads_ > 1)           \
    num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds =
          controlled_indexes(func_target_qubits, func_control_qubits,
                             qubits_sorted, CONTROL_VAL, k);
      std::forward<Lambda>(func)(data, inds);
    }
  }
}

template <typename T, typename Lambda, typename R1, typename R2, typename P>
void BlockUpdater::apply_controlled_lambda(T &data, const size_t nq,
                                           Lambda &&func,
                                           const R1 &func_target_qubits,
                                           const R2 &func_control_qubits,
                                           const P &params) const {
  // Check if control_qubits are empty, if so apply non-controlled lambda
  if (func_control_qubits.empty()) {
    apply_lambda(data, nq, func, func_target_qubits);
    return;
  }

  // Get sorted list of control + target qubits
  const auto N_TARG = func_target_qubits.size();
  const auto N_CTRL = func_control_qubits.size();
  dynamic_reg_t qubits_sorted;
  qubits_sorted.reserve(N_TARG + N_CTRL);
  qubits_sorted.insert(qubits_sorted.begin(), func_control_qubits.begin(),
                       func_control_qubits.end());
  qubits_sorted.insert(qubits_sorted.begin(), func_target_qubits.begin(),
                       func_target_qubits.end());
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Get look size and block size
  const int_t END = Indexing::BITS[nq] >> (N_TARG + N_CTRL);
  const int_t CONTROL_VAL = Indexing::MASKS[N_CTRL];
#pragma omp parallel if (nq > omp_nq_threshold_ && omp_threads_ > 1)           \
    num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds =
          controlled_indexes(func_target_qubits, func_control_qubits,
                             qubits_sorted, CONTROL_VAL, k);
      std::forward<Lambda>(func)(inds, params);
    }
  }
}

//------------------------------------------------------------------------------
// Reduction Lambda
//------------------------------------------------------------------------------

template <typename T, typename Lambda>
std::complex<double> BlockUpdater::apply_reduction_lambda(T &data,
                                                          const size_t nq,
                                                          Lambda &&func) const {
  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
  const int_t END = Indexing::BITS[nq];
#pragma omp parallel reduction(+:val_re, val_im) if (nq > omp_nq_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      std::forward<Lambda>(func)(data, k, val_re, val_im);
    }
  } // end omp parallel
  return std::complex<double>(val_re, val_im);
}

template <typename T, typename Lambda, typename R>
std::complex<double>
BlockUpdater::apply_reduction_lambda(T &data, const size_t nq, Lambda &&func,
                                     const R &func_qubits) const {

  const int_t END = Indexing::BITS[nq] >> func_qubits.size();
  auto qubits_sorted = func_qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (nq > omp_nq_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = indexes(func_qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(data, inds, val_re, val_im);
    }
  } // end omp parallel
  return std::complex<double>(val_re, val_im);
}

template <typename T, typename Lambda, typename R, typename P>
std::complex<double>
BlockUpdater::apply_reduction_lambda(T &data, const size_t nq, Lambda &&func,
                                     const R &func_qubits,
                                     const P &params) const {

  const int_t END = Indexing::BITS[nq] >> func_qubits.size();
  auto qubits_sorted = func_qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  double val_re = 0.;
  double val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (nq > omp_nq_threshold_ && omp_threads_ > 1)         \
                                               num_threads(omp_threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = indexes(func_qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(data, inds, params, val_re, val_im);
    }
  } // end omp parallel
  return std::complex<double>(val_re, val_im);
}

//------------------------------------------------------------------------------
} // namespace Tensor
//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif // end module
