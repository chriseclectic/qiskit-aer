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

#ifndef _kernel_omp_hpp_
#define _kernel_omp_hpp_

#include <complex>
#include <cstdint>

#include "simulators/statevector/indexes.hpp"

namespace Kernel {

//============================================================================
// OMP Kernel Class
//============================================================================
class OMPKernel {
public:
  // Type aliases
  using uint_t = Index::uint_t;
  using int_t = Index::int_t;

  template <typename Float> using cvector_t = std::vector<std::complex<Float>>;

  template <typename Float>
  using enable_if_float_t =
      std::enable_if_t<std::is_floating_point<Float>::value>;

  //-----------------------------------------------------------------------
  // General Matrix Multiplication
  //-----------------------------------------------------------------------

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit
  // matrix.
  template <typename Data = double, typename Container, typename T>
  void apply_matrix(Container &data, size_t sz, const cvector_t<T> &colmat,
                    const Index::reg_t qubits) const;

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit
  // matrix.
  template <typename Data = double, typename Container, typename T>
  void apply_matrix(Container &data, size_t sz, const cvector_t<T> &colmat,
                    const size_t qubit) const;

  //-----------------------------------------------------------------------
  // Diagonal Matrix Multiplication
  //-----------------------------------------------------------------------

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  template <typename Data = double, typename Container, typename T>
  void apply_diagonal_matrix(Container &data, size_t sz,
                             const cvector_t<T> &diag,
                             const Index::reg_t qubits) const;

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  template <typename Data = double, typename Container, typename T>
  void apply_diagonal_matrix(Container &data, size_t sz,
                             const cvector_t<T> &diag,
                             const size_t qubit) const;

  //-----------------------------------------------------------------------
  // Permutation Matrix Multiplication
  //-----------------------------------------------------------------------

  // Swap pairs of indicies in the underlying vector
  template <typename Data = double, typename Container>
  void
  apply_permutation_matrix(Container &data, size_t sz,
                           const std::vector<std::pair<uint_t, uint_t>> &pairs,
                           const Index::reg_t &qubits);

  //-----------------------------------------------------------------------
  // Statevector update with Lambda function
  //-----------------------------------------------------------------------
  // Apply a lambda function to all entries of the statevector.
  // The function signature should be:
  //
  // [&](const int_t k)->void
  //
  // where k is the index of the vector
  template <typename Lambda> void apply_lambda(size_t sz, Lambda &&func) const;

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
  // (Static): [&](const Index::areg_t<1ULL<<N> &_inds)->void
  // (Dynamic): [&](const Index::indexes_t &_inds)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function.
  template <typename Lambda, typename list_t>
  void apply_lambda(size_t sz, Lambda &&func, const list_t &qubits) const;

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
  template <typename Float, typename Lambda,
            typename = enable_if_float_t<Float>>
  void apply_reduction_lambda(size_t sz, std::complex<Float> &accum,
                              Lambda &&func) const;

  // As above but apply reduction on real value instead of complex
  template <typename Float, typename Lambda,
            typename = enable_if_float_t<Float>>
  void apply_reduction_lambda(size_t sz, Float &accum, Lambda &&func) const;

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
  // (Static): [&](const Index::areg_t<1ULL<<N> &_inds, const param_t &mat,
  //               double &val_re, double &val_im)->void
  // (Dynamic): [&](const Index::indexes_t &_inds, const param_t &mat,
  //                double &val_re, double &val_im)->void
  //
  // where `inds` are the 2 ** N indexes for each N-qubit block returned by
  // the `indexes` function, `val_re` and `val_im` are the doubles to
  // store the reduction returned as std::complex<double>(val_re, val_im).
  template <typename Float, typename Lambda, typename list_t,
            typename = enable_if_float_t<Float>>
  void apply_reduction_lambda(size_t sz, std::complex<Float> &accum,
                              Lambda &&func, const list_t &qubits) const;

  // As above but apply reduction on a real value instead of complex
  template <typename Float, typename Lambda, typename list_t,
            typename = enable_if_float_t<Float>>
  void apply_reduction_lambda(size_t sz, Float &accum, Lambda &&func,
                              const list_t &qubits) const;

  //-----------------------------------------------------------------------
  // OpenMP configuration settings
  //-----------------------------------------------------------------------

  // Set the maximum number of OpenMP thread for operations.
  void set_threads(int n) {
    if (n > 0)
      threads_ = n;
  }

  // Get the maximum number of OpenMP thread for operations.
  int get_threads() { return threads_; }

  // Set the size threshold for activating OpenMP.
  void set_threshold(size_t n) { threshold_ = n; }

  // Get the qubit threshold for activating OpenMP.
  size_t get_threshold() { return threshold_; }

  // Mixed precision complex multiplication
  template <typename T1, typename T2>
  std::complex<T1> cmul(const std::complex<T1> &a,
                        const std::complex<T2> &b) const {
    return a * std::complex<T1>(b);
  }
  template <typename T>
  std::complex<T> cmul(const std::complex<T> &a,
                       const std::complex<T> &b) const {
    return a * b;
  }

protected:
  // Size threshold for multithreading
  size_t threshold_ = 1ULL << 14;

  // Disable multithreading by default
  int threads_ = 1;

};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Container update
//------------------------------------------------------------------------------

template <typename Lambda>
void OMPKernel::apply_lambda(size_t sz, Lambda &&func) const {
  const int_t END = sz;
#pragma omp parallel if (sz >= threshold_ && threads_ > 1) num_threads(threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      std::forward<Lambda>(func)(k);
    }
  }
}

template <typename Lambda, typename list_t>
void OMPKernel::apply_lambda(size_t sz, Lambda &&func,
                             const list_t &qubits) const {
  const int_t END = sz >> qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
#pragma omp parallel if (sz >= threshold_ && threads_ > 1) num_threads(threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      // store entries touched by U
      const auto inds = Index::indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds);
    }
  }
}

//------------------------------------------------------------------------------
// Container complex reduction
//------------------------------------------------------------------------------

template <typename Float, typename Lambda, typename>
void OMPKernel::apply_reduction_lambda(size_t sz, std::complex<Float> &accum,
                                       Lambda &&func) const {
  // Reduction variables
  Float val_re = 0.;
  Float val_im = 0.;

  const int_t END = sz;
#pragma omp parallel reduction(+:val_re, val_im) if (sz >= threshold_ && threads_ > 1) num_threads(threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      std::forward<Lambda>(func)(k, val_re, val_im);
    }
  } // end omp parallel

  // Update accumulator
  accum.real(accum.real() + val_re);
  accum.imag(accum.imag() + val_im);
}

template <typename Float, typename Lambda, typename list_t, typename>
void OMPKernel::apply_reduction_lambda(size_t sz, std::complex<Float> &accum,
                                       Lambda &&func,
                                       const list_t &qubits) const {

  const int_t END = sz >> qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  Float val_re = 0.;
  Float val_im = 0.;
#pragma omp parallel reduction(+:val_re, val_im) if (sz >= threshold_ && threads_ > 1) num_threads(threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = Index::indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, val_re, val_im);
    }
  } // end omp parallel

  // Update accumulator
  accum.real(accum.real() + val_re);
  accum.imag(accum.imag() + val_im);
}

//------------------------------------------------------------------------------
// Container real reduction
//------------------------------------------------------------------------------

template <typename Float, typename Lambda, typename>
void OMPKernel::apply_reduction_lambda(size_t sz, Float &accum,
                                       Lambda &&func) const {
  // Reduction variables
  Float val = 0.;

  const int_t END = sz;
#pragma omp parallel reduction(+:val) if (sz >= threshold_ && threads_ > 1) num_threads(threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      std::forward<Lambda>(func)(k, val);
    }
  } // end omp parallel

  // Update accumulator
  accum += val;
}

template <typename Float, typename Lambda, typename list_t, typename>
void OMPKernel::apply_reduction_lambda(size_t sz, Float &accum, Lambda &&func,
                                       const list_t &qubits) const {

  const int_t END = sz >> qubits.size();
  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());

  // Reduction variables
  Float val = 0.;
#pragma omp parallel reduction(+:val) if (sz >= threshold_ && threads_ > 1) num_threads(threads_)
  {
#pragma omp for
    for (int_t k = 0; k < END; k++) {
      const auto inds = Index::indexes(qubits, qubits_sorted, k);
      std::forward<Lambda>(func)(inds, val);
    }
  } // end omp parallel

  // Update accumulator
  accum += val;
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// N-qubit matrix multiplication
//------------------------------------------------------------------------------

template <typename Data, typename Container, typename T>
void OMPKernel::apply_matrix(Container &data, size_t sz,
                             const cvector_t<T> &colmat,
                             const Index::reg_t qubits) const {

  // Static array optimized lambda functions
  const size_t N = qubits.size();

  switch (N) {
  case 1:
    apply_matrix<Data>(data, sz, colmat, qubits[0]);
    return;
  case 2: {
    // Lambda function for 2-qubit matrix multiplication
    auto lambda = [&](const Index::areg_t<4> &_inds) -> void {
      std::array<std::complex<Data>, 4> cache;
      for (size_t i = 0; i < 4; i++) {
        const auto ii = _inds[i];
        cache[i] = data[ii];
        data[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++)
          data[_inds[i]] += cmul(cache[j], colmat[i + 4 * j]);
    };
    apply_lambda(sz, lambda, Index::areg_t<2>({{qubits[0], qubits[1]}}));
    return;
  }
  case 3: {
    // Lambda function for 3-qubit matrix multiplication
    auto lambda = [&](const Index::areg_t<8> &_inds) -> void {
      std::array<std::complex<Data>, 8> cache;
      for (size_t i = 0; i < 8; i++) {
        const auto ii = _inds[i];
        cache[i] = data[ii];
        data[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < 8; i++)
        for (size_t j = 0; j < 8; j++)
          data[_inds[i]] += cmul(cache[j], colmat[i + 8 * j]);
    };
    apply_lambda(sz, lambda,
                 Index::areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
    return;
  }
  case 4: {
    // Lambda function for 4-qubit matrix multiplication
    auto lambda = [&](const Index::areg_t<16> &_inds) -> void {
      std::array<std::complex<Data>, 16> cache;
      for (size_t i = 0; i < 16; i++) {
        const auto ii = _inds[i];
        cache[i] = data[ii];
        data[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < 16; i++)
        for (size_t j = 0; j < 16; j++)
          data[_inds[i]] += cmul(cache[j], colmat[i + 16 * j]);
    };
    apply_lambda(
        sz, lambda,
        Index::areg_t<4>({{qubits[0], qubits[1], qubits[2], qubits[3]}}));
    return;
  }
  default: {
    // Lambda function for N-qubit matrix multiplication
    auto lambda = [&](const Index::indexes_t &_inds) -> void {
      const uint_t DIM = Index::BITS[N];
      auto cache = std::make_unique<std::complex<Data>[]>(DIM);
      for (size_t i = 0; i < DIM; i++) {
        const auto ii = _inds[i];
        cache[i] = data[ii];
        data[ii] = 0.;
      }
      // update state vector
      for (size_t i = 0; i < DIM; i++)
        for (size_t j = 0; j < DIM; j++)
          data[_inds[i]] += cmul(cache[j], colmat[i + DIM * j]);
    };
    apply_lambda(sz, lambda, qubits);
  }
  } // end switch
}
//------------------------------------------------------------------------------
// N-qubit diagonal matrix multiplication
//------------------------------------------------------------------------------

template <typename Data, typename Container, typename T>
void OMPKernel::apply_diagonal_matrix(Container &data, size_t sz,
                                      const cvector_t<T> &diag,
                                      const Index::reg_t qubits) const {

  if (qubits.size() == 1) {
    apply_diagonal_matrix<Data>(data, sz, diag, qubits[0]);
    return;
  }

  auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
    for (int_t i = 0; i < 2; ++i) {
      const int_t k = _inds[i];
      int_t iv = 0;
      for (int_t j = 0; j < qubits.size(); j++)
        if ((k & (Index::BITS[qubits[j]])) != 0)
          iv += Index::BITS[j];
      if (diag[iv] != (T)1.0)
        data[k] = std::complex<Data>(diag[iv]);
    }
  };
  apply_lambda(sz, lambda, Index::areg_t<1>({{qubits[0]}}));
}

//------------------------------------------------------------------------------
// 1-qubit matrix multiplication
//------------------------------------------------------------------------------

template <typename Data, typename Container, typename T>
void OMPKernel::apply_matrix(Container &data, size_t sz,
                             const cvector_t<T> &colmat,
                             const size_t qubit) const {

  // Check if matrix is diagonal and if so use optimized lambda
  if (colmat[1] == 0.0 && colmat[2] == 0.0) {
    const cvector_t<Data> diag = {
        {std::complex<Data>(colmat[0]), std::complex<Data>(colmat[3])}};
    apply_diagonal_matrix<Data>(data, sz, diag, qubit);
    return;
  }

  // Convert qubit to array register for lambda functions
  Index::areg_t<1> qubits = {{qubit}};

  // Check if anti-diagonal matrix and if so use optimized lambda
  if (colmat[0] == 0.0 && colmat[3] == 0.0) {
    if (colmat[1] == 1.0 && colmat[2] == 1.0) {
      // X-matrix
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        std::swap(data[_inds[0]], data[_inds[1]]);
      };
      apply_lambda(sz, lambda, qubits);
      return;
    }
    if (colmat[2] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        data[_inds[1]] = std::complex<Data>(colmat[1]) * data[_inds[0]];
        data[_inds[0]] = 0.0;
      };
      apply_lambda(sz, lambda, qubits);
      return;
    }
    if (colmat[1] == 0.0) {
      // Non-unitary projector
      // possibly used in measure/reset/kraus update
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        data[_inds[0]] = std::complex<Data>(colmat[2]) * data[_inds[1]];
        data[_inds[1]] = 0.0;
      };
      apply_lambda(sz, lambda, qubits);
      return;
    }
    // else we have a general anti-diagonal matrix
    auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
      const auto cache = data[_inds[0]];
      data[_inds[0]] = std::complex<Data>(colmat[2]) * data[_inds[1]];
      data[_inds[1]] = std::complex<Data>(colmat[1]) * cache;
    };
    apply_lambda(sz, lambda, qubits);
    return;
  }
  // Otherwise general single-qubit matrix multiplication
  auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
    const auto cache = data[_inds[0]];
    data[_inds[0]] = std::complex<Data>(colmat[0]) * cache +
                     std::complex<Data>(colmat[2]) * data[_inds[1]];
    data[_inds[1]] = std::complex<Data>(colmat[1]) * cache +
                     std::complex<Data>(colmat[3]) * data[_inds[1]];
  };
  apply_lambda(sz, lambda, qubits);
}

//------------------------------------------------------------------------------
// 1-qubit diagonal matrix multiplication
//------------------------------------------------------------------------------

template <typename Data, typename Container, typename T>
void OMPKernel::apply_diagonal_matrix(Container &data, size_t sz,
                                      const cvector_t<T> &diag,
                                      const size_t qubit) const {

  // TODO: This should be changed so it isn't checking doubles with ==
  if (diag[0] == T(1.0)) { // [[1, 0], [0, z]] matrix
    if (diag[1] == T(1.0))
      return; // Identity

    if (diag[1] == std::complex<T>(0., -1.)) { // [[1, 0], [0, -i]]
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        const auto k = _inds[1];
        const auto cache = data[k].imag();
        data[k].imag(data[k].real() * Data(-1.));
        data[k].real(cache);
      };
      apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
      return;
    }
    if (diag[1] == std::complex<T>(0., 1.)) {
      // [[1, 0], [0, i]]
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        const auto k = _inds[1];
        const auto cache = data[k].imag();
        data[k].imag(data[k].real());
        data[k].real(cache * Data(-1.));
      };
      apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
      return;
    }
    if (diag[0] == T(0.0)) {
      // [[1, 0], [0, 0]]
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        data[_inds[1]] = Data(0.0);
      };
      apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
      return;
    }
    // general [[1, 0], [0, z]]
    auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
      const auto k = _inds[1];
      data[k] *= std::complex<Data>(diag[1]);
    };
    apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
    return;
  } else if (diag[1] == T(1.0)) {
    // [[z, 0], [0, 1]] matrix
    if (diag[0] == std::complex<T>(0., Data(-1.))) {
      // [[-i, 0], [0, 1]]
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        const auto k = _inds[1];
        const auto cache = data[k].imag();
        data[k].imag(data[k].real() * Data(-1.));
        data[k].real(cache);
      };
      apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
      return;
    }
    if (diag[0] == std::complex<T>(0., 1.)) {
      // [[i, 0], [0, 1]]
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        const auto k = _inds[1];
        const auto cache = data[k].imag();
        data[k].imag(data[k].real());
        data[k].real(cache * Data(-1.));
      };
      apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
      return;
    }
    if (diag[0] == T(0.0)) {
      // [[0, 0], [0, 1]]
      auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
        data[_inds[0]] = Data(0.0);
      };
      apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
      return;
    }
    // general [[z, 0], [0, 1]]

    auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
      data[_inds[0]] *= std::complex<Data>(diag[0]);
    };
    apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
    return;
  } else {
    // Lambda function for diagonal matrix multiplication
    auto lambda = [&](const Index::areg_t<2> &_inds) -> void {
      data[_inds[0]] *= std::complex<double>(diag[0]);
      data[_inds[1]] *= std::complex<double>(diag[1]);
    };
    apply_lambda(sz, lambda, Index::areg_t<1>({{qubit}}));
  }
}

//-----------------------------------------------------------------------
// Permutation Matrix Multiplication
//-----------------------------------------------------------------------

template <typename Data, typename Container>
void OMPKernel::apply_permutation_matrix(
    Container &data, size_t sz,
    const std::vector<std::pair<uint_t, uint_t>> &pairs,
    const Index::reg_t &qubits) {
  const size_t N = qubits.size();
  switch (N) {
  case 1: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::areg_t<2> &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    apply_lambda(sz, lambda, Index::areg_t<1>({{qubits[0]}}));
    return;
  }
  case 2: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::areg_t<4> &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    apply_lambda(sz, lambda, Index::areg_t<2>({{qubits[0], qubits[1]}}));
    return;
  }
  case 3: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::areg_t<8> &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    apply_lambda(sz, lambda,
                 Index::areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
    return;
  }
  case 4: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::areg_t<16> &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    apply_lambda(
        sz, lambda,
        Index::areg_t<4>({{qubits[0], qubits[1], qubits[2], qubits[3]}}));
    return;
  }
  case 5: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::areg_t<32> &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    apply_lambda(sz, lambda,
                 Index::areg_t<5>({{qubits[0], qubits[1], qubits[2], qubits[3],
                                    qubits[4]}}));
    return;
  }
  case 6: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::areg_t<64> &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    apply_lambda(sz, lambda,
                 Index::areg_t<6>({{qubits[0], qubits[1], qubits[2], qubits[3],
                                    qubits[4], qubits[5]}}));
    return;
  }
  default: {
    // Lambda function for permutation matrix
    auto lambda = [&](const Index::indexes_t &inds) -> void {
      for (const auto &p : pairs) {
        std::swap(data[inds[p.first]], data[inds[p.second]]);
      }
    };
    // Use the lambda function
    apply_lambda(sz, lambda, qubits);
  }
  } // end switch
}

//------------------------------------------------------------------------------
} // namespace Kernel
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
#endif // end module
