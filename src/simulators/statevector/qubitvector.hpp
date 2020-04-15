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



#ifndef _qv_qubit_vector_hpp_
#define _qv_qubit_vector_hpp_

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

#include "framework/json.hpp"
#include "simulators/statevector/omp_kernel.hpp"

namespace QV {

// Type aliases
using uint_t = uint64_t;
using int_t = int64_t;
using reg_t = std::vector<uint_t>;
using indexes_t = std::unique_ptr<uint_t[]>;
template <size_t N> using areg_t = std::array<uint_t, N>;
template <typename T> using cvector_t = std::vector<std::complex<T>>;

//============================================================================
// QubitVector class
//============================================================================

// Template class for qubit vector.
// The arguement of the template must have an operator[] access method.
// The following methods may also need to be template specialized:
//   * set_num_qubits(size_t)
//   * initialize()
//   * initialize_from_vector(cvector_t<data_t>)
// If the template argument does not have these methods then template
// specialization must be used to override the default implementations.

template <typename data_t = double>
class QubitVector {

public:

  //-----------------------------------------------------------------------
  // Constructors and Destructor
  //-----------------------------------------------------------------------

  QubitVector();
  explicit QubitVector(size_t num_qubits);
  virtual ~QubitVector();
  QubitVector(const QubitVector& obj) = delete;
  QubitVector &operator=(const QubitVector& obj) = delete;

  //-----------------------------------------------------------------------
  // Data access
  //-----------------------------------------------------------------------

  // Element access
  std::complex<data_t> &operator[](uint_t element);
  std::complex<data_t> operator[](uint_t element) const;

  // Returns a reference to the underlying data_t data class
  std::complex<data_t>* &data() {return data_;}

  // Returns a copy of the underlying data_t data class
  std::complex<data_t>* data() const {return data_;}

  //-----------------------------------------------------------------------
  // Utility functions
  //-----------------------------------------------------------------------

  // Return the string name of the QUbitVector class
  static std::string name() {return "statevector";}

  // Set the size of the vector in terms of qubit number
  void set_num_qubits(size_t num_qubits);

  // Returns the number of qubits for the current vector
  virtual uint_t num_qubits() const {return num_qubits_;}

  // Returns the size of the underlying n-qubit vector
  uint_t size() const {return data_size_;}

  // Returns required memory
  size_t required_memory_mb(uint_t num_qubits) const;

  // Returns a copy of the underlying data_t data as a complex vector
  cvector_t<data_t> vector() const;

  // Return JSON serialization of QubitVector;
  json_t json() const;

  // Set all entries in the vector to 0.
  void zero();

  // convert vector type to data type of this qubit vector
  cvector_t<data_t> convert(const cvector_t<double>& v) const;

  // State initialization of a component
  // Initialize the specified qubits to a desired statevector
  // (leaving the other qubits in their current state)
  // assuming the qubits being initialized have already been reset to the zero state
  // (using apply_reset)
  void initialize_component(const Index::reg_t &qubits, const cvector_t<double> &state);

  //-----------------------------------------------------------------------
  // Check point operations
  //-----------------------------------------------------------------------

  // Create a checkpoint of the current state
  void checkpoint();

  // Revert to the checkpoint
  void revert(bool keep);

  // Compute the inner product of current state with checkpoint state
  std::complex<double> inner_product() const;

  //-----------------------------------------------------------------------
  // Initialization
  //-----------------------------------------------------------------------

  // Initializes the current vector so that all qubits are in the |0> state.
  void initialize();

  // Initializes the vector to a custom initial state.
  // If the length of the data vector does not match the number of qubits
  // an exception is raised.
  void initialize_from_vector(const cvector_t<double> &data);

  // Initializes the vector to a custom initial state.
  // If num_states does not match the number of qubits an exception is raised.
  void initialize_from_data(const std::complex<data_t>* data, const size_t num_states);

  //-----------------------------------------------------------------------
  // Apply Matrices
  //-----------------------------------------------------------------------

  // Apply a 1-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  void apply_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_matrix(const Index::reg_t &qubits, const cvector_t<double> &mat);

  // Apply a stacked set of 2^control_count target_count--qubit matrix to the state vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  void apply_multiplexer(const Index::reg_t &control_qubits, const Index::reg_t &target_qubits, const cvector_t<double> &mat);

  // Apply a 1-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const uint_t qubit, const cvector_t<double> &mat);

  // Apply a N-qubit diagonal matrix to the state vector.
  // The matrix is input as vector of the matrix diagonal.
  void apply_diagonal_matrix(const Index::reg_t &qubits, const cvector_t<double> &mat);
  
  // Swap pairs of indicies in the underlying vector
  void apply_permutation_matrix(const Index::reg_t &qubits,
                                const std::vector<std::pair<uint_t, uint_t>> &pairs);

  //-----------------------------------------------------------------------
  // Apply Specialized Gates
  //-----------------------------------------------------------------------

  // Apply a general N-qubit multi-controlled X-gate
  // If N=1 this implements an optimized X gate
  // If N=2 this implements an optimized CX gate
  // If N=3 this implements an optimized Toffoli gate
  void apply_mcx(const Index::reg_t &qubits);

  // Apply a general multi-controlled Y-gate
  // If N=1 this implements an optimized Y gate
  // If N=2 this implements an optimized CY gate
  // If N=3 this implements an optimized CCY gate
  void apply_mcy(const Index::reg_t &qubits);
  
  // Apply a general multi-controlled single-qubit phase gate
  // with diagonal [1, ..., 1, phase]
  // If N=1 this implements an optimized single-qubit phase gate
  // If N=2 this implements an optimized CPhase gate
  // If N=3 this implements an optimized CCPhase gate
  // if phase = -1 this is a Z, CZ, CCZ gate
  void apply_mcphase(const Index::reg_t &qubits, const std::complex<double> phase);

  // Apply a general multi-controlled single-qubit unitary gate
  // If N=1 this implements an optimized single-qubit U gate
  // If N=2 this implements an optimized CU gate
  // If N=3 this implements an optimized CCU gate
  void apply_mcu(const Index::reg_t &qubits, const cvector_t<double> &mat);

  // Apply a general multi-controlled SWAP gate
  // If N=2 this implements an optimized SWAP  gate
  // If N=3 this implements an optimized Fredkin gate
  void apply_mcswap(const Index::reg_t &qubits);

  //-----------------------------------------------------------------------
  // Z-measurement outcome probabilities
  //-----------------------------------------------------------------------

  // Return the Z-basis measurement outcome probability P(outcome) for
  // outcome in [0, 2^num_qubits - 1]
  virtual double probability(const uint_t outcome) const;

  // Return the probabilities for all measurement outcomes in the current vector
  // This is equivalent to returning a new vector with  new[i]=|orig[i]|^2.
  // Eg. For 2-qubits this is [P(00), P(01), P(010), P(11)]
  virtual std::vector<double> probabilities() const;

  // Return the Z-basis measurement outcome probabilities [P(0), ..., P(2^N-1)]
  // for measurement of N-qubits.
  virtual std::vector<double> probabilities(const Index::reg_t &qubits) const;

  // Return M sampled outcomes for Z-basis measurement of all qubits
  // The input is a length M list of random reals between [0, 1) used for
  // generating samples.
  virtual reg_t sample_measure(const std::vector<double> &rnds) const;

  //-----------------------------------------------------------------------
  // Norms
  //-----------------------------------------------------------------------
  
  // Returns the norm of the current vector
  double norm() const;

  // These functions return the norm <psi|A^dagger.A|psi> obtained by
  // applying a matrix A to the vector. It is equivalent to returning the
  // expectation value of A^\dagger A, and could probably be removed because
  // of this.

  // Return the norm for of the vector obtained after apply the 1-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized 1-qubit matrix.
  double norm(const uint_t qubit, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // matrix mat to the vector.
  // The matrix is input as vector of the column-major vectorized N-qubit matrix.
  double norm(const Index::reg_t &qubits, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the 1-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const;

  // Return the norm for of the vector obtained after apply the N-qubit
  // diagonal matrix mat to the vector.
  // The matrix is input as vector of the matrix diagonal.
  double norm_diagonal(const Index::reg_t &qubits, const cvector_t<double> &mat) const;

  //-----------------------------------------------------------------------
  // JSON configuration settings
  //-----------------------------------------------------------------------

  // Set the threshold for chopping values to 0 in JSON
  void set_json_chop_threshold(double threshold);

  // Set the threshold for chopping values to 0 in JSON
  double get_json_chop_threshold() {return json_chop_threshold_;}

  //-----------------------------------------------------------------------
  // OpenMP configuration settings
  //-----------------------------------------------------------------------

  // Set the maximum number of OpenMP thread for operations.
  void set_omp_threads(int n);

  // Get the maximum number of OpenMP thread for operations.
  uint_t get_omp_threads() {return omp_threads_;}

  // Set the qubit threshold for activating OpenMP.
  // If self.qubits() > threshold OpenMP will be activated.
  void set_omp_threshold(int n);

  // Get the qubit threshold for activating OpenMP.
  uint_t get_omp_threshold() {return omp_threshold_;}

  //-----------------------------------------------------------------------
  // Optimization configuration settings
  //-----------------------------------------------------------------------

  // Set the sample_measure index size
  void set_sample_measure_index_size(int n) {sample_measure_index_size_ = n;}

  // Get the sample_measure index size
  int get_sample_measure_index_size() {return sample_measure_index_size_;}

protected:

  //-----------------------------------------------------------------------
  // Protected data members
  //-----------------------------------------------------------------------
  size_t num_qubits_;
  size_t data_size_;
  std::complex<data_t>* data_;
  std::complex<data_t>* checkpoint_;

  //-----------------------------------------------------------------------
  // OMP Kernel
  //-----------------------------------------------------------------------
  Kernel::OMPKernel kernel_;

  //-----------------------------------------------------------------------
  // Config settings
  //----------------------------------------------------------------------- 
  uint_t omp_threads_ = 1;     // Disable multithreading by default
  uint_t omp_threshold_ = 14;  // Qubit threshold for multithreading when enabled
  int sample_measure_index_size_ = 10; // Sample measure indexing qubit size
  double json_chop_threshold_ = 0;  // Threshold for choping small values
                                    // in JSON serialization

  //-----------------------------------------------------------------------
  // Error Messages
  //-----------------------------------------------------------------------

  void check_qubit(const uint_t qubit) const;
  void check_vector(const cvector_t<data_t> &diag, uint_t nqubits) const;
  void check_matrix(const cvector_t<data_t> &mat, uint_t nqubits) const;
  void check_dimension(const QubitVector &qv) const;
  void check_checkpoint() const;

};

/*******************************************************************************
 *
 * Implementations
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// JSON Serialization
//------------------------------------------------------------------------------

template <typename data_t>
inline void to_json(json_t &js, const QubitVector<data_t> &qv) {
  js = qv.json();
}

template <typename data_t>
json_t QubitVector<data_t>::json() const {
  const json_t ZERO = std::complex<data_t>(0.0, 0.0);
  json_t js = json_t(data_size_, ZERO);

  if (json_chop_threshold_ > 0) {
    auto lambda = [&](int_t j) {
      if (std::abs(data_[j].real()) > json_chop_threshold_)
        js[j][0] = data_[j].real();
      if (std::abs(data_[j].imag()) > json_chop_threshold_)
        js[j][1] = data_[j].imag();
    };
    kernel_.apply_lambda(data_size_, lambda);
  } else {
    auto lambda = [&](int_t j) {
      js[j][0] = data_[j].real();
      js[j][1] = data_[j].imag();
    };
    kernel_.apply_lambda(data_size_, lambda);
  }
  return js;
}

//------------------------------------------------------------------------------
// Error Handling
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::check_qubit(const uint_t qubit) const {
  if (qubit + 1 > num_qubits_) {
    std::string error = "QubitVector: qubit index " + std::to_string(qubit) +
                        " > " + std::to_string(num_qubits_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_matrix(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = Index::BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM * DIM) {
    std::string error = "QubitVector: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM * DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_vector(const cvector_t<data_t> &vec, uint_t nqubits) const {
  const size_t DIM = Index::BITS[nqubits];
  const auto SIZE = vec.size();
  if (SIZE != DIM) {
    std::string error = "QubitVector: vector size is " + std::to_string(SIZE) +
                        " != " + std::to_string(DIM);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_dimension(const QubitVector &qv) const {
  if (data_size_ != qv.size_) {
    std::string error = "QubitVector: vectors are different shape " +
                         std::to_string(data_size_) + " != " +
                         std::to_string(qv.num_states_);
    throw std::runtime_error(error);
  }
}

template <typename data_t>
void QubitVector<data_t>::check_checkpoint() const {
  if (!checkpoint_) {
    throw std::runtime_error("QubitVector: checkpoint must exist for inner_product() or revert()");
  }
}

//------------------------------------------------------------------------------
// Constructors & Destructor
//------------------------------------------------------------------------------

template <typename data_t>
QubitVector<data_t>::QubitVector(size_t num_qubits) : num_qubits_(0), data_(nullptr), checkpoint_(0){
  set_num_qubits(num_qubits);
}

template <typename data_t>
QubitVector<data_t>::QubitVector() : QubitVector(0) {}

template <typename data_t>
QubitVector<data_t>::~QubitVector() {
  if (data_)
    free(data_);

  if (checkpoint_)
    free(checkpoint_);
}

//------------------------------------------------------------------------------
// Element access operators
//------------------------------------------------------------------------------

template <typename data_t>
std::complex<data_t> &QubitVector<data_t>::operator[](uint_t element) {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVector: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
  return data_[element];
}

template <typename data_t>
std::complex<data_t> QubitVector<data_t>::operator[](uint_t element) const {
  // Error checking
  #ifdef DEBUG
  if (element > data_size_) {
    std::string error = "QubitVector: vector index " + std::to_string(element) +
                        " > " + std::to_string(data_size_);
    throw std::runtime_error(error);
  }
  #endif
  return data_[element];
}

template <typename data_t>
cvector_t<data_t> QubitVector<data_t>::vector() const {
  cvector_t<data_t> ret(data_size_, 0.);
  auto lambda = [&](int_t k)-> void {ret[k] = data_[k];};
  kernel_.apply_lambda(data_size_, lambda);
  return ret;
}


//------------------------------------------------------------------------------
// State initialize component
//------------------------------------------------------------------------------
template <typename data_t>
void QubitVector<data_t>::initialize_component(const Index::reg_t &qubits, const cvector_t<double> &state0) {

  cvector_t<data_t> state = convert(state0);

  // Lambda function for initializing component
  auto lambda = [&](const Index::indexes_t &inds)->void {
    const uint_t DIM = Index::BITS[qubits.size()];
    std::complex<data_t> cache = data_[inds[0]];  // the k-th component of non-initialized vector
    for (size_t i = 0; i < DIM; i++) {
      data_[inds[i]] = kernel_.cmul(cache, state[i]);  // set component to psi[k] * state[i]
    }    // (where psi is is the post-reset state of the non-initialized qubits)
   };
  // Use the lambda function
  kernel_.apply_lambda(data_size_, lambda, qubits);
}

//------------------------------------------------------------------------------
// Utility
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::zero() {
  auto lambda = [&](const int_t k)->void {
    data_[k] = data_t(0.0);
  };
  kernel_.apply_lambda(data_size_, lambda);
}

template <typename data_t>
cvector_t<data_t> QubitVector<data_t>::convert(const cvector_t<double>& v) const {
  cvector_t<data_t> ret(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    ret[i] = v[i];
  return ret;
}


template <typename data_t>
void QubitVector<data_t>::set_num_qubits(size_t num_qubits) {

  size_t prev_num_qubits = num_qubits_;
  num_qubits_ = num_qubits;
  data_size_ = Index::BITS[num_qubits];

  if (checkpoint_) {
    free(checkpoint_);
    checkpoint_ = nullptr;
  }

  // Free any currently assigned memory
  if (data_) {
    if (prev_num_qubits != num_qubits_) {
      free(data_);
      data_ = nullptr;
    }
  }

  // Allocate memory for new vector
  if (data_ == nullptr)
    data_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>) * data_size_));
}

template <typename data_t>
size_t QubitVector<data_t>::required_memory_mb(uint_t num_qubits) const {

  size_t unit = std::log2(sizeof(std::complex<data_t>));
  size_t shift_mb = std::max<int_t>(0, num_qubits + unit - 20);
  size_t mem_mb = 1ULL << shift_mb;
  return mem_mb;
}


template <typename data_t>
void QubitVector<data_t>::checkpoint() {
  if (!checkpoint_)
    checkpoint_ = reinterpret_cast<std::complex<data_t>*>(malloc(sizeof(std::complex<data_t>) * data_size_));

  auto lambda = [&](int_t k) {checkpoint_[k] = data_[k];};
  kernel_.apply_lambda(data_size_, lambda);
}


template <typename data_t>
void QubitVector<data_t>::revert(bool keep) {

  #ifdef DEBUG
  check_checkpoint();
  #endif

  // If we aren't keeping checkpoint we don't need to copy memory
  // we can simply swap the pointers and free discarded memory
  if (!keep) {
    free(data_);
    data_ = checkpoint_;
    checkpoint_ = nullptr;
  } else {
    auto lambda = [&](int_t k) {data_[k] = checkpoint_[k];};
    kernel_.apply_lambda(data_size_, lambda);
  }
}

template <typename data_t>
std::complex<double> QubitVector<data_t>::inner_product() const {

  #ifdef DEBUG
  check_checkpoint();
  #endif
  // Lambda function for inner product with checkpoint state
  auto lambda = [&](int_t k, double &val_re, double &val_im)->void {
    const std::complex<double> z = data_[k] * std::conj(checkpoint_[k]);
    val_re += std::real(z);
    val_im += std::imag(z);
  };
  std::complex<double> accum = 0.0;
  kernel_.apply_reduction_lambda(data_size_, accum, lambda);
  return accum;
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::initialize() {
  zero();
  data_[0] = 1.;
}

template <typename data_t>
void QubitVector<data_t>::initialize_from_vector(const cvector_t<double> &statevec) {
  if (data_size_ != statevec.size()) {
    std::string error = "QubitVector::initialize input vector is incorrect length (" + 
                        std::to_string(data_size_) + "!=" +
                        std::to_string(statevec.size()) + ")";
    throw std::runtime_error(error);
  }
  auto lambda = [&](int_t k) {data_[k] = statevec[k];};
  kernel_.apply_lambda(data_size_, lambda);
}

template <typename data_t>
void QubitVector<data_t>::initialize_from_data(const std::complex<data_t>* statevec, const size_t num_states) {
  if (data_size_ != num_states) {
    std::string error = "QubitVector::initialize input vector is incorrect length (" +
                        std::to_string(data_size_) + "!=" + std::to_string(num_states) + ")";
    throw std::runtime_error(error);
  }
  auto lambda = [&](int_t k) {data_[k] = statevec[k];};
  kernel_.apply_lambda(data_size_, lambda);
}


/*******************************************************************************
 *
 * CONFIG SETTINGS
 *
 ******************************************************************************/

template <typename data_t>
void QubitVector<data_t>::set_omp_threads(int n) {
  if (n > 0)
    omp_threads_ = n;

  // Set threads of OMP Kernel
  kernel_.set_threads(n);
}

template <typename data_t>
void QubitVector<data_t>::set_omp_threshold(int n) {
  if (n > 0)
    omp_threshold_ = n;
  // Set threshold of OMP Kernel
  kernel_.set_threshold(1ULL << n);
}

template <typename data_t>
void QubitVector<data_t>::set_json_chop_threshold(double threshold) {
  json_chop_threshold_ = threshold;
}


/*******************************************************************************
 *
 * MATRIX MULTIPLICATION
 *
 ******************************************************************************/
template <typename data_t>
void QubitVector<data_t>::apply_matrix(const Index::reg_t &qubits,
                                       const cvector_t<double> &mat) {
  kernel_.apply_matrix<data_t>(data_, data_size_, mat, qubits);
}


template <typename data_t>
void QubitVector<data_t>::apply_diagonal_matrix(const Index::reg_t &qubits,
                                                const cvector_t<double> &diag) {

  kernel_.apply_diagonal_matrix<data_t>(data_, data_size_, diag, qubits);
}


template <typename data_t>
void QubitVector<data_t>::apply_permutation_matrix(const Index::reg_t& qubits,
                                                   const std::vector<std::pair<uint_t, uint_t>> &pairs) {
  kernel_.apply_permutation_matrix<data_t>(data_, data_size_, pairs, qubits);
}


template <typename data_t>
void QubitVector<data_t>::apply_multiplexer(const Index::reg_t &control_qubits,
                                            const Index::reg_t &target_qubits,
                                            const cvector_t<double>  &mat) {
  auto lambda = [&](const Index::indexes_t &inds) -> void {
  // General implementation
  const size_t control_count = control_qubits.size();
  const size_t target_count = target_qubits.size();
  const uint_t DIM = Index::BITS[(target_count + control_count)];
  const uint_t columns = Index::BITS[target_count];
  const uint_t blocks = Index::BITS[control_count];
  // Lambda function for stacked matrix multiplication
  auto cache = std::make_unique<std::complex<data_t>[]>(DIM);
  for (uint_t i = 0; i < DIM; i++) {
    const auto ii = inds[i];
    cache[i] = data_[ii];
    data_[ii] = 0.;
  }
  // update state vector
  for (uint_t b = 0; b < blocks; b++)
    for (uint_t i = 0; i < columns; i++)
      for (uint_t j = 0; j < columns; j++) {
        data_[inds[i + b * columns]] += kernel_.cmul(
            cache[b * columns + j], mat[i + b * columns + DIM * j]);
      }
  };

  // Use the lambda function
  auto qubits = target_qubits;
  for (const auto &q : control_qubits) {
    qubits.push_back(q);
  }
  kernel_.apply_lambda(data_size_, lambda, qubits);
}


/*******************************************************************************
 *
 * APPLY OPTIMIZED GATES
 *
 ******************************************************************************/

//------------------------------------------------------------------------------
// Multi-controlled gates
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::apply_mcx(const Index::reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = Index::MASKS[N - 1];
  const size_t pos1 = Index::MASKS[N];

  switch (N) {
    case 1: {
      // Lambda function for X gate
      auto lambda = [&](const Index::areg_t<2> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CX gate
      auto lambda = [&](const Index::areg_t<4> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for Toffli gate
      auto lambda = [&](const Index::areg_t<8> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled X gate
      auto lambda = [&](const Index::indexes_t &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, qubits);
    }
  } // end switch
}

template <typename data_t>
void QubitVector<data_t>::apply_mcy(const Index::reg_t &qubits) {
  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = Index::MASKS[N - 1];
  const size_t pos1 = Index::MASKS[N];
  const std::complex<data_t> I(0., 1.);

  switch (N) {
    case 1: {
      // Lambda function for Y gate
      auto lambda = [&](const Index::areg_t<2> &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CY gate
      auto lambda = [&](const Index::areg_t<4> &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for CCY gate
      auto lambda = [&](const Index::areg_t<8> &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled Y gate
      auto lambda = [&](const Index::indexes_t &inds)->void {
        const std::complex<data_t> cache = data_[inds[pos0]];
        data_[inds[pos0]] = -I * data_[inds[pos1]];
        data_[inds[pos1]] = I * cache;
      };
      kernel_.apply_lambda(data_size_, lambda, qubits);
    }
  } // end switch
}

template <typename data_t>
void QubitVector<data_t>::apply_mcswap(const Index::reg_t &qubits) {
  // Calculate the swap positions for the last two qubits.
  // If N = 2 this is just a regular SWAP gate rather than a controlled-SWAP gate.
  const size_t N = qubits.size();
  const size_t pos0 = Index::MASKS[N - 1];
  const size_t pos1 = pos0 + Index::BITS[N - 2];

  switch (N) {
    case 2: {
      // Lambda function for SWAP gate
      auto lambda = [&](const Index::areg_t<4> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for C-SWAP gate
      auto lambda = [&](const Index::areg_t<8> &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled SWAP gate
      auto lambda = [&](const Index::indexes_t &inds)->void {
        std::swap(data_[inds[pos0]], data_[inds[pos1]]);
      };
      kernel_.apply_lambda(data_size_, lambda, qubits);
    }
  } // end switch
}

template <typename data_t>
void QubitVector<data_t>::apply_mcphase(const Index::reg_t &qubits, const std::complex<double> _phase) {
  const size_t N = qubits.size();
  // Cast phase to correct data type
  const auto phase = std::complex<data_t>(_phase);
  switch (N) {
    case 1: {
      // Lambda function for arbitrary Phase gate with diagonal [1, phase]
      auto lambda = [&](const Index::areg_t<2> &inds)->void {
        data_[inds[1]] *= phase;
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<1>({{qubits[0]}}));
      return;
    }
    case 2: {
      // Lambda function for CPhase gate with diagonal [1, 1, 1, phase]
      auto lambda = [&](const Index::areg_t<4> &inds)->void {
        data_[inds[3]] *= phase;
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      auto lambda = [&](const Index::areg_t<8> &inds)->void {
         data_[inds[7]] *= phase;
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled Phase gate
      // with diagonal [1, ..., 1, phase]
      auto lambda = [&](const Index::indexes_t &inds)->void {
         data_[inds[Index::MASKS[N]]] *= phase;
      };
      kernel_.apply_lambda(data_size_, lambda, qubits);
    }
  } // end switch
}

template <typename data_t>
void QubitVector<data_t>::apply_mcu(const Index::reg_t &qubits,
                                    const cvector_t<double> &mat) {

  // Calculate the permutation positions for the last qubit.
  const size_t N = qubits.size();
  const size_t pos0 = Index::MASKS[N - 1];
  const size_t pos1 = Index::MASKS[N];

  // Check if matrix is actually diagonal and if so use 
  // diagonal matrix lambda function
  // TODO: this should be changed to not check doubles with ==
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    // Check if actually a phase gate
    if (mat[0] == 1.0) {
      apply_mcphase(qubits, mat[3]);
      return;
    }
    // Otherwise apply general diagonal gate
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    // Diagonal version
    switch (N) {
      case 1: {
        // If N=1 this is just a single-qubit matrix
        kernel_.apply_diagonal_matrix<data_t>(data_, data_size_, diag, qubits[0]);
        return;
      }
      case 2: {
        // Lambda function for CU gate
        auto lambda = [&](const Index::areg_t<4> &inds)->void {
          data_[inds[pos0]] = kernel_.cmul(data_[inds[pos0]], diag[0]);
          data_[inds[pos1]] = kernel_.cmul(data_[inds[pos1]], diag[1]);
        };
        kernel_.apply_lambda(data_size_, lambda, areg_t<2>({{qubits[0], qubits[1]}}));
        return;
      }
      case 3: {
        // Lambda function for CCU gate
        auto lambda = [&](const Index::areg_t<8> &inds)->void {
          data_[inds[pos0]] = kernel_.cmul(data_[inds[pos0]], diag[0]);
          data_[inds[pos1]] = kernel_.cmul(data_[inds[pos1]], diag[1]);
        };
        kernel_.apply_lambda(data_size_, lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
        return;
      }
      default: {
        // Lambda function for general multi-controlled U gate
        auto lambda = [&](const Index::indexes_t &inds)->void {
          data_[inds[pos0]] = kernel_.cmul(data_[inds[pos0]], diag[0]);
          data_[inds[pos1]] = kernel_.cmul(data_[inds[pos1]], diag[1]);
        };
        kernel_.apply_lambda(data_size_, lambda, qubits);
        return;
      }
    } // end switch
  }

  // Non-diagonal version
  switch (N) {
    case 1: {
      // If N=1 this is just a single-qubit matrix
      kernel_.apply_matrix<data_t>(data_, data_size_, mat, qubits[0]);
      return;
    }
    case 2: {
      // Lambda function for CU gate
      auto lambda = [&](const Index::areg_t<4> &inds)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = kernel_.cmul(data_[inds[pos0]], mat[0]) + kernel_.cmul(data_[inds[pos1]], mat[2]);
      data_[inds[pos1]] = kernel_.cmul(cache, mat[1]) + kernel_.cmul(data_[inds[pos1]], mat[3]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<2>({{qubits[0], qubits[1]}}));
      return;
    }
    case 3: {
      // Lambda function for CCU gate
      auto lambda = [&](const Index::areg_t<8> &inds)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = kernel_.cmul(data_[inds[pos0]], mat[0]) + kernel_.cmul(data_[inds[pos1]], mat[2]);
      data_[inds[pos1]] = kernel_.cmul(cache, mat[1]) + kernel_.cmul(data_[inds[pos1]], mat[3]);
      };
      kernel_.apply_lambda(data_size_, lambda, areg_t<3>({{qubits[0], qubits[1], qubits[2]}}));
      return;
    }
    default: {
      // Lambda function for general multi-controlled U gate
      auto lambda = [&](const Index::indexes_t &inds)->void {
      const auto cache = data_[inds[pos0]];
      data_[inds[pos0]] = kernel_.cmul(data_[inds[pos0]], mat[0]) + kernel_.cmul(data_[inds[pos1]], mat[2]);
      data_[inds[pos1]] = kernel_.cmul(cache, mat[1]) + kernel_.cmul(data_[inds[pos1]], mat[3]);
      };
      kernel_.apply_lambda(data_size_, lambda, qubits);
      return;
    }
  } // end switch
}

//------------------------------------------------------------------------------
// Single-qubit matrices
//------------------------------------------------------------------------------

template <typename data_t>
void QubitVector<data_t>::apply_matrix(const uint_t qubit,
                                       const cvector_t<double>& mat) {
  kernel_.apply_matrix<data_t>(data_, data_size_, mat, qubit);
}

template <typename data_t>
void QubitVector<data_t>::apply_diagonal_matrix(const uint_t qubit,
                                                const cvector_t<double>& diag) {
  kernel_.apply_diagonal_matrix<data_t>(data_, data_size_, diag, qubit);
}

/*******************************************************************************
 *
 * NORMS
 *
 ******************************************************************************/
template <typename data_t>
double QubitVector<data_t>::norm() const {
  // Lambda function for norm
  auto lambda = [&](int_t k, double &val)->void {
    val += std::real(data_[k] * std::conj(data_[k]));
  };
  double accum = 0.0;
  kernel_.apply_reduction_lambda(data_size_, accum, lambda);
  return accum;
}

template <typename data_t>
double QubitVector<data_t>::norm(const Index::reg_t &qubits, const cvector_t<double> &mat) const {

  // Static array optimized lambda functions
  switch (qubits.size()) {
    case 1:
      return norm(qubits[0], mat);
    case 2: {
      // Lambda function for 2-qubit matrix norm
      auto lambda = [&](const Index::areg_t<4> &inds,  double &val)->void {
        for (size_t i = 0; i < 4; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < 4; j++)
            vi += kernel_.cmul(mat[i + 4 * j], data_[inds[j]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      areg_t<2> qubits_arr = {{qubits[0], qubits[1]}};
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits_arr);
      return accum;
    }
    case 3: {
      // Lambda function for 3-qubit matrix norm
      auto lambda = [&](const Index::areg_t<8> &inds, double &val)->void {
        for (size_t i = 0; i < 8; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < 8; j++)
            vi += kernel_.cmul(mat[i + 8 * j], data_[inds[j]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      areg_t<3> qubits_arr = {{qubits[0], qubits[1], qubits[2]}};
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits_arr);
      return accum;
    }
    case 4: {
      // Lambda function for 4-qubit matrix norm
      auto lambda = [&](const Index::areg_t<16> &inds, double &val)->void {
        for (size_t i = 0; i < 16; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < 16; j++)
            vi += kernel_.cmul(mat[i + 16 * j], data_[inds[j]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      areg_t<4> qubits_arr = {{qubits[0], qubits[1], qubits[2], qubits[3]}};
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits_arr);
      return accum;
    }
    default: {
      // Lambda function for N-qubit matrix norm
      const uint_t DIM = Index::BITS[qubits.size()];
      auto lambda = [&](const Index::indexes_t &inds, double &val)->void {
        for (size_t i = 0; i < DIM; i++) {
          std::complex<data_t> vi = 0;
          for (size_t j = 0; j < DIM; j++)
            vi += kernel_.cmul(mat[i + DIM * j], data_[inds[j]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      // Use the lambda function
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits);
      return accum;
    }
  } // end switch
}

template <typename data_t>
double QubitVector<data_t>::norm_diagonal(const Index::reg_t &qubits, const cvector_t<double> &mat) const {

  // Static array optimized lambda functions
  switch (qubits.size()) {
    case 1:
      return norm_diagonal(qubits[0], mat);
    case 2: {
      // Lambda function for 2-qubit matrix norm
      auto lambda = [&](const Index::areg_t<4> &inds, double &val)->void {
        for (size_t i = 0; i < 4; i++) {
          const auto vi = kernel_.cmul(mat[i], data_[inds[i]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      areg_t<2> qubits_arr = {{qubits[0], qubits[1]}};
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits_arr);
      return accum;
    }
    case 3: {
      // Lambda function for 3-qubit matrix norm
      auto lambda = [&](const Index::areg_t<8> &inds, double &val)->void {
        for (size_t i = 0; i < 8; i++) {
          const auto vi = kernel_.cmul(mat[i], data_[inds[i]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      areg_t<3> qubits_arr = {{qubits[0], qubits[1], qubits[2]}};
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits_arr);
      return accum;
    }
    case 4: {
      // Lambda function for 4-qubit matrix norm
      auto lambda = [&](const Index::areg_t<16> &inds, double &val)->void {
        for (size_t i = 0; i < 16; i++) {
          const auto vi = kernel_.cmul(mat[i], data_[inds[i]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      areg_t<4> qubits_arr = {{qubits[0], qubits[1], qubits[2], qubits[3]}};
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits_arr);
      return accum;
    }
    default: {
      // Lambda function for N-qubit matrix norm
      const uint_t DIM = Index::BITS[qubits.size()];
      auto lambda = [&](const Index::indexes_t &inds, double &val)->void {
        for (size_t i = 0; i < DIM; i++) {
          const auto vi = kernel_.cmul(mat[i], data_[inds[i]]);
          val += std::real(vi * std::conj(vi));
        }
      };
      double accum = 0.0;
      kernel_.apply_reduction_lambda(data_size_, accum, lambda, qubits);
      return accum;
    }
  } // end switch
}

//------------------------------------------------------------------------------
// Single-qubit specialization
//------------------------------------------------------------------------------
template <typename data_t>
double QubitVector<data_t>::norm(const uint_t qubit, const cvector_t<double> &mat) const {
  // Check if input matrix is diagonal, and if so use diagonal function.
  if (mat[1] == 0.0 && mat[2] == 0.0) {
    const cvector_t<double> diag = {{mat[0], mat[3]}};
    return norm_diagonal(qubit, diag);
  }

  // Lambda function for norm reduction to real value.
  auto lambda = [&](const Index::areg_t<2> &inds, double &val)->void {
    auto v0 = kernel_.cmul(mat[0], data_[inds[0]]) + kernel_.cmul(mat[2], data_[inds[1]]);
    auto v1 = kernel_.cmul(mat[1], data_[inds[0]]) + kernel_.cmul(mat[3], data_[inds[1]]);
    val += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  double accum = 0.0;
  kernel_.apply_reduction_lambda(data_size_, accum, lambda, areg_t<1>({{qubit}}));
  return accum;
}

template <typename data_t>
double QubitVector<data_t>::norm_diagonal(const uint_t qubit, const cvector_t<double> &mat) const {
  // Lambda function for norm reduction to real value.
  auto lambda = [&](const Index::areg_t<2> &inds, double &val)->void {
    auto v0 = kernel_.cmul(mat[0], data_[inds[0]]);
    auto v1 = kernel_.cmul(mat[1], data_[inds[1]]);
    val += std::real(v0 * std::conj(v0)) + std::real(v1 * std::conj(v1));
  };
  double accum = 0.0;
  kernel_.apply_reduction_lambda(data_size_, accum, lambda, areg_t<1>({{qubit}}));
  return accum;
}


/*******************************************************************************
 *
 * Probabilities
 *
 ******************************************************************************/
template <typename data_t>
double QubitVector<data_t>::probability(const uint_t outcome) const {
  return std::real(data_[outcome] * std::conj(data_[outcome]));
}

template <typename data_t>
std::vector<double> QubitVector<data_t>::probabilities() const {
  std::vector<double> probs(data_size_, 0.);
  auto lambda = [&] (int_t j) {probs[j] = probability(j);};
  kernel_.apply_lambda(data_size_, lambda);
  return probs;
}

template <typename data_t>
std::vector<double> QubitVector<data_t>::probabilities(const Index::reg_t &qubits) const {

  const size_t N = qubits.size();
  const int_t DIM = Index::BITS[N];
  const int_t END = Index::BITS[num_qubits() - N];

  auto qubits_sorted = qubits;
  std::sort(qubits_sorted.begin(), qubits_sorted.end());
  if ((N == num_qubits_) && (qubits == qubits_sorted))
    return probabilities();

  std::vector<double> probs(DIM, 0.);
  #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
  {
    std::vector<data_t> probs_private(DIM, 0.);
    #pragma omp for
      for (int_t k = 0; k < END; k++) {
        auto idx = Index::indexes(qubits, qubits_sorted, k);
        for (int_t m = 0; m < DIM; ++m) {
          probs_private[m] += probability(idx[m]);
        }
      }
    #pragma omp critical
    for (int_t m = 0; m < DIM; ++m) {
      probs[m] += probs_private[m];
    }
  }
  
  return probs;
}

//------------------------------------------------------------------------------
// Sample measure outcomes
//------------------------------------------------------------------------------
template <typename data_t>
reg_t QubitVector<data_t>::sample_measure(const std::vector<double> &rnds) const {

  const int_t END = 1LL << num_qubits();
  const int_t SHOTS = rnds.size();
  reg_t samples;
  samples.assign(SHOTS, 0);

  const int INDEX_SIZE = sample_measure_index_size_;
  const int_t INDEX_END = Index::BITS[INDEX_SIZE];
  // Qubit number is below index size, loop over shots
  if (END < INDEX_END) {
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < SHOTS; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample;
        for (sample = 0; sample < END - 1; ++sample) {
          p += probability(sample);
          if (rnd < p)
            break;
        }
        samples[i] = sample;
      }
    } // end omp parallel
  }
  // Qubit number is above index size, loop over index blocks
  else {
    // Initialize indexes
    std::vector<double> idxs;
    idxs.assign(INDEX_END, 0.0);
    uint_t loop = (END >> INDEX_SIZE);
    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < INDEX_END; ++i) {
        uint_t base = loop * i;
        double total = .0;
        double p = .0;
        for (uint_t j = 0; j < loop; ++j) {
          uint_t k = base | j;
          p = probability(k);
          total += p;
        }
        idxs[i] = total;
      }
    } // end omp parallel

    #pragma omp parallel if (num_qubits_ > omp_threshold_ && omp_threads_ > 1) num_threads(omp_threads_)
    {
      #pragma omp for
      for (int_t i = 0; i < SHOTS; ++i) {
        double rnd = rnds[i];
        double p = .0;
        int_t sample = 0;
        for (uint_t j = 0; j < idxs.size(); ++j) {
          if (rnd < (p + idxs[j])) {
            break;
          }
          p += idxs[j];
          sample += loop;
        }

        for (; sample < END - 1; ++sample) {
          p += probability(sample);
          if (rnd < p){
            break;
          }
        }
        samples[i] = sample;
      }
    } // end omp parallel
  }
  return samples;
}

//------------------------------------------------------------------------------
} // end namespace QV
//------------------------------------------------------------------------------

// ostream overload for templated qubitvector
template <typename data_t>
inline std::ostream &operator<<(std::ostream &out, const QV::QubitVector<data_t>&qv) {

  out << "[";
  size_t last = qv.size() - 1;
  for (size_t i = 0; i < qv.size(); ++i) {
    out << qv[i];
    if (i != last)
      out << ", ";
  }
  out << "]";
  return out;
}

//------------------------------------------------------------------------------
#endif // end module
