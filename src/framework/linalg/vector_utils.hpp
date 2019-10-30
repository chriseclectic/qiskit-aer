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

#ifndef _aer_framework_linalg_vector_utils_hpp_
#define _aer_framework_linalg_vector_utils_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/linalg/enable_if_numeric.hpp"
#include "framework/types.hpp"

namespace AER {
namespace Linalg {

//------------------------------------------------------------------------------
// Vector functions
//------------------------------------------------------------------------------

// Return true of the vector has norm-1.
template <typename T>
double is_unit_vector(const std::vector<T> &vec);

// Conjugate a vector
template <typename T>
std::vector<std::complex<T>> conjugate(const std::vector<std::complex<T>> &v);

// Compute the Euclidean 2-norm of a vector
template <typename T>
double norm(const std::vector<T> &vec);

// Return the matrix formed by taking the outproduct of two vector |ket><bra|
template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra);

template <typename T>
inline matrix<T> projector(const std::vector<T> &ket) {
  return outer_product(ket, ket);
}

// Tensor product vector
template <typename T>
std::vector<T> tensor_product(const std::vector<T> &v, const std::vector<T> &w);

// Return a new vector formed by multiplying each element of the input vector
// with a scalar. The product of types T1 * T2 must be valid.
template <typename T1, typename T2>
std::vector<T1> scalar_multiply(const std::vector<T1> &vec, T2 val);

// Inplace multiply each entry in a vector by a scalar and returns a reference
// to the input vector argument. The product of types T1 * T2 must be valid.
template <typename T1, typename T2>
std::vector<T1> &scalar_multiply_inplace(std::vector<T1> &vec, T2 scalar);

// Truncate the first argument its absolute value is less than epsilon
// this function returns a refernce to the chopped first argument
double &chop_inplace(double &val, double epsilon);
std::complex<double> &chop_inplace(std::complex<double> &val, double epsilon);

double chop(double val, double epsilon);

// As above for complex first arguments
template <typename T>
std::complex<T> chop(std::complex<T> val, double epsilon);
// Truncate each element in a vector if its absolute value is less than epsilon
// This function returns a reference to the chopped input vector
template <typename T>
std::vector<T> &chop_inplace(std::vector<T> &vec, double epsilon);

template <typename T>
std::vector<T> chop(const std::vector<T> &vec, double epsilon);

// Add rhs vector to lhs using move semantics.
// rhs should not be used after this operation.
template <class T>
void combine(std::vector<T> &lhs, const std::vector<T> &rhs);

// Convert a dense vector into sparse ket form.
// epsilon determins the threshold for which small values will be removed from
// the output. The base of the ket (2-10 for qudits, or 16 for hexadecimal)
// specifies the subsystem dimension and the base of the dit-string labels.
template <typename T>
std::map<std::string, T> vec2ket(const std::vector<T> &vec, double epsilon,
                                 uint_t base = 2);

//==============================================================================
// Implementations
//==============================================================================

template <class T>
bool is_unit_vector(const std::vector<T> &vec, double threshold) {
  return (std::abs(norm<T>(vec) - 1.0) < threshold);
}

template <typename T>
std::vector<std::complex<T>> conjugate(const std::vector<std::complex<T>> &v) {
  std::vector<std::complex<T>> ret;
  std::transform(
      v.cbegin(), v.cend(), std::back_inserter(ret),
      [](const std::complex<T> &c) -> std::complex<T> { return std::conj(c); });
  return ret;
}

template <typename T>
double norm(const std::vector<T> &vec) {
  double val = 0.0;
  for (const auto v : vec) {
    val += std::real(v * std::conj(v));
  }
  return std::sqrt(val);
}

template <typename T>
matrix<T> outer_product(const std::vector<T> &ket, const std::vector<T> &bra) {
  const uint_t d1 = ket.size();
  const uint_t d2 = bra.size();
  matrix<T> ret(d1, d2);
  for (uint_t i = 0; i < d1; i++)
    for (uint_t j = 0; j < d2; j++) {
      ret(i, j) = ket[i] * std::conj(bra[j]);
    }
  return ret;
}

template <typename T>
std::vector<T> tensor_product(const std::vector<T> &vec1,
                              const std::vector<T> &vec2) {
  std::vector<T> ret;
  ret.reserve(vec1.size() * vec2.size());
  for (const auto &a : vec1)
    for (const auto &b : vec2) {
      ret.push_back(a * b);
    }
  return ret;
}

template <typename T1, typename T2>
std::vector<T1> scalar_multiply(const std::vector<T1> &vec, T2 val) {
  std::vector<T1> ret;
  ret.reserve(vec.size());
  for (const auto &elt : vec) {
    ret.push_back(val * elt);
  }
  return ret;
}

template <typename T1, typename T2>
std::vector<T1> &scalar_multiply_inplace(std::vector<T1> &vec, T2 val) {
  for (auto &elt : vec) {
    elt = val * elt;  // use * incase T1 doesn't have *= method
  }
  return vec;
}

double &chop_inplace(double &val, double epsilon) {
  if (std::abs(val) < epsilon) val = 0.;
  return val;
}

std::complex<double> &chop_inplace(std::complex<double> &val, double epsilon) {
  val.real(chop(val.real(), epsilon));
  val.imag(chop(val.imag(), epsilon));
  return val;
}

template <typename T>
std::vector<T> &chop_inplace(std::vector<T> &vec, double epsilon) {
  if (epsilon > 0.)
    for (auto &v : vec) chop_inplace(v, epsilon);
  return vec;
}

double chop(double val, double epsilon) {
  return (std::abs(val) < epsilon) ? 0. : val;
}

template <typename T>
std::complex<T> chop(std::complex<T> val, double epsilon) {
  return {chop(val.real(), epsilon), chop(val.imag(), epsilon)};
}

template <typename T>
std::vector<T> chop(const std::vector<T> &vec, double epsilon) {
  std::vector<T> tmp;
  tmp.reserve(vec.size());
  for (const auto &v : vec) tmp.push_back(chop(v, epsilon));
  return tmp;
}

template <class T>
void combine(std::vector<T> &lhs, const std::vector<T> &rhs) {
  // if lhs is empty, set it to be rhs vector
  if (lhs.size() == 0) {
    lhs = rhs;
    return;
  }
  // if lhs is not empty rhs must be same size
  if (lhs.size() != rhs.size()) {
    throw std::invalid_argument(
        "Utils::combine (vectors are not same length.)");
  }
  for (size_t j = 0; j < lhs.size(); ++j) {
    lhs[j] += rhs[j];
  }
}

template <typename T>
std::map<std::string, T> vec2ket(const std::vector<T> &vec, double epsilon,
                                 uint_t base) {
  bool hex_output = false;
  if (base == 16) {
    hex_output = true;
    base = 2;  // If hexadecimal strings we convert to bin first
  }
  // check vector length
  size_t dim = vec.size();
  double n = std::log(dim) / std::log(base);
  uint_t nint = std::trunc(n);
  if (std::abs(nint - n) > 1e-5) {
    std::stringstream ss;
    ss << "vec2ket (vector dimension " << dim << " is not of size " << base
       << "^n)";
    throw std::invalid_argument(ss.str());
  }
  std::map<std::string, T> ketmap;
  for (size_t k = 0; k < dim; ++k) {
    T val = chop(vec[k], epsilon);
    if (std::abs(val) > epsilon) {
      std::string key =
          (hex_output) ? Utils::int2hex(k) : Utils::int2string(k, base, nint);
      ketmap.insert({key, val});
    }
  }
  return ketmap;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif