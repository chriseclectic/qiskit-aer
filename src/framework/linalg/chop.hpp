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

#ifndef _aer_framework_linalg_chop_hpp_
#define _aer_framework_linalg_chop_hpp_

#include <array>
#include <functional>
#include <map>
#include <unordered_map>
#include <vector>

#include "framework/json.hpp"
#include "framework/linalg/enable_if_numeric.hpp"
#include "framework/matrix.hpp"
#include "framework/types.hpp"

namespace AER {
namespace Linalg {

// This defines functions 'square' for entrywise square of
// numeric types, and 'isquare' for inplace entywise square.

//----------------------------------------------------------------------------
// Chop of general type
//----------------------------------------------------------------------------

template <class T, typename = enable_if_numeric_t<T>>
T chop(const T& val, double epsilon) {
  if (std::abs(val) < epsilon) return T(0);
  return val;
}

template <class T, typename = enable_if_numeric_t<T>>
T& ichop(T& val, double epsilon) {
  if (std::abs(val) < epsilon) val = T(0);
  return val;
}

//----------------------------------------------------------------------------
// Chop of complex type
//----------------------------------------------------------------------------

template <class T, typename = enable_if_numeric_t<T>>
std::complex<T> chop(const std::complex<T>& val, double epsilon) {
  return complex_t(chop(T(val.real()), epsilon), chop(T(val.imag()), epsilon));
}

template <class T, typename = enable_if_numeric_t<T>>
std::complex<T>& ichop(std::complex<T>& val, double epsilon) {
  val.real(ichop(T(val.real()), epsilon));
  val.imag(ichop(T(val.imag()), epsilon));
  return val;
}

//----------------------------------------------------------------------------
// Entrywise chop of std::array
//----------------------------------------------------------------------------

template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N>& ichop(std::array<T, N>& val, double epsilon) {
  for (const auto& v : val) ichop(v, epsilon);
  return val;
}

template <class T, size_t N, typename = enable_if_numeric_t<T>>
std::array<T, N> chop(const std::array<T, N>& val, double epsilon) {
  std::array<T, N> result = val;
  return ichop(result, epsilon);
}

//----------------------------------------------------------------------------
// Entrywise chop of std::vector
//----------------------------------------------------------------------------

template <typename T, typename = enable_if_numeric_t<T>>
std::vector<T> chop(const std::vector<T>& val, double epsilon) {
  std::vector<T> result;
  result.reserve(val.size());
  for (const auto& v : val) result.push_back(chop(v, epsilon));
  return result;
}

template <typename T, typename = enable_if_numeric_t<T>>
std::vector<T>& ichop(std::vector<T>& val, double epsilon) {
  for (auto& v : val) ichop(v, epsilon);
  return val;
}

//----------------------------------------------------------------------------
// Entrywise chop of std::map
//----------------------------------------------------------------------------

template <class T1, class T2, class T3, class T4,
          typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4> chop(const std::map<T1, T2, T3, T4>& val,
                              double epsilon) {
  std::map<T1, T2, T3, T4> result;
  for (const auto& pair : val) {
    result[pair.first] = chop(pair.second, epsilon);
  }
  return result;
}

template <class T1, class T2, class T3, class T4,
          typename = enable_if_numeric_t<T2>>
std::map<T1, T2, T3, T4>& ichop(std::map<T1, T2, T3, T4>& val, double epsilon) {
  for (auto& pair : val) {
    ichop(pair.second, epsilon);
  }
  return val;
}

//----------------------------------------------------------------------------
// Entrywise chop of std::unordered_map
//----------------------------------------------------------------------------

template <class T1, class T2, class T3, class T4, class T5,
          typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5> chop(
    const std::unordered_map<T1, T2, T3, T4, T5>& val, double epsilon) {
  std::unordered_map<T1, T2, T3, T4, T5> result;
  for (const auto& pair : val) {
    result[pair.first] = chop(pair.second, epsilon);
  }
  return result;
}

template <class T1, class T2, class T3, class T4, class T5,
          typename = enable_if_numeric_t<T2>>
std::unordered_map<T1, T2, T3, T4, T5>& ichop(
    std::unordered_map<T1, T2, T3, T4, T5>& val, double epsilon) {
  for (auto& pair : val) {
    ichop(pair.second, epsilon);
  }
  return val;
}

//----------------------------------------------------------------------------
// Entrywise chop of matrix
//----------------------------------------------------------------------------

template <class T, typename = enable_if_numeric_t<T>>
matrix<T>& ichop(matrix<T>& val, double epsilon) {
  for (size_t j = 0; j < val.size(); j++) {
    ichop(val[j]);
  }
  return val;
}

template <class T, typename = enable_if_numeric_t<T>>
matrix<T> chop(const matrix<T>& val, double epsilon) {
  matrix<T> result = val;
  return ichop(result);
}

//----------------------------------------------------------------------------
// Entrywise chop of JSON
//----------------------------------------------------------------------------

json_t& ichop(json_t& val, double epsilon) {
  // Terminating case
  if (val.is_number()) {
    double num = val;
    val = ichop(num, epsilon);
    return val;
  }
  // Recursive cases
  if (val.is_array()) {
    for (size_t pos = 0; pos < val.size(); pos++) {
      ichop(val[pos], epsilon);
    }
    return val;
  }
  if (val.is_object()) {
    for (auto it = val.begin(); it != val.end(); ++it) {
      ichop(val[it.key()], epsilon);
    }
    return val;
  }
  throw std::invalid_argument("Input JSONs cannot be chopped.");
}

json_t chop(const json_t& val, double epsilon) {
  json_t result = val;
  return ichop(result, epsilon);
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif