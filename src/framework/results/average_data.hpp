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

#ifndef _aer_framework_results_average_data_hpp_
#define _aer_framework_results_average_data_hpp_

#include "framework/json.hpp"
#include "framework/linalg/linalg.hpp"
#include "framework/types.hpp"

namespace AER {

template <typename T>
class AverageData {
 public:
  // Access data
  stringmap_t<T>& value();

  // Add a new datum to the snapshot at the specified key
  // Uses copy semantics
  void add_average_data(const std::string& key, const T &datum);

  // Add a new datum to the snapshot at the specified key
  // Uses move semantics
  void add_average_data(const std::string& key, T &&datum) noexcept;

  // Combine with another snapshot container
  // Uses move semantics
  void combine(AverageData<T> &&other) noexcept;

  // Add to a JSON object
  void add_to_json(json_t& js);

  // Clear all stored data
  void clear();

  // Return True if this data type is enabled
  bool enabled() { return enabled_; }

  // Enable this data type (default is enabled)
  void enable() { enabled_ = true; }

  // Disable this data type
  void disable() { enabled_ = false; }

  // Divide accum by counts to conver to the normalized mean
  void normalize();

 protected:
  // Flag for whether this data type is enabled
  bool enabled_ = true;
  
  // Accumulated data
  stringmap_t<T> data_;

  // Number of datum that have been accumulated
  stringmap_t<size_t> counts_;

  // Flag for whether the accumulated data has been divided
  // by the count
  bool normalized_ = false;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
stringmap_t<T>& AverageData<T>::value() {
  normalize();
  return data_;
}

template <typename T>
void AverageData<T>::normalize() {
  if (normalized_)
    return;

  for (auto& pair : data_) {
    const auto& count = counts_[pair.first];
    if (count > 1) {
      Linalg::idiv(pair.second, double(count));
    }
  }
  normalized_ = true;
}

template <typename T>
void AverageData<T>::clear() {
  data_.clear();
  counts_.clear();
  normalized_ = false;
}

template <typename T>
void AverageData<T>::combine(AverageData<T> &&other) noexcept {
  if (!enabled_)
    return;

  for (auto& pair: other.data_) {
    const auto& key = pair.first;
    // If empty we copy data without accumulating
    if (data_.find(pair.first) == data_.end()) {
      counts_[key] = other.counts_[key];
      data_[key] = std::move(pair.second);
    } else {
      counts_[key] += other.counts_[key];
      Linalg::iadd(data_[key], std::move(pair.second));
    }
  }
  // Now that we have moved we clear the other to initial state.
  other.clear();
}

template <typename T>
void AverageData<T>::add_average_data(const std::string &key, const T &datum) {
  if (!enabled_)
    return;

  if (data_.find(key) == data_.end()) {
    data_[key] = datum;
    counts_[key] = 1;
  } else {
    Linalg::iadd(data_[key], datum);
    counts_[key] += 1;
  }
}

template <typename T>
void AverageData<T>::add_average_data(const std::string& key, T &&datum) noexcept {
  if (!enabled_)
    return;

  if (data_.find(key) == data_.end()) {
    data_[key] = std::move(datum);
    counts_[key] = 1;
  } else {
    Linalg::iadd(data_[key], std::move(datum));
    counts_[key] += 1;
  }
}


template <typename T>
void AverageData<T>::add_to_json(json_t& js) {
  if (!enabled_)
    return;
  for (auto &pair : value()) {
    js[pair.first] = pair.second;
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
