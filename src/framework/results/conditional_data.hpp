/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_results_conditional_data_hpp_
#define _aer_framework_results_conditional_data_hpp_

#include "framework/json.hpp"
#include "framework/types.hpp"
#include "framework/results/average_data.hpp"

namespace AER {

//------------------------------------------------------------------------------
// Conditional Snapshot data storage class
//------------------------------------------------------------------------------

template <typename T>
class ConditionalData {
  // Inner snapshot data map type
 public:
  // Access data
  stringmap_t<AverageData<T>> &value();

  // Add a new datum to the snapshot at the specified key
  // Uses copy semantics
  void add_conditional_data(const std::string& key,
                            const std::string &memory,
                            const T &datum);

  // Add a new datum to the snapshot at the specified key
  // Uses move semantics
  void add_conditional_data(const std::string& key,
                            const std::string &memory,
                            T &&datum) noexcept;

  // Combine with another snapshot container
  // Uses move semantics
  void combine(ConditionalData<T> &&other) noexcept;

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

  // Internal Storage
  // Inner (AverageData) key is memory
  stringmap_t<AverageData<T>> data_;

  // Flag for whether the accumulated data has been divided
  // by the count
  bool normalized_ = false;
};

//------------------------------------------------------------------------------
// Implementation: ConditionalSnapshot class methods
//------------------------------------------------------------------------------
template <typename T>
stringmap_t<AverageData<T>>& ConditionalData<T>::value() {
  normalize();
  return data_;
}

template <typename T>
void ConditionalData<T>::normalize() {
  if (normalized_)
    return;

  for (auto& pair : data_) {
    pair.second.normalize();
  }
  normalized_ = true;
}

template <typename T>
void ConditionalData<T>::clear() {
  data_.clear();
  normalized_ = false;
}

template <typename T>
void ConditionalData<T>::combine(ConditionalData<T> &&other) noexcept {
  if (!enabled_)
    return;

  for (auto& pair: other.data_) {
    const auto& key = pair.first;
    if (data_.find(pair.first) == data_.end()) {
      data_[key] = std::move(pair.second);
    } else {
      data_[key].combine(std::move(pair.second));
    }
  }
  other.clear();
}

template <typename T>
void ConditionalData<T>::add_conditional_data(const std::string& key,
                                              const std::string &memory,
                                              const T &datum) {
  if (!enabled_)
    return;
  data_[key].add_average_data(memory, datum);
}

template <typename T>
void ConditionalData<T>::add_conditional_data(const std::string& key,
                                              const std::string &memory,
                                              T &&datum) noexcept {
  if (!enabled_)
    return;
  data_[key].add_average_data(memory, std::move(datum));
}


template <typename T>
void ConditionalData<T>::add_to_json(json_t& js) {
  if (!enabled_)
    return;
  for (auto &pair : value()) {
    pair.second.add_to_json(js[pair.first]);
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
