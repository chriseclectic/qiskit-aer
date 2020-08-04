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

#ifndef _aer_framework_results_oneshot_data_hpp_
#define _aer_framework_results_oneshot_data_hpp_

#include "framework/json.hpp"
#include "framework/types.hpp"

namespace AER {

template <typename T>
class OneshotData {
 public:
  // Access data
  stringmap_t<T>& value() { return data_; }

  // Add a new shot of data by appending to data vector
  // Uses copy semantics
  void add_oneshot_data(const std::string& key, const T& datum);

  // Add a new shot of data by appending to data vector
  // Uses move semantics
  void add_oneshot_data(const std::string& key, T&& datum) noexcept;

  // Combine with another pershot data by concatinating
  // Uses move semantics
  void combine(OneshotData<T>&& other) noexcept;

  // Add to a JSON object
  void add_to_json(json_t& js);

  // Clear all stored data
  void clear() { data_.clear(); }

  // Return True if this data type is enabled
  bool enabled() { return enabled_; }

  // Enable this data type (default is enabled)
  void enable() { enabled_ = true; }

  // Disable this data type
  void disable() { enabled_ = false; }

 protected:
  // Flag for whether this data type is enabled
  bool enabled_ = true;

  // Internal Storage
  stringmap_t<T> data_;
};

//------------------------------------------------------------------------------
// Implementation
//------------------------------------------------------------------------------

template <typename T>
void OneshotData<T>::combine(OneshotData<T> &&other) noexcept {
  if (!enabled_)
    return;
  
  for (auto& pair: other.data_) {
    data_[pair.first] = std::move(pair.second);
  }
  other.clear();
}

template <typename T>
void OneshotData<T>::add_oneshot_data(const std::string &key, const T &datum) {
  if (!enabled_)
    return;
  data_[key] = datum;
}

template <typename T>
void OneshotData<T>::add_oneshot_data(const std::string& key, T &&datum) noexcept {
  if (!enabled_)
    return;
  data_[key] = std::move(datum);
}

template <typename T>
void OneshotData<T>::add_to_json(json_t& js) {
  if (!enabled_)
    return;
  for (auto &pair : data_) {
    js[pair.first] = pair.second;
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
