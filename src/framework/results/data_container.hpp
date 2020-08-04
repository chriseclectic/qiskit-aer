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

#ifndef _aer_framework_results_data_container_hpp_
#define _aer_framework_results_data_container_hpp_

#include "framework/json.hpp"
#include "framework/results/oneshot_data.hpp"
#include "framework/results/pershot_data.hpp"
#include "framework/results/average_data.hpp"
#include "framework/results/conditional_data.hpp"
#include "framework/utils.hpp"

namespace AER {

//============================================================================
// Output data class for Qiskit-Aer
//============================================================================

/**************************************************************************
 * Data config options:
 *
 * - "counts" (bool): Return counts object in circuit data [Default: True]
 * - "snapshots" (bool): Return snapshots object in circuit data [Default: True]
 * - "memory" (bool): Return memory array in circuit data [Default: False]
 **************************************************************************/

template <typename T>
class DataContainer :
  public OneshotData<T>,
  public PershotData<T>,
  public AverageData<T>,
  public ConditionalData<T> {

public:

  //----------------------------------------------------------------
  // Add data
  //----------------------------------------------------------------

  using OneshotData<T>::add_oneshot_data;
  using PershotData<T>::add_pershot_data;
  using AverageData<T>::add_average_data;
  using ConditionalData<T>::add_conditional_data;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Set the output data config options
  void set_config(const json_t &config);

  // Empty engine of stored data
  void clear();

  // Serialize engine data to JSON
  void add_to_json(json_t &js);

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  DataContainer<T> &combine(DataContainer<T> &&data); // Move semantics

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Enable additional data types
  bool enable_additional_data_ = true;
};

//============================================================================
// DataContainer Implementations
//============================================================================

template <typename T>
void DataContainer<T>::set_config(const json_t &config) {

  // Check for overall types
  JSON::get_value(enable_additional_data_, "enable_additional_data", config);
  if (!enable_additional_data_) {

    // Oneshot data types
    OneshotData<T>::disable();

    // Pershot data types
    PershotData<T>::disable();

    // Average data types
    AverageData<T>::disable();

    // Conditional data types
    ConditionalData<T>::disable();
  }
}

//------------------------------------------------------------------
// Clear and combine
//------------------------------------------------------------------

template <typename T>
void DataContainer<T>::clear() {
  OneshotData<T>::clear();
  PershotData<T>::clear();
  AverageData<T>::clear();
  ConditionalData<T>::clear();
}

template <typename T>
DataContainer<T> &DataContainer<T>::combine(DataContainer<T> &&other) {
  
  OneshotData<T>::combine(std::move(other));
  PershotData<T>::combine(std::move(other));
  AverageData<T>::combine(std::move(other));
  ConditionalData<T>::combine(std::move(other));

  // Clear any remaining data from other container
  other.clear();

  return *this;
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------

template <typename T>
void DataContainer<T>::add_to_json(json_t& js) {
  OneshotData<T>::add_to_json(js);
  PershotData<T>::add_to_json(js);
  AverageData<T>::add_to_json(js);
  ConditionalData<T>::add_to_json(js);
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
