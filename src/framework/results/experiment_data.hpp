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

#ifndef _aer_framework_results_experiment_data_hpp_
#define _aer_framework_results_experiment_data_hpp_

#include "framework/json.hpp"
#include "framework/results/oneshot_data.hpp"
#include "framework/results/pershot_data.hpp"
#include "framework/results/average_data.hpp"
#include "framework/results/conditional_data.hpp"
#include "framework/results/data_container.hpp"
#include "framework/results/generic_data_container.hpp"
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

class ExperimentData :
  public GenericDataContainer,
  public DataContainer<std::complex<double>>,
  public DataContainer<std::vector<std::complex<float>>>,
  public DataContainer<std::vector<std::complex<double>>>,
  public DataContainer<matrix<std::complex<float>>>,
  public DataContainer<matrix<std::complex<double>>>,
  public DataContainer<std::map<std::string, double>>,
  public DataContainer<std::map<std::string, std::complex<double>>> {

public:
  //----------------------------------------------------------------
  // Measurement Data
  //----------------------------------------------------------------

  // Add a single memory value to the counts map
  void add_memory_count(const std::string &memory);

  // Add a single memory value to the memory vector
  void add_pershot_memory(const std::string &memory);

  //----------------------------------------------------------------
  // Metadata
  //----------------------------------------------------------------

  // Access metadata map
  stringmap_t<json_t> &metadata() { return metadata_; }
  const stringmap_t<json_t> &metadata() const { return metadata_; }

  // Add new data to metadata at the specified key.
  // This will use the json conversion method `to_json` for data type T.
  // If they key already exists this will update the current data
  // with the new data.
  template <typename T>
  void add_metadata(const std::string &key, T &&data);

  //----------------------------------------------------------------
  // Additional Data
  //----------------------------------------------------------------

  // Additional data comes from the parent DataContainer methods
  using GenericDataContainer::add_oneshot_data;
  using GenericDataContainer::add_pershot_data;
  using GenericDataContainer::add_average_data;
  using GenericDataContainer::add_conditional_data;
  
  using DataContainer<std::complex<double>>::add_oneshot_data;
  using DataContainer<std::complex<double>>::add_pershot_data;
  using DataContainer<std::complex<double>>::add_average_data;
  using DataContainer<std::complex<double>>::add_conditional_data;

  using DataContainer<std::vector<std::complex<double>>>::add_oneshot_data;
  using DataContainer<std::vector<std::complex<double>>>::add_pershot_data;
  using DataContainer<std::vector<std::complex<double>>>::add_average_data;
  using DataContainer<std::vector<std::complex<double>>>::add_conditional_data;

  using DataContainer<std::vector<std::complex<float>>>::add_oneshot_data;
  using DataContainer<std::vector<std::complex<float>>>::add_pershot_data;
  using DataContainer<std::vector<std::complex<float>>>::add_average_data;
  using DataContainer<std::vector<std::complex<float>>>::add_conditional_data;

  using DataContainer<matrix<std::complex<double>>>::add_oneshot_data;
  using DataContainer<matrix<std::complex<double>>>::add_pershot_data;
  using DataContainer<matrix<std::complex<double>>>::add_average_data;
  using DataContainer<matrix<std::complex<double>>>::add_conditional_data;

  using DataContainer<matrix<std::complex<float>>>::add_oneshot_data;
  using DataContainer<matrix<std::complex<float>>>::add_pershot_data;
  using DataContainer<matrix<std::complex<float>>>::add_average_data;
  using DataContainer<matrix<std::complex<float>>>::add_conditional_data;

  using DataContainer<std::map<std::string, double>>::add_oneshot_data;
  using DataContainer<std::map<std::string, double>>::add_pershot_data;
  using DataContainer<std::map<std::string, double>>::add_average_data;
  using DataContainer<std::map<std::string, double>>::add_conditional_data;

  using DataContainer<std::map<std::string, std::complex<double>>>::add_oneshot_data;
  using DataContainer<std::map<std::string, std::complex<double>>>::add_pershot_data;
  using DataContainer<std::map<std::string, std::complex<double>>>::add_average_data;
  using DataContainer<std::map<std::string, std::complex<double>>>::add_conditional_data;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Set the output data config options
  void set_config(const json_t &config);

  // Empty engine of stored data
  void clear();

  // Serialize engine data to JSON
  json_t to_json();

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  ExperimentData &combine(ExperimentData &&data); // Move semantics

  //----------------------------------------------------------------
  // Measurement data
  //----------------------------------------------------------------

  // Histogram of memory counts over shots
  std::map<std::string, uint_t> counts_;

  // Memory state for each shot as hex string
  std::vector<std::string> memory_;

  //----------------------------------------------------------------
  // Metadata
  //----------------------------------------------------------------

  // This will be passed up to the experiment_result level
  // metadata field
  stringmap_t<json_t> metadata_;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  bool enable_counts_ = true;
  bool enable_memory_ = false;

  // Enable additional data types to counts and memory
  bool enable_additional_data_ = true;

  // Enable non-templated Data types that must be converted to JSON
  bool enable_generic_data_ = true;

};

//============================================================================
// Implementations
//============================================================================

void ExperimentData::set_config(const json_t &config) {
  JSON::get_value(enable_counts_, "counts", config);
  JSON::get_value(enable_memory_, "memory", config);

  GenericDataContainer::set_config(config);
  DataContainer<std::complex<double>>::set_config(config);
  DataContainer<std::vector<std::complex<double>>>::set_config(config);
  DataContainer<matrix<std::complex<float>>>::set_config(config);
  DataContainer<matrix<std::complex<double>>>::set_config(config);
  DataContainer<std::map<std::string, double>>::set_config(config);
  DataContainer<std::map<std::string, std::complex<double>>>::set_config(config);
}

//------------------------------------------------------------------
// Classical data
//------------------------------------------------------------------

void ExperimentData::add_memory_count(const std::string &memory) {
  // Memory bits value
  if (enable_counts_ && !memory.empty()) {
    counts_[memory] += 1;
  }
}

void ExperimentData::add_pershot_memory(const std::string &memory) {
  // Memory bits value
  if (enable_memory_ && !memory.empty()) {
    memory_.push_back(memory);
  }
}

//------------------------------------------------------------------
// Metadata
//------------------------------------------------------------------

template <typename T>
void ExperimentData::add_metadata(const std::string &key, T &&data) {
  // Use implicit to_json conversion function for T
  json_t jdata = data;
  add_metadata(key, std::move(jdata));
}

template <>
void ExperimentData::add_metadata(const std::string &key, json_t &&data) {
  auto elt = metadata_.find("key");
  if (elt == metadata_.end()) {
    // If key doesn't already exist add new data
    metadata_[key] = std::move(data);
  } else {
    // If key already exists append with additional data
    elt->second.update(data.begin(), data.end());
  }
}

template <>
void ExperimentData::add_metadata(const std::string &key, const json_t &data) {
  auto elt = metadata_.find("key");
  if (elt == metadata_.end()) {
    // If key doesn't already exist add new data
    metadata_[key] = data;
  } else {
    // If key already exists append with additional data
    elt->second.update(data.begin(), data.end());
  }
}

template <>
void ExperimentData::add_metadata(const std::string &key, json_t &data) {
  const json_t &const_data = data;
  add_metadata(key, const_data);
}

//------------------------------------------------------------------
// Clear and combine
//------------------------------------------------------------------

void ExperimentData::clear() {

  // Clear measurement data
  counts_.clear();
  memory_.clear();

  // Clear metadata
  metadata_.clear();

  // Clear additional data types
  GenericDataContainer::clear();
  DataContainer<std::complex<double>>::clear();
  DataContainer<std::vector<std::complex<float>>>::clear();
  DataContainer<std::vector<std::complex<double>>>::clear();
  DataContainer<matrix<std::complex<float>>>::clear();
  DataContainer<matrix<std::complex<double>>>::clear();
  DataContainer<std::map<std::string, double>>::clear();
  DataContainer<std::map<std::string, std::complex<double>>>::clear();
}

ExperimentData &ExperimentData::combine(ExperimentData &&other) {
  // Combine measure
  std::move(other.memory_.begin(), other.memory_.end(),
            std::back_inserter(memory_));

  // Combine counts
  for (auto pair : other.counts_) {
    counts_[pair.first] += pair.second;
  }

  // Combine metadata
  for (auto &pair : other.metadata_) {
    metadata_[pair.first] = std::move(pair.second);
  }

  // Combine all additional data
  GenericDataContainer::combine(std::move(other));
  DataContainer<std::complex<double>>::combine(std::move(other));
  DataContainer<std::vector<std::complex<float>>>::combine(std::move(other));
  DataContainer<std::vector<std::complex<double>>>::combine(std::move(other));
  DataContainer<matrix<std::complex<float>>>::combine(std::move(other));
  DataContainer<matrix<std::complex<double>>>::combine(std::move(other));
  DataContainer<std::map<std::string, double>>::combine(std::move(other));
  DataContainer<std::map<std::string, std::complex<double>>>::combine(std::move(other));

  // Clear any remaining data from other container
  other.clear();

  return *this;
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------

json_t ExperimentData::to_json() {
  // Initialize output as additional data JSON
  json_t js;

  // Add additional data types
  GenericDataContainer::add_to_json(js);
  DataContainer<std::complex<double>>::add_to_json(js);
  DataContainer<std::vector<std::complex<float>>>::add_to_json(js);
  DataContainer<std::vector<std::complex<double>>>::add_to_json(js);
  DataContainer<matrix<std::complex<float>>>::add_to_json(js);
  DataContainer<matrix<std::complex<double>>>::add_to_json(js);
  DataContainer<std::map<std::string, double>>::add_to_json(js);
  DataContainer<std::map<std::string, std::complex<double>>>::add_to_json(js);

  // Measure data
  if (enable_counts_ && counts_.empty() == false) js["counts"] = counts_;
  if (enable_memory_ && memory_.empty() == false) js["memory"] = memory_;

  // Check if data is null (empty) and if so return an empty JSON object
  if (js.is_null()) return json_t::object();
  return js;
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
