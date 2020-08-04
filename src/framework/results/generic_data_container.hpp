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

#ifndef _aer_framework_results_generic_data_container_hpp_
#define _aer_framework_results_generic_data_container_hpp_

#include "framework/results/data_container.hpp"

namespace AER {

//============================================================================
// Special JSON Data container that allows implicit conversion to JSON
//============================================================================

class GenericDataContainer :
  public DataContainer<json_t> {

public:

  //----------------------------------------------------------------
  // Add Data with conversion
  //----------------------------------------------------------------

  // Add oneshot data with conversion to JSON
  template <typename T>
  void add_oneshot_data(const std::string &key, T &&datum);

  // Add pershot data with conversion to JSON
  template <typename T>
  void add_pershot_data(const std::string &key, T &&datum);

  // Add average data with conversion to JSON
  template <typename T>
  void add_average_data(const std::string &key, T &&datum);

  // Add conditional data with conversion to JSON
  template <typename T>
  void add_conditional_data(const std::string &key,
                            const std::string& memory,
                            T &&datum);

  //----------------------------------------------------------------
  // Add JSON data
  //----------------------------------------------------------------

  using DataContainer<json_t>::add_oneshot_data;
  using DataContainer<json_t>::add_pershot_data;
  using DataContainer<json_t>::add_average_data;
  using DataContainer<json_t>::add_conditional_data;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Set the output data config options
  void set_config(const json_t &config);

  // Empty engine of stored data
  using DataContainer<json_t>::clear;

  // Serialize engine data to JSON
  using DataContainer<json_t>::add_to_json;

  // Combine engines for accumulating data
  // Second engine should no longer be used after combining
  // as this function should use move semantics to minimize copying
  using DataContainer<json_t>::combine;

  //----------------------------------------------------------------
  // Config
  //----------------------------------------------------------------

  // Enable conversion of arbitrary types to JSON
  bool enable_generic_data_ = true;
};

//============================================================================
// GenericDataContainer Implementations
//============================================================================


void GenericDataContainer::set_config(const json_t &config) {
  DataContainer<json_t>::set_config(config);
  // Allow generic converions of data via JSON  
  JSON::get_value(enable_generic_data_, "enable_generic_data", config);
}

//------------------------------------------------------------------
// Generic Data
//------------------------------------------------------------------
// use implicit to_json conversion function for T

template <typename T>
void GenericDataContainer::add_pershot_data(const std::string &key, T &&datum) {
  if (enable_generic_data_) {
    json_t tmp = datum;
    DataContainer<json_t>::add_pershot_data(key, std::move(datum));
  }
}

template <typename T>
void GenericDataContainer::add_oneshot_data(const std::string &key, T &&datum) {
  if (enable_generic_data_) {
    json_t tmp = datum;
    DataContainer<json_t>::add_oneshot_data(key, std::move(datum));
  }
}

template <typename T>
void GenericDataContainer::add_average_data(const std::string &key, T &&datum) {
  if (enable_generic_data_) {
    json_t tmp = datum;
    DataContainer<json_t>::add_average_data(key, std::move(datum));
  }
}

template <typename T>
void GenericDataContainer::add_conditional_data(const std::string &key,
                                          const std::string &memory,
                                          T &&datum) {
  if (enable_generic_data_) {
    json_t tmp = datum;  // use implicit to_json conversion function for T
    DataContainer<json_t>::add_conditional_data(key, memory, std::move(datum));
  }
}

//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif
