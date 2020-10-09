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

#ifndef _aer_framework_results_experiment_result_hpp_
#define _aer_framework_results_experiment_result_hpp_

#include "framework/results/legacy/experiment_data.hpp"
#include "framework/results/metadata.hpp"

namespace AER {

//============================================================================
// Result container for Qiskit-Aer
//============================================================================

struct ExperimentResult {
public:

  // Status 
  enum class Status {empty, completed, error};

  // Experiment data
  ExperimentData legacy_data; // Legacy data
  uint_t shots;
  uint_t seed;
  double time_taken;

  // Success and status
  Status status = Status::empty;
  std::string message; // error message

  // Metadata
  json_t header;
  Metadata metadata;

  // Serialize engine data to JSON
  json_t to_json();

  // Set the output data config options
  void set_config(const json_t &config) { legacy_data.set_config(config); }

  // Combine stored data
  ExperimentResult& combine(ExperimentResult &&other);
};

//------------------------------------------------------------------------------
// Combine
//------------------------------------------------------------------------------
ExperimentResult& ExperimentResult::combine(ExperimentResult &&other) {
  legacy_data.combine(std::move(other.legacy_data));
  metadata.combine(std::move(other.metadata));
  return *this;
}

//------------------------------------------------------------------------------
// JSON serialization
//------------------------------------------------------------------------------
json_t ExperimentResult::to_json() {
  // Initialize output as additional data JSON
  json_t result;
  result["data"] = legacy_data.to_json();
  result["shots"] = shots;
  result["seed_simulator"] = seed;
  result["success"] = (status == Status::completed);
  switch (status) {
    case Status::completed:
      result["status"] = std::string("DONE");
      break;
    case Status::error:
      result["status"] = std::string("ERROR: ") + message;
      break;
    case Status::empty:
      result["status"] = std::string("EMPTY");
  }
  result["time_taken"] = time_taken;
  if (header.empty() == false)
    result["header"] = header;
  result["metadata"] = metadata.to_json();
  return result;
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
