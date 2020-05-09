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

#ifndef _aer_framework_qobj_hpp_
#define _aer_framework_qobj_hpp_

#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "framework/circuit.hpp"
#include "noise/noise_model.hpp"

namespace AER {

//============================================================================
// Qobj data structure
//============================================================================

struct Qobj {
  // Experiment Data
  std::vector<Circuit> circuits;        // Experiment circuits
  Noise::NoiseModel noise_model;        // (optional) Noise model for execution
  json_t config;                        // (optional) qobj level config data

  // Metadata
  std::string id;                       // qobj identifier passed to result
  std::string type = "QASM";            // currently we only support QASM
  json_t header;                        // (optional) passed through to result
};

//============================================================================
// JSON initialization and deserialization
//============================================================================

void from_json(const json_t &js,  Qobj &qobj) {
  // Check required fields
  if (JSON::get_value(qobj.id, "qobj_id", js) == false) {
    throw std::invalid_argument(R"(Invalid qobj: no "qobj_id" field)");
  };
  JSON::get_value(qobj.type, "type", js);
  if (qobj.type != "QASM") {
    throw std::invalid_argument(R"(Invalid qobj: "type" != "QASM".)");
  };
  if (JSON::check_key("experiments", js) == false) {
    throw std::invalid_argument(R"(Invalid qobj: no "experiments" field.)");
  }

  // Get header and config;
  JSON::get_value(qobj.config, "config", js);
  JSON::get_value(qobj.header, "header", js);

  // Get noise model
  JSON::get_value(qobj.noise_model, "noise_model", qobj.config);

  // Check for fixed simulator seed
  // If seed is negative a random seed will be chosen for each
  // experiment. Otherwise each experiment will be set to a fixed
  // (but different) seed.
  int_t seed = -1;
  uint_t seed_shift = 0;
  JSON::get_value(seed, "seed_simulator", qobj.config);
  const json_t &circs = js["experiments"];
  const size_t num_circs = circs.size();

  // Get qobj shots
  uint_t shots = 1;  // Default value
  JSON::get_value(shots, "shots", qobj.config);

  // Check if parameterized qobj
  // It should be of the form
  // [exp0_params, exp1_params, ...]
  // where:
  //    expk_params = [((i, j), pars), ....]
  //    i is the instruction index in the experiment
  //    j is the param index in the instruction
  //    pars = [par0, par1, ...] is a list of different parameterizations
  using pos_t = std::pair<uint_t, uint_t>;
  using exp_params_t = std::vector<std::pair<pos_t, std::vector<double>>>;
  std::vector<exp_params_t> param_table;
  JSON::get_value(param_table, "parameterizations", qobj.config);

  // Validate parameterizations for number of circuis
  if (!param_table.empty() && param_table.size() != num_circs) {
    throw std::invalid_argument(
        R"(Invalid parameterized qobj: "parameterizations" length does not match number of circuits.)");
  }

  // Load circuits
  for (size_t i=0; i<num_circs; i++) {
    // Get base circuit from qobj
    Circuit circuit = circs[i];

    // Get shots from qobj config
    // This will be overriden if specified in circuit config
    circuit.shots = shots;

    // Set random circuit seed
    circuit.seed = std::random_device()();

    if (param_table.empty() || param_table[i].empty()) {
      // Non parameterized circuit
      qobj.circuits.push_back(circuit);
    } else {
      // Load different parameterizations of the initial circuit
      const auto circ_params = param_table[i];
      const size_t num_params = circ_params[0].second.size();
      const size_t num_instr = circuit.ops.size();
      for (size_t j=0; j<num_params; j++) {
        // Make a copy of the initial circuit
        Circuit param_circuit = circuit;
        for (const auto &params : circ_params) {
          const auto instr_pos = params.first.first;
          const auto param_pos = params.first.second;
          // Validation
          if (instr_pos >= num_instr) {
            throw std::invalid_argument(R"(Invalid parameterized qobj: instruction position out of range)");
          }
          auto &op = param_circuit.ops[instr_pos];
          if (param_pos >= op.params.size()) {
            throw std::invalid_argument(R"(Invalid parameterized qobj: instruction param position out of range)");
          }
          if (j >= params.second.size()) {
            throw std::invalid_argument(R"(Invalid parameterized qobj: parameterization value out of range)");
          }
          // Update the param
          op.params[param_pos] = params.second[j];
        }
        param_circuit.seed = std::random_device()();
        qobj.circuits.push_back(param_circuit);
      }
    }
  }
  // Override random seed with fixed seed if set
  // We shift the seed for each successive experiment
  // So that results aren't correlated between experiments
  if (seed >= 0) {
    for (auto& circuit : qobj.circuits) {
      circuit.seed = seed + seed_shift;
      seed_shift += 2113;  // Shift the seed
    }
  }
}

//------------------------------------------------------------------------------
}  // namespace AER
//------------------------------------------------------------------------------
#endif