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

namespace AER {

//============================================================================
// Qobj data structure
//============================================================================

struct Qobj {
  //----------------------------------------------------------------
  // Data
  //----------------------------------------------------------------
  std::string id;                 // Qobj identifier passed to result
  std::string type = "QASM";      // currently we only support QASM
  std::vector<Circuit> circuits;  // List of circuits
  std::vector<uint_t> shots;      // List of circuit shots;

  // Metadata
  json_t config;                          // (optional) qobj level config data
  json_t header;                          // (optional) passed through to result
  std::vector<json_t> circuit_headers;    // (optional) circuit header data
};

//============================================================================
// JSON initialization and deserialization
//============================================================================

// JSON deserialization
inline void from_json(const json_t &js, Qobj &qobj) {

  // Initialize empty Qobj
  qobj = Qobj();

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

  // Check shot number
  uint_t config_shots = 1;
  JSON::get_value(config_shots, "shots", qobj.config);

  const json_t &circs = js["experiments"];
  const size_t num_circs = circs.size();

  // Get qobj shots
  uint_t shots = 1;  // Default value
  JSON::get_value(shots, "shots", config);

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
    
    // Check for shots
    uint_t circ_shots = config_shots;
    if (JSON::check_key("config", circs[i])) {
       JSON::get_value(circ_shots, "shots", circs[i]["config"]);
    }

    // Check for circuit name
    json_t header = json_t::object();
    JSON::get_value(header, "header", circs[i]);

    if (param_table.empty() || param_table[i].empty()) {
      // Non parameterized circuit
      qobj.circuits.emplace_back(std::move(circuit));
      qobj.circuit_headers.emplace_back(std::move(header));
      qobj.shots.emplace_back(circ_shots);
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
        qobj.circuits.push_back(param_circuit);
        qobj.shots.push_back(circ_shots);
        qobj.circuit_headers.push_back(header);
      }
    }
  }
}

//------------------------------------------------------------------------------
}  // namespace AER
//------------------------------------------------------------------------------
#endif