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

#ifndef _aer_framework_circuit_hpp_
#define _aer_framework_circuit_hpp_

#include "framework/operations.hpp"
#include "framework/opset.hpp"
#include "framework/json.hpp"

namespace AER {

//============================================================================
// Circuit class for Qiskit-Aer
//============================================================================

// A circuit is a list of Ops along with a specification of maximum needed
// qubits, memory bits, and register bits for the input operators.
class Circuit {
public:
  // Circuit operations
  std::vector<Operations::Op> ops;

  // Circuit parameters updated by from ops by set_params
  uint_t num_qubits = 0;        // maximum number of qubits needed for ops
  uint_t num_memory = 0;        // maximum number of memory clbits needed for ops
  uint_t num_registers = 0;     // maximum number of registers clbits needed for ops
  
  // Measurement params
  bool has_conditional = false; // True if any ops are conditional
  bool can_sample = true;       // True if circuit tail contains measure, roerror, barrier.
  size_t first_measure_pos = 0; // Position of first measure instruction

  // Circuit metadata constructed from json QobjExperiment
  std::string name = "";        // Circuit name string
  uint_t shots = 1;             // TODO: remove from circuit class
  uint_t seed;                  // TODO: remove from circuit class
  json_t header;                // TODO: remove from circuit class

  // Constructors
  // The operations constructors automatically calculate the
  // num_qubits, num_memory, num_registers
  // parameters by scanning the input list of ops.
  Circuit() = default;
  Circuit(const std::vector<Operations::Op> &_ops); // copy ops vector
  Circuit(std::vector<Operations::Op> &&_ops); // move ops vector

  //-----------------------------------------------------------------------
  // Set containers
  //-----------------------------------------------------------------------

  // Return the opset for the circuit
  const Operations::OpSet& opset() const {return opset_;}

  // Return the used qubits for the circuit
  const std::set<uint_t>& qubits() const {return qubitset_;}

  // Return the used qubits for the circuit
  const std::set<uint_t>& memory() const {return memoryset_;}

  // Return the used qubits for the circuit
  const std::set<uint_t>& registers() const {return registerset_;}

  //-----------------------------------------------------------------------
  // Utility methods 
  //-----------------------------------------------------------------------

  // Append another circuits ops to current circuit
  Circuit& operator+=(const Circuit& rhs); // Copy append
  Circuit& operator+=(Circuit&& rhs);      // Move append
  Circuit operator+(const Circuit& rhs) const;

  // Append ops vector to circuit
  Circuit& operator+=(const std::vector<Operations::Op> &_ops); // Copy append
  Circuit& operator+=(std::vector<Operations::Op> &&_ops);      // Move append
  Circuit operator+(const std::vector<Operations::Op> &_ops) const;

  // Automatically set the number of qubits, memory, registers, and check
  // for conditionals based on ops
  void set_params();

private:
  Operations::OpSet opset_;      // Set of operation types contained in circuit
  std::set<uint_t> qubitset_;    // Set of qubits used in the circuit
  std::set<uint_t> memoryset_;   // Set of memory bits used in the circuit
  std::set<uint_t> registerset_; // Set of register bits used in the circuit

  // Helper function for combining metadata from source into target
  // Note that this will invalidate the circuit unless the other circuits
  // ops are also appended to the current circuit
  void append_circuit_metadata(const Circuit& source);
};


//============================================================================
// Implementation: Circuit methods
//============================================================================

void Circuit::set_params() {

  // Clear current containers
  opset_ = Operations::OpSet();
  qubitset_.clear();
  memoryset_.clear();
  registerset_.clear();
  can_sample = true;
  first_measure_pos = 0;

  // Check maximum qubit, and register size
  // Memory size is loaded from qobj config
  // Also check if measure sampling is in principle possible
  bool first_measure = true;
  for (size_t i = 0; i < ops.size(); ++i) {
    const auto& op = ops[i];
    has_conditional |= op.conditional;
    opset_.insert(op);
    qubitset_.insert(op.qubits.begin(), op.qubits.end());
    memoryset_.insert(op.memory.begin(), op.memory.end());
    registerset_.insert(op.registers.begin(), op.registers.end());

    // Compute measure sampling check
    if (can_sample) {
      if (first_measure) {
        if (op.type == Operations::OpType::measure ||
            op.type == Operations::OpType::roerror) {
          first_measure = false;
        } else {
          first_measure_pos++;
        }
      } else if((op.type == Operations::OpType::barrier ||
                 op.type == Operations::OpType::measure ||
                 op.type == Operations::OpType::roerror) == false) {
        can_sample = false;
      }
    }
  }

  // Get required number of qubits, memory, registers from set maximums
  // Since these are std::set containers the largest element is the
  // Last element of the (non-empty) container. 
  num_qubits = (qubitset_.empty()) ? 0 : 1 + *qubitset_.rbegin();
  num_memory = (memoryset_.empty()) ? 0 : 1 + *memoryset_.rbegin();
  num_registers = (registerset_.empty()) ? 0 : 1 + *registerset_.rbegin();
}

Circuit::Circuit(const std::vector<Operations::Op> &_ops) {
  ops = _ops;
  set_params();
}

Circuit::Circuit(std::vector<Operations::Op> &&_ops) {
  ops = std::move(_ops);
  set_params();
}

void Circuit::append_circuit_metadata(const Circuit& other) {
  // Get max required parameters
  num_qubits = std::max(num_qubits, other.num_qubits);
  num_memory = std::max(num_memory, other.num_qubits);
  num_registers = std::max(num_registers, other.num_qubits);
  has_conditional = has_conditional || other.has_conditional;

  // Sets of operations
  opset_.insert(other.opset_);
  qubitset_.insert(other.qubitset_.begin(), other.qubitset_.end());
  memoryset_.insert(other.qubitset_.begin(), other.qubitset_.end());
  registerset_.insert(other.qubitset_.begin(), other.qubitset_.end());

  // Check measure sampling condition
  if (first_measure_pos == ops.size()) {
    // No measurement in first circuit so we may use the added
    // circuits value
    first_measure_pos = ops.size() + other.first_measure_pos;
    can_sample = other.can_sample;
  } else if (other.first_measure_pos == 0) {
    // If first circuit has measurement we can't sample unless
    // the other circuit has measurement as its first operation
    can_sample &= other.can_sample;
  } else {
    can_sample = false;
  }
}                     

Circuit& Circuit::operator+=(Circuit&& rhs) {
  std::move(rhs.ops.begin(), rhs.ops.end(), std::back_inserter(ops));
  append_circuit_metadata(rhs);
  return *this;
}

Circuit& Circuit::operator+=(const Circuit& rhs) {
  std::copy(rhs.ops.begin(), rhs.ops.end(), std::back_inserter(ops));
  append_circuit_metadata(rhs);
  return *this;
}

Circuit Circuit::operator+(const Circuit& rhs) const {
  Circuit ret = *this;
  ret += rhs;
  return ret;
}

Circuit& Circuit::operator+=(const std::vector<Operations::Op> &_ops) {
  *this += Circuit(ops);
  return *this;
}

Circuit& Circuit::operator+=(std::vector<Operations::Op> &&_ops) {
  *this += Circuit(std::move(ops));
  return *this;
}

Circuit Circuit::operator+(const std::vector<Operations::Op> &_ops) const {
  Circuit ret = *this;
  ret += Circuit(_ops);
  return ret;
}

//============================================================================
// JSON Converison
//============================================================================

void from_json(const json_t &js, Circuit &circ) {

  // Load instructions
  if (JSON::check_key("instructions", js) == false) {
    throw std::invalid_argument("Invalid Qobj experiment: no \"instructions\" field.");
  }

  std::vector<Operations::Op> ops;
  const json_t &jops = js["instructions"];
  for(auto jop: jops){
    ops.emplace_back(Operations::json_to_op(jop));
  }

  // Construct circuit
  circ = Circuit(std::move(ops));

  // Add additional metadata from circ config
  if (JSON::check_key("header", js)) {
    JSON::get_value(circ.name, "name", js["header"]);
    
    // Copy header
    // TODO: this should be moved to qobj instead
    circ.header = js["header"];
  }
  if (JSON::check_key("config", js)) {
    const json_t& config = js["config"];

    // Get shots and seeds
    JSON::get_value(circ.shots, "shots", config);

    // Check for specified memory slots
    if (JSON::check_key("memory_slots",  config)) {
      uint_t memory_slots = config["memory_slots"];
      if (memory_slots < circ.num_memory) {
        throw std::invalid_argument("Invalid Qobj experiment: not enough memory slots.");
      }
      // override memory slot number
      circ.num_memory = memory_slots;
    }

    // Check for specified n_qubits
    if (JSON::check_key("n_qubits", config)) {
      uint_t n_qubits = config["n_qubits"];
      if (n_qubits < circ.num_qubits) {
        throw std::invalid_argument("Invalid Qobj experiment: n_qubits < instruction qubits.");
      }
      // override qubit number
      circ.num_qubits = n_qubits;
    }
  }
}

//------------------------------------------------------------------------------
} // end namespace AER
//------------------------------------------------------------------------------
#endif
