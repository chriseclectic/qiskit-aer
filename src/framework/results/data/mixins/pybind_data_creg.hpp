/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_result_data_pybind_data_creg_hpp_
#define _aer_framework_result_data_pybind_data_creg_hpp_

#include "framework/results/data/mixins/data_creg.hpp"
#include "framework/results/data/subtypes/pybind_data_map.hpp"

//------------------------------------------------------------------------------
// Aer C++ -> Python Conversion
//------------------------------------------------------------------------------

namespace AerToPy {

// Move an DataCReg container object to a new Python dict
py::object to_python(AER::DataCReg &&data);

// Move an DataCReg container object to an existing new Python dict
void add_to_python(py::dict &pydata, AER::DataCReg &&data);

} //end namespace AerToPy


//============================================================================
// Implementations
//============================================================================

py::object AerToPy::to_python(AER::DataCReg &&data) {
  py::dict pydata;
  AerToPy::add_to_python(pydata, std::move(data));
  return std::move(pydata);
}

void AerToPy::add_to_python(py::dict &pydata, AER::DataCReg &&data) {
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::ListData, std::string, 1>&&>(data));
  AerToPy::add_to_python(pydata, static_cast<AER::DataMap<AER::AccumData, AER::uint_t, 2>&&>(data));
}

#endif
