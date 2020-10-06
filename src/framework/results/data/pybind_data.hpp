/**
 * This code is part of Qiskit.
 *
 * (C) Copyright IBM 2018, 2019, 2020.
 *
 * This code is licensed under the Apache License, Version 2.0. You may
 * obtain a copy of this license in the LICENSE.txt file in the root directory
 * of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
 *
 * Any modifications or derivative works of this code must retain this
 * copyright notice, and modified files need to carry a notice indicating
 * that they have been altered from the originals.
 */

#ifndef _aer_framework_result_new_data_pybind_data_hpp_
#define _aer_framework_result_new_data_pybind_data_hpp_

#include "framework/pybind_basics.hpp"
#include "framework/results/data/single_data.hpp"
#include "framework/results/data/accum_data.hpp"
#include "framework/results/data/average_data.hpp"
#include "framework/results/data/list_data.hpp"
#include "framework/results/data/data_map.hpp"

namespace AerToPy {

// Move a SingleData object to python
template <typename T>
py::object to_python(AER::SingleData<T> &&src);

// Move an AccumData object to python
template <typename T>
py::object to_python(AER::AccumData<T> &&src);

// Move an AverageData object to python
template <typename T>
py::object to_python(AER::AverageData<T> &&src);

// Move an ListData object to python
template <typename T>
py::object to_python(AER::ListData<T> &&src);

// Move an DataMap object to python
template <template <class> class Data, class T, size_t N>
py::object to_python(AER::DataMap<Data, T, N> &&src);

// Move an DataMap object into an existing Python dict
template <template <class> class Data, class T, size_t N>
void add_to_python(py::dict &pydata, AER::DataMap<Data, T, N> &&src);

// Move an DataMap object into an existing Python dict
template <template <class> class Data, class T>
void add_to_python(py::dict &pydata, AER::DataMap<Data, T, 1> &&src);

} //end namespace AerToPy


//============================================================================
// Implementations
//============================================================================

template <typename T>
py::object AerToPy::to_python(AER::SingleData<T> &&src) {
  return AerToPy::to_python(std::move(src.value()));
}

template <typename T>
py::object AerToPy::to_python(AER::AccumData<T> &&src) {
  return AerToPy::to_python(std::move(src.value()));
}

template <typename T>
py::object AerToPy::to_python(AER::AverageData<T> &&src) {
  return AerToPy::to_python(std::move(src.value()));
}

template <typename T>
py::object AerToPy::to_python(AER::ListData<T> &&src) {
  return AerToPy::to_python(std::move(src.value()));
}

template <template <class> class Data, class T, size_t N>
py::object AerToPy::to_python(AER::DataMap<Data, T, N> &&src) {
  py::dict pydata;
  for (auto& elt : src.value()) {
    pydata[elt.first.data()] = AerToPy::to_python(std::move(elt.second));
  }
  return std::move(pydata);
}

template <template <class> class Data, class T, size_t N>
void AerToPy::add_to_python(py::dict &pydata, AER::DataMap<Data, T, N> &&src) {
  for (auto& elt : src.value()) {
    auto& key = elt.first;
    py::dict item = (pydata.contains(key.data())) 
      ? std::move(pydata[key.data()])
      : py::dict();
    AerToPy::add_to_python(item, std::move(elt.second));
    pydata[key.data()] = std::move(item);
  }
}

template <template <class> class Data, class T>
void AerToPy::add_to_python(py::dict &pydata, AER::DataMap<Data, T, 1> &&src) {
  for (auto& elt : src.value()) {
    pydata[elt.first.data()] = AerToPy::to_python(std::move(elt.second));
  }
}

#endif
