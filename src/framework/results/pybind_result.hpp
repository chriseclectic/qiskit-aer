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

#ifndef _aer_framework_results_pybind_result_hpp_
#define _aer_framework_results_pybind_result_hpp_

#include <pybind11/pybind11.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include "framework/pybind_json.hpp"
#include "framework/results/result.hpp"

namespace AerToPy {

//============================================================================
// Pybind11 conversion of Data containers
//============================================================================

// Move an arbitrary object to Python by calling Pybind11 cast with move
// Template specialization is used with this function for adding custom
// conversion for other types
// NOTE: Can this function be replaced by overload py::cast for custom types?
template <typename T> py::object to_python(T &&obj);

// Move a JSON object to Python
template <> py::object to_python(json_t &&obj);
template <> py::object to_python(json_t &obj);
template <> py::object to_python(const json_t &obj);

// Move a matrix to Python via conversion to Numpy array
template <typename T> py::object to_python(matrix<T> &&obj);

// Move a vector to Python. If the vector is complex or real float/double
// it will be converted to a Numpy array
template <typename T> py::object to_python(std::vector<T> &&obj);

//------------------------------------------------------------------------------
// Convert To Numpy Arrays
//------------------------------------------------------------------------------

// Convert a matrix to a 2D Numpy array in Fortan order
template <typename T>
py::array_t<T, py::array::f_style> to_numpy(matrix<T> &&obj);

// Convert a vector to a 1D Numpy array
template <typename T>
py::array_t<T> to_numpy(std::vector<T> &&obj);

template <> py::object to_python(std::vector<AER::int_t> &&obj);
template <> py::object to_python(std::vector<AER::uint_t> &&obj);
template <> py::object to_python(std::vector<float> &&obj);
template <> py::object to_python(std::vector<std::complex<double>> &&obj);
template <> py::object to_python(std::vector<std::complex<float>> &&obj);

//------------------------------------------------------------------------------
// Circuit Result Data
//------------------------------------------------------------------------------

template <> py::object to_python(AER::ExperimentData &&obj);
template <> py::object to_python(AER::ExperimentResult &&obj);
template <> py::object to_python(AER::Result &&obj);

template <typename T>
void add_to_python(py::dict &pydata, AER::DataContainer<T> &&data);

template <typename T>
void add_to_python(py::dict &pydata, AER::AverageData<T> &&data);

template <typename T>
void add_to_python(py::dict &pydata, AER::PershotData<T> &&data);

template <typename T>
void add_to_python(py::dict &pydata, AER::OneshotData<T> &&data);

template <typename T>
void add_to_python(py::dict &pydata, AER::ConditionalData<T> &&data);

//============================================================================
// Implementation
//============================================================================

//------------------------------------------------------------------------------
// Basic Types
//------------------------------------------------------------------------------

template <typename T>
py::object to_python(T &&obj) {
  return py::cast(obj, py::return_value_policy::move);
}

template <>
py::object to_python(json_t &&obj) {
  py::object pydata;
  std::from_json(obj, pydata);
  return std::move(pydata);
}

template <>
py::object to_python(const json_t &obj) {
  py::object pydata;
  std::from_json(obj, pydata);
  return std::move(pydata);
}

template <>
py::object to_python(json_t &obj) {
  py::object pydata;
  std::from_json(obj, pydata);
  return std::move(pydata);
}

template <typename T>
py::object to_python(std::vector<T> &&obj) {
  py::list pydata;
  for(auto& elt : obj) {
    pydata.append(to_python(std::move(elt)));
  }
  return std::move(pydata);
}

template <typename T>
py::object to_python(matrix<T> &&obj) {
  return to_numpy(std::move(obj));
}

//------------------------------------------------------------------------------
// Array Types
//------------------------------------------------------------------------------

template <typename T>
py::array_t<T, py::array::f_style> to_numpy(matrix<T> &&src) {
  std::array<py::ssize_t, 2> shape {static_cast<py::ssize_t>(src.GetRows()),
                                    static_cast<py::ssize_t>(src.GetColumns())};
  matrix<T>* src_ptr = new matrix<T>(std::move(src));
  auto capsule = py::capsule(src_ptr, [](void* p) { delete reinterpret_cast<matrix<T>*>(p); });
  return py::array_t<T, py::array::f_style>(shape, src_ptr->data(), capsule);
}

template <typename T>
py::array_t<T> to_numpy(std::vector<T> &&src) {
  // Move entire object to heap (Ensure is moveable!). Memory handled via Python capsule
  std::vector<T>* src_ptr = new std::vector<T>(std::move(src));
  auto capsule = py::capsule(src_ptr, [](void* p) { delete reinterpret_cast<std::vector<T>*>(p); });
  return py::array_t<T>(
    src_ptr->size(),  // shape of array
    src_ptr->data(),  // c-style contiguous strides for vector
    capsule           // numpy array references this parent
  );
}

template <>
py::object to_python(std::vector<AER::int_t> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<AER::uint_t> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<double> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<float> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<std::complex<double>> &&obj) {
  return to_numpy(std::move(obj));
}

template <>
py::object to_python(std::vector<std::complex<float>> &&obj) {
  return to_numpy(std::move(obj));
}

//------------------------------------------------------------------------------
// Data Types
//------------------------------------------------------------------------------

template <typename T>
void add_to_python(py::dict &pydata, AER::DataContainer<T> &&data) {
  
  // Add oneshot data at top level
  add_to_python(pydata, static_cast<AER::OneshotData<T>&&>(data));

  // NOTE: We are treating all these data types as "snapshots" and adding to a
  // snapshot dict directory to be semi-backwards compatible with previous version.
  // In the future these should be flattened out to arbitrary key values in the data
  // dict.
  py::dict snapshots;
  add_to_python(snapshots, static_cast<AER::PershotData<T>&&>(data));
  add_to_python(snapshots, static_cast<AER::AverageData<T>&&>(data));
  add_to_python(snapshots, static_cast<AER::ConditionalData<T>&&>(data));
  if (!snapshots.empty()) {
    if (pydata.contains("snapshots")) {
      for (auto& pair: snapshots) {
        pydata["snapshots"][pair.first] = std::move(pair.second);
      }
    } else {
      pydata["snapshots"] = std::move(snapshots);
    }
  }
}

template <typename T>
void add_to_python(py::dict &pydata, AER::AverageData<T> &&data) {
  if (!data.enabled())
    return;
  for (auto &pair : data.value()) {
    pydata[pair.first.data()] = to_python(std::move(pair.second));
  }
}

template <typename T>
void add_to_python(py::dict &pydata, AER::PershotData<T> &&data) {
  if (!data.enabled())
    return;
  for (auto &pair : data.value()) {
    pydata[pair.first.data()] = to_python(std::move(pair.second));
  }
}

template <typename T>
void add_to_python(py::dict &pydata, AER::OneshotData<T> &&data) {
  if (!data.enabled())
    return;
  for (auto &pair : data.value()) {
    pydata[pair.first.data()] = to_python(std::move(pair.second));
  }
}

template <typename T>
void add_to_python(py::dict &pydata, AER::ConditionalData<T> &&data) {
  if (!data.enabled())
    return;
  for (auto &pair : data.value()) {
    py::dict tmp;
    add_to_python(tmp, std::move(pair.second));
    pydata[pair.first.data()] = std::move(tmp);
  }
}


//------------------------------------------------------------------------------
// Result Types
//------------------------------------------------------------------------------

template <>
py::object to_python(AER::Result &&result) {
  py::dict pydata;
  pydata["qobj_id"] = result.qobj_id;

  pydata["backend_name"] = result.backend_name;
  pydata["backend_version"] = result.backend_version;
  pydata["date"] = result.date;
  pydata["job_id"] = result.job_id;

  pydata["results"] = to_python(std::move(result.results));

  if (result.header.empty() == false) {
    pydata["header"] = to_python(std::move(result.header));
  }
  if (result.metadata.empty() == false) {
    pydata["metadata"] = to_python(std::move(result.metadata));
  }

  pydata["success"] = (result.status == AER::Result::Status::completed);
  switch (result.status) {
    case AER::Result::Status::completed:
      pydata["status"] = "COMPLETED";
      break;
    case AER::Result::Status::partial_completed:
      pydata["status"] = "PARTIAL COMPLETED";
      break;
    case AER::Result::Status::error:
      pydata["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::Result::Status::empty:
      pydata["status"] = "EMPTY";
  }
  return std::move(pydata);
}

template <>
py::object to_python(AER::ExperimentResult &&result) {
  py::dict pydata;

  pydata["shots"] = result.shots;
  pydata["seed_simulator"] = result.seed;

  pydata["data"] = to_python(std::move(result.data));

  pydata["success"] = (result.status == AER::ExperimentResult::Status::completed);
  switch (result.status) {
    case AER::ExperimentResult::Status::completed:
      pydata["status"] = "DONE";
      break;
    case AER::ExperimentResult::Status::error:
      pydata["status"] = std::string("ERROR: ") + result.message;
      break;
    case AER::ExperimentResult::Status::empty:
      pydata["status"] = "EMPTY";
  }
  pydata["time_taken"] = result.time_taken;
  if (result.header.empty() == false) {
    py::object tmp;
    from_json(result.header, tmp);
    pydata["header"] = std::move(tmp);
  }
  if (result.metadata.empty() == false) {
    py::object tmp;
    from_json(result.metadata, tmp);
    pydata["metadata"] = std::move(tmp);
  }
  return std::move(pydata);
}

template <>
py::object to_python(AER::ExperimentData &&data) {
  py::dict pydata;

  // Add containers
  add_to_python(pydata, static_cast<AER::DataContainer<json_t>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<std::complex<double>>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<std::vector<std::complex<float>>>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<std::vector<std::complex<double>>>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<matrix<std::complex<float>>>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<matrix<std::complex<double>>>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<std::map<std::string, double>>&&>(data));
  add_to_python(pydata, static_cast<AER::DataContainer<std::map<std::string, std::complex<double>>>&&>(data));

  // Add measurement data
  if (data.enable_counts_ && !data.counts_.empty()) {
    pydata["counts"] = std::move(data.counts_);
  }
  if (data.enable_memory_ && !data.memory_.empty()) {
    pydata["memory"] = std::move(data.memory_);
  }

  // Add metadata
  if (!data.metadata_.empty()) {
    py::dict metadata;
    for (auto& pair : data.metadata_) {
      metadata[pair.first.data()] = std::move(pair.second);
    }
    pydata["metadata"] = std::move(metadata);
  }
  return std::move(pydata);
}

//------------------------------------------------------------------------------
}  // end namespace AerToPy
//------------------------------------------------------------------------------
#endif
