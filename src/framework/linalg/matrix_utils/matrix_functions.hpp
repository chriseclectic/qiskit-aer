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

#ifndef _aer_framework_linalg_matrix_functions_hpp_
#define _aer_framework_linalg_matrix_functions_hpp_

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

#include "framework/linalg/enable_if_numeric.hpp"
#include "framework/linalg/matrix_utils.hpp"
#include "framework/types.hpp"

namespace AER {
namespace Linalg {

//------------------------------------------------------------------------------
// Matrix Functions
//------------------------------------------------------------------------------

// Construct a matrix from a vector of matrix-row vectors
template <class T>
matrix<T> make_matrix(const std::vector<std::vector<T>> &mat);

// Reshape a length column-major vectorized matrix into a square matrix
template <class T>
matrix<T> devectorize_matrix(const std::vector<T> &vec);

// Vectorize a matrix by stacking matrix columns (column-major vectorization)
template <class T>
std::vector<T> vectorize_matrix(const matrix<T> &mat);

// Return the transpose a matrix
template <class T>
matrix<T> transpose(const matrix<T> &A);

// Inplace transpose of a matrix, returns reference to input
template <class T>
matrix<T> &itranspose(matrix<T> &A);

// Return the adjoint (Hermitian-conjugate) of a matrix
template <class T>
matrix<std::complex<T>> dagger(const matrix<std::complex<T>> &A);

// Inplace adjoint of a matrix, returns reference to input
template <class T>
matrix<std::complex<T>> &idagger(matrix<std::complex<T>> &A);

// Return the complex conjugate of a matrix
template <class T>
matrix<std::complex<T>> conjugate(const matrix<std::complex<T>> &A);

// Inplace conjugate of a matrix, returns reference to input
template <class T>
matrix<std::complex<T>> &iconjugate(matrix<std::complex<T>> &A);

// Given a list of matrices for a multiplexer stacks and packs them 0/1/2/...
// into a single 2^control x (2^target x 2^target) cmatrix_t)
// Equivalent to a 2^qubits x 2^target "flat" matrix
template <class T>
matrix<T> stacked_matrix(const std::vector<matrix<T>> &mmat);

// Return a vector containing the diagonal of a matrix
template <class T>
std::vector<T> matrix_diagonal(const matrix<T> &mat);

// Tracing
template <class T>
T trace(const matrix<T> &A);
template <class T>
matrix<T> partial_trace_a(const matrix<T> &rho, size_t dimA);
template <class T>
matrix<T> partial_trace_b(const matrix<T> &rho, size_t dimB);

// Tensor product
template <class T>
matrix<T> tensor_product(const matrix<T> &A, const matrix<T> &B);

template <class T>
matrix<T> unitary_superop(const matrix<T> &mat);

// concatenate
// Returns a matrix that is the concatenation of two matrices A, B
// The matrices must have the same dimensions
// If axis == 0, place rows of B after rows of A (vertical extension)
// If axis == 1, place columns of B after columns of A (horizontal extension)
template <class T>
matrix<T> concatenate(const matrix<T> &A, const matrix<T> &B, uint_t axis);

// split
// Splits A into 2 matrices B and C equal in dimensions
// If axis == 0, split A by rows. A must have an even number of rows.
// If axis == 1, split A by columns. A must have an even number of columns.
template <class T>
void split(const matrix<T> &A, matrix<T> &B, matrix<T> &C, uint_t axis);

// Elementwise matrix multiplication
template <class T>
matrix<T> elementwise_multiplication(const matrix<T> &A, const matrix<T> &B);

// Matrix sum of elements
template <class T>
T sum(const matrix<T> &A);

//==============================================================================
// Implementations
//==============================================================================

template <class T>
matrix<T> devectorize_matrix(const std::vector<T> &vec) {
  size_t dim = std::sqrt(vec.size());
  matrix<T> mat(dim, dim);
  for (size_t col = 0; col < dim; col++)
    for (size_t row = 0; row < dim; row++) {
      mat(row, col) = vec[dim * col + row];
    }
  return mat;
}

template <class T>
std::vector<T> vectorize_matrix(const matrix<T> &mat) {
  std::vector<T> vec;
  vec.resize(mat.size(), 0.);
  size_t nrows = mat.GetRows();
  size_t ncols = mat.GetColumns();
  for (size_t col = 0; col < ncols; col++)
    for (size_t row = 0; row < nrows; row++) {
      vec[nrows * col + row] = mat(row, col);
    }
  return vec;
}

template <class T>
matrix<T> make_matrix(const std::vector<std::vector<T>> &mat) {
  size_t nrows = mat.size();
  size_t ncols = mat[0].size();
  matrix<T> ret(nrows, ncols);
  for (size_t row = 0; row < nrows; row++)
    for (size_t col = 0; col < nrows; col++) {
      ret(row, col) = mat[row][col];
    }
  return ret;
}

template <class T>
matrix<T> transpose(const matrix<T> &A) {
  // Transposes a Matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<T> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = A(i, j);
    }
  }
  return temp;
}

template <class T>
matrix<std::complex<T>> dagger(const matrix<std::complex<T>> &A) {
  // Take the Hermitian conjugate of a complex matrix
  const size_t cols = A.GetColumns(), rows = A.GetRows();
  matrix<std::complex<T>> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(j, i) = std::conj(A(i, j));
    }
  }
  return temp;
}

template <class T>
matrix<std::complex<T>> conjugate(const matrix<std::complex<T>> &A) {
  // Take the complex conjugate of a complex matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<std::complex<T>> temp(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      temp(i, j) = std::conj(A(i, j));
    }
  }
  return temp;
}

template <class T>
matrix<T> stacked_matrix(const std::vector<matrix<T>> &mmat) {
  size_t size_of_controls =
      mmat[0].GetRows();  // or GetColumns, as these matrices are (should be)
                          // square
  size_t number_of_controls = mmat.size();

  // Pack vector of matrices into single (stacked) matrix ... note: matrix dims:
  // rows = (stacked_rows x size_of_controls) where:
  //     stacked_rows is the number of control matrices * the size (#rows or
  //     #columns) of each control matrix size_of_controls is the #rows (or
  //     #columns) of each control matrix
  uint_t stacked_rows =
      number_of_controls *
      size_of_controls;  // Used only for clarity in allocating the matrix

  cmatrix_t stacked_matrix(stacked_rows, size_of_controls);
  for (uint_t row = 0; row < stacked_rows; row++)
    for (uint_t col = 0; col < size_of_controls; col++)
      stacked_matrix(row, col) = {0.0, 0.0};

  for (uint_t mmat_number = 0; mmat_number < mmat.size(); mmat_number++) {
    for (uint_t row = 0; row < size_of_controls; row++) {
      for (uint_t col = 0; col < size_of_controls; col++) {
        stacked_matrix(mmat_number * size_of_controls + row, col) =
            mmat[mmat_number](row, col);
      }
    }
  }
  return stacked_matrix;
}

template <class T>
std::vector<T> matrix_diagonal(const matrix<T> &mat) {
  std::vector<T> vec;
  size_t size = std::min(mat.GetRows(), mat.GetColumns());
  vec.resize(size, 0.);
  for (size_t i = 0; i < size; i++) vec[i] = mat(i, i);
  return vec;
}

template <class T>
matrix<T> &itranspose(matrix<T> &A) {
  // Transposes a Matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = i + 1; j < cols; j++) {
      const auto tmp = A(i, j);
      A(i, j) = A(j, i);
      A(j, i) = tmp;
    }
  }
  return A;
}

template <class T>
matrix<std::complex<T>> &idagger(matrix<std::complex<T>> &A) {
  // Take the Hermitian conjugate of a complex matrix
  const size_t cols = A.GetColumns(), rows = A.GetRows();
  matrix<std::complex<T>> temp(cols, rows);
  for (size_t i = 0; i < rows; i++) {
    A(i, i) = std::conj(A(i, i));
    for (size_t j = i + 1; j < cols; j++) {
      const auto tmp = std::conj(A(i, j));
      A(i, j) = std::conj(A(j, i));
      A(j, i) = tmp;
    }
  }
  return A;
}

template <class T>
matrix<std::complex<T>> &iconjugate(matrix<std::complex<T>> &A) {
  // Take the complex conjugate of a complex matrix
  const size_t rows = A.GetRows(), cols = A.GetColumns();
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      A(i, j) = std::conj(A(i, j));
    }
  }
  return A;
}

template <class T>
T trace(const matrix<T> &A) {
  // Finds the trace of a matrix
  size_t rows = A.GetRows(), cols = A.GetColumns();
  if (rows != cols) {
    throw std::invalid_argument("MU::trace: matrix is not square");
  }
  T temp = 0.0;
  for (size_t i = 0; i < rows; i++) {
    temp = temp + A(i, i);
  }
  return temp;
}

template <class T>
matrix<T> partial_trace_a(const matrix<T> &rho, size_t dimA) {
  // Traces out first system (dimension dimA) of composite Hilbert space
  size_t rows = rho.GetRows(), cols = rho.GetColumns();
  if (rows != cols) {
    throw std::invalid_argument("MU::partial_trace_a: matrix is not square");
  }
  if (rows % dimA != 0) {
    throw std::invalid_argument(
        "MU::partial_trace_a: dim(rho)/dim(system b) is not an integer");
  }
  size_t dimB = rows / dimA;
  matrix<T> rhoB(dimB, dimB);
  T temp = 0.0;
  for (size_t i = 0; i < dimB; i++) {
    for (size_t j = 0; j < dimB; j++) {
      for (size_t k = 0; k < dimA; k++) {
        temp = temp + rho(i + dimB * k, j + dimB * k);
      }
      rhoB(i, j) = temp;
      temp = 0.0;
    }
  }
  return rhoB;
}

template <class T>
matrix<T> partial_trace_b(const matrix<T> &rho, size_t dimB) {
  // Traces out second system (dimension dimB) of composite Hilbert space
  size_t rows = rho.GetRows(), cols = rho.GetColumns();
  if (rows != cols) {
    throw std::invalid_argument("MU::partial_trace_b: matrix is not square");
  }
  if (rows % dimB != 0) {
    throw std::invalid_argument(
        "MU::partial_trace_b: dim(rho)/dim(system a) is not an integer");
  }
  size_t dimA = rows / dimB;
  matrix<T> rhoA(dimA, dimA);
  T temp = 0.0;
  for (size_t i = 0; i < dimA; i++) {
    size_t offsetX = i * dimB;
    for (size_t j = 0; j < dimA; j++) {
      size_t offsetY = j * dimB;
      for (size_t k = 0; k < dimB; k++) {
        temp = temp + rho(offsetX + k, offsetY + k);
      }
      rhoA(i, j) = temp;
      temp = 0.0;
    }
  }
  return rhoA;
}

template <class T>
matrix<T> tensor_product(const matrix<T> &A, const matrix<T> &B) {
  // Works out the TensorProduct of two matricies A tensor B
  // Note that if A is i x j and B is p x q then A \otimes B is an ip x jq
  // rmatrix

  // If A or B is empty it will return the other matrix
  if (A.size() == 0) return B;
  if (B.size() == 0) return A;

  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  size_t rows_new = rows1 * rows2, cols_new = cols1 * cols2, n, m;
  matrix<T> temp(rows_new, cols_new);
  // a11 B, a12 B ... a1j B
  // ai1 B, ai2 B ... aij B
  for (size_t i = 0; i < rows1; i++) {
    for (size_t j = 0; j < cols1; j++) {
      for (size_t p = 0; p < rows2; p++) {
        for (size_t q = 0; q < cols2; q++) {
          n = i * rows2 + p;
          m = j * cols2 + q;  //  0 (0 + 1)  + 1*dimb=2 + (0 + 1 )  (j*dimb+q)
          temp(n, m) = A(i, j) * B(p, q);
        }
      }
    }
  }
  return temp;
}

template <class T>
matrix<T> unitary_superop(const matrix<T> &mat) {
  return tensor_product(conjugate(mat), mat);
}

template <class T>
matrix<T> concatenate(const matrix<T> &A, const matrix<T> &B, uint_t axis) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("Linalg::concatenate: axis must be 0 or 1");
  }
  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  matrix<T> temp = A;
  if (axis == 0) {
    if (cols1 != cols2) {
      throw std::invalid_argument("Linalg::concatenate: axis must be 0 or 1");
    }
    temp.resize(rows1 + rows2, cols1);
    for (size_t i = 0; i < rows2; i++)
      for (size_t j = 0; j < cols1; j++) temp(rows1 + i, j) = B(i, j);
  } else if (axis == 1) {
    if (rows1 != rows2) {
      throw std::invalid_argument(
          "Linalg::concatenate: the 2 matrices have a different number of "
          "rows");
    }
    temp.resize(rows1, cols1 + cols2);
    for (size_t i = 0; i < rows1; i++)
      for (size_t j = 0; j < cols2; j++) temp(i, cols1 + j) = B(i, j);
  }
  return temp;
}

template <class T>
void split(const matrix<T> &A, matrix<T> &B, matrix<T> &C, uint_t axis) {
  if (axis != 0 && axis != 1) {
    throw std::invalid_argument("Linalg::split: axis must be 0 or 1");
  }
  size_t rows = A.GetRows(), cols = A.GetColumns();
  matrix<T> temp = A;
  if (axis == 0) {
    if (rows % 2 != 0) {
      throw std::invalid_argument(
          "Linalg::split: can't split matrix A by rows");
    }
    B.resize(rows / 2, cols);
    C.resize(rows / 2, cols);
    for (size_t i = 0; i < rows / 2; i++) {
      for (size_t j = 0; j < cols; j++) {
        B(i, j) = A(i, j);
        C(i, j) = A(i + rows / 2, j);
      }
    }
  } else if (axis == 1) {
    if (cols % 2 != 0) {
      throw std::invalid_argument(
          "Linalg::split: can't split matrix A by columns");
    }
    B.resize(rows, cols / 2);
    C.resize(rows, cols / 2);
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols / 2; j++) {
        B(i, j) = A(i, j);
        C(i, j) = A(i, j + cols / 2);
      }
    }
  }
}

template <class T>
matrix<T> elementwise_multiplication(const matrix<T> &A, const matrix<T> &B) {
  // Works out an elementwise multiplication of two matrices A, B
  // If A or B is empty it will return the other matrix
  size_t rows1 = A.GetRows(), rows2 = B.GetRows(), cols1 = A.GetColumns(),
         cols2 = B.GetColumns();
  if (rows1 != rows2 || cols1 != cols2) {
    throw std::invalid_argument(
        "Linalg::elementwise_multiplication: matrices have different sizes");
  }
  matrix<T> temp(rows1, cols1);
  for (size_t i = 0; i < rows1; i++)
    for (size_t j = 0; j < cols1; j++) temp(i, j) = A(i, j) * B(i, j);
  return temp;
}

template <class T>
T sum(const matrix<T> &A) {
  T temp = 0;
  for (uint_t i = 0; i < A.size(); i++) temp += A[i];
  return temp;
}

//------------------------------------------------------------------------------
}  // end namespace Linalg
//------------------------------------------------------------------------------
}  // end namespace AER
//------------------------------------------------------------------------------
#endif