#ifndef AMREX_SPGEMM_H_
#define AMREX_SPGEMM_H_
#include <AMReX_Config.H>

#include <AMReX_GpuComplex.H>
#include <AMReX_OpenMP.H>
#include <AMReX_SpMatrix.H>

namespace amrex {

namespace detail {

template <typename T, template <typename> class V>
CSR<T,V> spgemm_symbolic (Long nrows, Long ncols, CSR<T,V> const& A, CSR<T,V> const& B)
{
    CSR<T,V> C;

    V<int> row_nnz(nrows);

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        V<Long> marker(ncols, -1);

        // First pass to get nnz per row and fill row_offset
#ifdef AMREX_USE_OMP
#pragma omp for
#endif
        for (Long i = 0; i < nrows; ++i) {
            int count = 0;
            for (auto ap = A.row_offset[i]; ap < A.row_offset[i+1]; ++ap) {
                auto const k = A.col_index[ap];
                for (auto bp = B.row_offset[k]; bp < B.row_offset[k+1]; ++bp) {
                    auto const j = B.col_index[bp];
                    if (marker[j] != i) {
                        marker[j] = i;
                        ++count;
                    }
                }
            }
            row_nnz[i] = count;
        }
    }

    C.row_offset.resize(nrows+1);
    C.row_offset[0] = 0;
    for (Long i = 0; i < nrows; ++i) {
        C.row_offset[i+1] = C.row_offset[i] + row_nnz[i];
    }

    C.nnz = C.row_offset.back();
    C.col_index.resize(C.nnz);
    C.mat.resize(C.nnz, T(0));

#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        V<Long> marker(ncols, -1);
        V<Long> cols;
        cols.reserve(256);

        // Second pass to fill col_index
#ifdef AMREX_USE_OMP
#pragma omp for
#endif
        for (Long i = 0; i < nrows; ++i) {
            cols.clear();
            for (auto ap = A.row_offset[i]; ap < A.row_offset[i+1]; ++ap) {
                auto const k = A.col_index[ap];
                for (auto bp = B.row_offset[k]; bp < B.row_offset[k+1]; ++bp) {
                    auto const j = B.col_index[bp];
                    if (marker[j] != i) {
                        marker[j] = i;
                        cols.push_back(j);
                    }
                }
            }

            std::sort(cols.begin(), cols.end());

            auto cp = C.row_offset[i];
            for (auto j : cols) {
                C.col_index[cp++] = j;
            }
        }
    }

    return C;
}

template <typename T, template <typename> class V>
void spgemm_numeric (Long nrows, Long ncols, CSR<T,V> const& A, CSR<T,V> const& B,
                     CSR<T,V>& C)
{
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
    {
        V<Long> pos(ncols, -1);

        // It's assumed that the current values in C all zeros.

#ifdef AMREX_USE_OMP
#pragma omp for
#endif
        for (Long i = 0; i < nrows; ++i) {
            auto c0 = C.row_offset[i];
            auto c1 = C.row_offset[i+1];

            for (auto p = c0; p < c1; ++p) {
                pos[C.col_index[p]] = p;
            }

            // C(i,:) = sum _k A(i,k) * B(k,:)
            for (auto ap = A.row_offset[i]; ap < A.row_offset[i+1]; ++ap) {
                auto k = A.col_index[ap];
                T const a = A.mat[ap];
                for (auto bp = B.row_offset[k]; bp < B.row_offset[k+1]; ++bp) {
                    auto j = B.col_index[bp];
                    C.mat[pos[j]] += a * B.mat[bp];
                }
            }
        }
    }
}

}

template <typename T, template <typename> class V>
CSR<T,V> SpGEMM (Long nrows, Long ncols, CSR<T,V> const& A, CSR<T,V> const& B)
{
    AMREX_ASSERT(nrows == A.nrows());

#if defined(AMREX_USE_CUDA)

    AMREX_ALWAYS_ASSERT(nrows < Long(std::numeric_limits<int>::max()) &&
                        ncols < Long(std::numeric_limits<int>::max()));

    cusparseHandle_t handle;
    AMREX_CUSPARSE_SAFE_CALL(cusparseCreate(&handle));
    AMREX_CUSPARSE_SAFE_CALL(cusparseSetStream(handle, Gpu::gpuStream()));

    cudaDataType data_type;
    if constexpr (std::is_same_v<T,float>) {
        data_type = CUDA_R_32F;
    } else if constexpr (std::is_same_v<T,double>) {
        data_type = CUDA_R_64F;
    } else if constexpr (std::is_same_v<T,GpuComplex<float>>) {
        data_type = CUDA_C_32F;
    } else if constexpr (std::is_same_v<T,GpuComplex<double>>) {
        data_type = CUDA_C_64F;
    } else {
        amrex::Abort("SpMV: unsupported data type");
    }

    cusparseIndexType_t index_type = CUSPARSE_INDEX_32I; // 64-bit not supported by cusparse.

    CsrIndex<int,V> ciA, ciB, ciC;
    ciA.copyFrom(A);
    ciB.copyFrom(B);
    ciC.row_offset.resize(nrows+1);

    cusparseSpMatDescr_t matA;
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseCreateCsr(&matA, nrows, B.nrows(), A.nnz,
                           (void*)ciA.row_offset.data(),
                           (void*)ciA.col_index.data(),
                           (void*)A.mat.data(),
                           index_type, index_type, CUSPARSE_INDEX_BASE_ZERO,
                           data_type));
    cusparseSpMatDescr_t matB;
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseCreateCsr(&matB, B.nrows(), ncols, B.nnz,
                           (void*)ciB.row_offset.data(),
                           (void*)ciB.col_index.data(),
                           (void*)B.mat.data(),
                           index_type, index_type, CUSPARSE_INDEX_BASE_ZERO,
                           data_type));
    cusparseSpMatDescr_t matC;
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseCreateCsr(&matC, nrows, ncols, 0,
                           (void*)ciC.row_offset.data(), nullptr, nullptr,
                           index_type, index_type, CUSPARSE_INDEX_BASE_ZERO,
                           data_type));

    cusparseSpGEMMDescr_t spgemm_descr;
    AMREX_CUSPARSE_SAFE_CALL( cusparseSpGEMM_createDescr(&spgemm_descr) );

    T alpha = T(1);
    T beta  = T(0);
    cusparseOperation_t op = CUSPARSE_OPERATION_NON_TRANSPOSE;

    std::size_t bufferSize1;
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseSpGEMM_workEstimation(handle, op, op, &alpha, matA, matB,
                                       &beta, matC, data_type,
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       spgemm_descr, &bufferSize1, nullptr));
    auto* dBuffer1 = (void*) The_Arena()->alloc(bufferSize1);
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseSpGEMM_workEstimation(handle, op, op, &alpha, matA, matB,
                                       &beta, matC, data_type,
                                       CUSPARSE_SPGEMM_DEFAULT,
                                       spgemm_descr, &bufferSize1, dBuffer1));

    std::size_t bufferSize2;
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseSpGEMM_compute(handle, op, op, &alpha, matA, matB,
                                &beta, matC, data_type,
                                CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_descr, &bufferSize2, nullptr));
    auto* dBuffer2 = (void*) The_Arena()->alloc(bufferSize2);
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseSpGEMM_compute(handle, op, op, &alpha, matA, matB,
                                &beta, matC, data_type,
                                CUSPARSE_SPGEMM_DEFAULT,
                                spgemm_descr, &bufferSize2, dBuffer2));

    std::int64_t C_num_rows, C_num_cols, C_nnz;
    AMREX_CUSPARSE_SAFE_CALL
        (cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz) );
    AMREX_ALWAYS_ASSERT(C_num_rows == nrows && C_num_cols == ncols);

    ciC.col_index.resize(C_nnz);

    CSR<T,V> C;
    C.resize(nrows, C_nnz);

    AMREX_CUSPARSE_SAFE_CALL
        (cusparseCsrSetPointers(matC, (void*)ciC.row_offset.data(),
                                (void*)ciC.col_index.data(), (void*)C.mat.data()));

    AMREX_CUSPARSE_SAFE_CALL
        (cusparseSpGEMM_copy(handle, op, op, &alpha, matA, matB,
                             &beta, matC, data_type,
                             CUSPARSE_SPGEMM_DEFAULT, spgemm_descr));

    ciC.copyTo(C);

    Gpu::streamSynchronize();
    AMREX_CUSPARSE_SAFE_CALL( cusparseSpGEMM_destroyDescr(spgemm_descr) );
    AMREX_CUSPARSE_SAFE_CALL( cusparseDestroySpMat(matA) );
    AMREX_CUSPARSE_SAFE_CALL( cusparseDestroySpMat(matB) );
    AMREX_CUSPARSE_SAFE_CALL( cusparseDestroySpMat(matC) );
    AMREX_CUSPARSE_SAFE_CALL( cusparseDestroy(handle) );
    The_Arena()->free(dBuffer1);
    The_Arena()->free(dBuffer2);

    return C;

#else

    auto C = detail::spgemm_symbolic(nrows, ncols, A,B);
    detail::spgemm_numeric<T,V>(nrows, ncols, A, B, C);
    return C;

#endif
}

/**
 * \brief Sparse matrix-matrix multiplication
 *
 * \param A First matrix.
 * \param B Second matrix.
 * \param col_partition Column partition of the product.
 *
 * \return Product of matrices `A` and `B`
 */
template <typename T, template <typename> class V>
SpMatrix<T,V> SpGEMM (SpMatrix<T,V> const& A, SpMatrix<T,V> const& B,
                      AlgPartition const& col_partition)
{
//xxxxx    const_cast<SpMatrix<T,V>&>(A).startComm_mm(const_cast<SpMatrix<T,V>&>(B));

    auto nrows = A.numLocalRows();
    auto ncols = col_partition.numLocalRows();
    auto csrC = SpGEMM(nrows, ncols, A.m_csr, B.m_csr);

    SpMatrix<T,V> C(A.partition(), std::move(csrC));
    C.m_col_end = ncols;


    // We should tentatively set m_col_end for C.

    // finish comm

    return C;
}

}

#endif
