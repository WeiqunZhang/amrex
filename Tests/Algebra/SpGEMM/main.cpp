
#include <AMReX_Algebra.H>
#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

using namespace amrex;

int main (int argc, char *argv[])
{
    amrex::Initialize(argc, argv);

#if 0
    // Identity Matrix
    {
        int nrows = 30;
        int ncols = 27;
        AlgPartition rpart(nrows);
        AlgPartition cpart(ncols);
        auto Ir = IdentityMatrix<Real>(rpart);
        auto Ic = IdentityMatrix<Real>(cpart);

        Real lambda = 2.8;
        int nnz_per_row_max = nrows/3;
        auto A = RandomMatrix<Real>(rpart, nrows, ncols, lambda, nnz_per_row_max);

        auto A2 = amrex::SpGEMM(Ir, A, cpart);
        AMREX_ALWAYS_ASSERT(amrex::almostEqual(A,A2));

        A2 = amrex::SpGEMM(A2, Ic, cpart);
        AMREX_ALWAYS_ASSERT(amrex::almostEqual(A,A2));

        auto II = amrex::SpGEMM(Ir, Ir, rpart);
        AMREX_ALWAYS_ASSERT(amrex::almostEqual(II, Ir));
    }

    // Permutation Matrix
    {
        std::random_device rd;
        std::uniform_int_distribution<unsigned> dist(0, std::numeric_limits<unsigned>::max());
        unsigned seed = dist(rd);
        ParallelDescriptor::Bcast(&seed, 1);
        std::mt19937 gen(seed);

        int nrows = 240;
        Gpu::PinnedVector<Long> perm(nrows);
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), gen);

        Gpu::DeviceVector<Long> perm_dv(nrows);
        Gpu::copyAsync(Gpu::hostToDevice, perm.begin(), perm.end(),
                       perm_dv.begin());
        auto const* pp = perm_dv.data();

        SpMatrix<float> P(AlgPartition(nrows), 1);
        P.setVal([=] AMREX_GPU_DEVICE (Long row, Long* col, float* val)
        {
            *col = pp[row];
            *val = 1.0F;
        }, CsrSorted{true});

        auto PT = amrex::transpose(P, P.partition());

        auto PPT = amrex::SpGEMM(P, PT, P.partition());
        auto PTP = amrex::SpGEMM(PT, P, P.partition());
        auto I = IdentityMatrix<float>(P.partition());

        AMREX_ALWAYS_ASSERT(amrex::almostEqual(PPT,PTP));
        AMREX_ALWAYS_ASSERT(amrex::almostEqual(PPT,I));
    }
#endif

    // 1D Laplacian
    {
        // int nrows = 128;
        int nrows = 6;
        SpMatrix<Real> Lap(AlgPartition(nrows), 3);
        Lap.setVal([=] AMREX_GPU_DEVICE (Long row, Long* col, Real* val)
        {
            if (row == 0) {
                col[0] = 0;
                col[1] = 1;
                col[2] = nrows-1;
                val[0] = Real(2);
                val[1] = Real(-1);
                val[2] = Real(-1);
            } else if (row < nrows-1) {
                col[0] = row-1;
                col[1] = row;
                col[2] = row+1;
                val[0] = Real(-1);
                val[1] = Real(2);
                val[2] = Real(-1);
            } else {
                col[0] = 0;
                col[1] = row-1;
                col[2] = row;
                val[0] = Real(-1);
                val[1] = Real(-1);
                val[2] = Real(2);
            }
        }, CsrSorted{true});

        SpMatrix<Real> Lap2(Lap.partition(), 5);
        Lap2.setVal([=] AMREX_GPU_DEVICE (Long row, Long* col, Real* val)
        {
            if (row == 0) {
                col[0] = 0;
                col[1] = 1;
                col[2] = 2;
                col[3] = nrows-2;
                col[4] = nrows-1;
                val[0] = Real(6);
                val[1] = Real(-4);
                val[2] = Real(1);
                val[3] = Real(1);
                val[4] = Real(-4);
            } else if (row == 1) {
                col[0] = 0;
                col[1] = 1;
                col[2] = 2;
                col[3] = 3;
                col[4] = nrows-1;
                val[0] = Real(-4);
                val[1] = Real(6);
                val[2] = Real(-4);
                val[3] = Real(1);
                val[4] = Real(1);
            } else if (row < nrows-2) {
                col[0] = row-2;
                col[1] = row-1;
                col[2] = row;
                col[3] = row+1;
                col[4] = row+2;
                val[0] = Real(1);
                val[1] = Real(-4);
                val[2] = Real(6);
                val[3] = Real(-4);
                val[4] = Real(1);
            } else if (row == nrows-2) {
                col[0] = 0;
                col[1] = row-2;
                col[2] = row-1;
                col[3] = row;
                col[4] = row+1;
                val[0] = Real(1);
                val[1] = Real(1);
                val[2] = Real(-4);
                val[3] = Real(6);
                val[4] = Real(-4);
            } else { // row == nrows-1
                col[0] = 0;
                col[1] = 1;
                col[2] = row-2;
                col[3] = row-1;
                col[4] = row;
                val[0] = Real(-4);
                val[1] = Real(1);
                val[2] = Real(1);
                val[3] = Real(-4);
                val[4] = Real(6);
            }
        }, CsrSorted{true});

        auto LL = amrex::SpGEMM(Lap, Lap, Lap.partition());

        LL.printToFile("LL");

        AMREX_ALWAYS_ASSERT(amrex::almostEqual(LL,Lap2));
    }

#if 0
    // (A*B)^T = B^T * A^T
    {
        int n1 = 75;
        int n2 = 100;
        int n3 = 80;
        AlgPartition pt1(n1);
        AlgPartition pt2(n2);
        AlgPartition pt3(n3);
        Real lambda = 3.4;
        int nnz_per_row_max = 9;
        auto A = RandomMatrix<Real>(pt1, n1, n2, lambda, nnz_per_row_max);
        auto B = RandomMatrix<Real>(pt2, n2, n3, lambda, nnz_per_row_max);
        auto AT = amrex::transpose(A, pt2);
        auto BT = amrex::transpose(B, pt3);
        auto AB = amrex::SpGEMM(A, B, pt3);
        auto ABT = amrex::transpose(AB, pt3);
        auto BTAT = amrex::SpGEMM(BT, AT, pt1);
        AMREX_ALWAYS_ASSERT(amrex::almostEqual(ABT,BTAT));
    }
#endif

    amrex::Finalize();
}
