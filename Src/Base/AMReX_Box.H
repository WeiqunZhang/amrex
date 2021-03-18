
#ifndef AMREX_BOX_H_
#define AMREX_BOX_H_
#include <AMReX_Config.H>

#include <AMReX_Algorithm.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_ccse-mpi.H>
#include <AMReX_IntVect.H>
#include <AMReX_IndexType.H>
#include <AMReX_Orientation.H>
#include <AMReX_SPACE.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_Vector.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuControl.H>

#include <iosfwd>

namespace amrex
{
class BoxCommHelper;

/**
* \brief A Rectangular Domain on an Integer Lattice
*
* A Box is an abstraction for defining discrete regions of
* SPACEDIM indexing space.  Boxes have an IndexType, which defines
* IndexType::CELL or IndexType::NODE based points for each direction
* and a low and high INTVECT which defines the lower and upper corners
* of the Box.  Boxes can exist in positive and negative indexing space.
*
* Box is a dimension dependent class, so SPACEDIM must be
* defined as either 1, 2, or 3 when compiling.
*/
class Box
{
    friend MPI_Datatype ParallelDescriptor::Mpi_typemap<Box>::type();
    friend class BoxCommHelper;

public:
    /*
    * \brief The default constructor.  For safety, the constructed Box is
    * invalid and may be tested for validity with ok().
    * DO NOT CHANGE THIS BEHAVIOR!
    */
    AMREX_GPU_HOST_DEVICE
    constexpr Box () noexcept
        : smallend(1),
          bigend(0),
          btype(),
          dimen(amrex::maxDim)
        {}

    //! Construct an empty d-dimensional Box
    AMREX_GPU_HOST_DEVICE
    constexpr Box (Dimension d) noexcept
        : smallend(amrex::convert(IntVect(1),d)),
          bigend(amrex::convert(IntVect(0),d)),
          btype(),
          dimen(d)
        {}

    //! Construct cell-centered type Box.
    AMREX_GPU_HOST_DEVICE
    constexpr Box (const IntVect& small, const IntVect& big, Dimension d = amrex::maxDim) noexcept
        : smallend(amrex::convert(small,d)),
          bigend(amrex::convert(big,d)),
          btype(),
          dimen(d)
        {}

    //! Construct box with specified lengths.
    AMREX_GPU_HOST_DEVICE
    Box (const IntVect& small, const int* vec_len, Dimension d = amrex::maxDim) noexcept
        : smallend(amrex::convert(small,d)),
          bigend(AMREX_D6_DECL(small.vect[0]+vec_len[0]-1,
                               (d > Dimension::one  ) ? small.vect[1]+vec_len[1]-1 : 0,
                               (d > Dimension::two  ) ? small.vect[2]+vec_len[2]-1 : 0,
                               (d > Dimension::three) ? small.vect[3]+vec_len[3]-1 : 0,
                               (d > Dimension::four ) ? small.vect[4]+vec_len[4]-1 : 0,
                               (d > Dimension::five ) ? small.vect[5]+vec_len[5]-1 : 0)),
          btype(),
          dimen(d)
        {}

    /**
    * \brief Construct Box with given type.  small and big are expected
    * to be consistent with given type.
    */
    AMREX_GPU_HOST_DEVICE
    constexpr Box (const IntVect& small, const IntVect& big, const IntVect& typ,
         Dimension d = amrex::maxDim) noexcept
        : smallend(amrex::convert(small,d)),
          bigend(amrex::convert(big,d)),
          btype(amrex::convert(typ,d)),
          dimen(d)
        {}

    //! Construct dimension specific Boxes.
    AMREX_GPU_HOST_DEVICE
    constexpr Box (const IntVect& small, const IntVect& big, IndexType t,
        Dimension d = amrex::maxDim) noexcept
        : smallend(amrex::convert(small,d)),
          bigend(amrex::convert(big,d)),
          btype(amrex::convert(t,d)),
          dimen(d)
        {}

    template <typename T>
    AMREX_GPU_HOST_DEVICE
    explicit Box (Array4<T> const& a) noexcept
        : smallend(AMREX_D6_DECL(a.begin.x,a.begin.y,a.begin.z,0,0,0)),
          bigend  (AMREX_D6_DECL(a.end.x-1,a.end.y-1,a.end.z-1,0,0,0))
        {}

    // dtor, copy-ctor, copy-op=, move-ctor, and move-op= are compiler generated.

    //! Get the smallend of the box.
    AMREX_GPU_HOST_DEVICE
    const IntVect& smallEnd () const& noexcept { return smallend; }
    const IntVect& smallEnd () && = delete;

    //! Returns the coordinate of the low end in the given direction.
    AMREX_GPU_HOST_DEVICE
    int smallEnd (int dir) const& noexcept { return smallend[dir]; }

    //! Get the bigend.
    AMREX_GPU_HOST_DEVICE
    const IntVect& bigEnd () const& noexcept { return bigend; }
    const IntVect& bigEnd () && = delete;

    //! Returns the coordinate of the high end in the given direction.
    AMREX_GPU_HOST_DEVICE
    int bigEnd (int dir) const noexcept { return bigend[dir]; }

    //! Returns the indexing type.
    AMREX_GPU_HOST_DEVICE
    IndexType ixType () const noexcept { return btype; }

    //! Returns the indexing type.
    AMREX_GPU_HOST_DEVICE
    IntVect type () const noexcept { return btype.ixType(); }

    //! Returns the indexing type in the specified direction.
    AMREX_GPU_HOST_DEVICE
    IndexType::CellIndex type (int dir) const noexcept { return btype.ixType(dir); }

    // Return the dimension of the Box.
    AMREX_GPU_HOST_DEVICE
    Dimension dimension () const noexcept { return dimen; }

    // Return the dimension of the Box.
    AMREX_GPU_HOST_DEVICE
    int dimensionInt () const noexcept { return static_cast<int>(dimen); }

    //! Return the length of the Box.
    AMREX_GPU_HOST_DEVICE
    IntVect size () const noexcept
    {
        return IntVect(AMREX_D6_DECL(bigend[0]-smallend[0] + 1,
                                     bigend[1]-smallend[1] + 1,
                                     bigend[2]-smallend[2] + 1,
                                     bigend[3]-smallend[3] + 1,
                                     bigend[4]-smallend[4] + 1,
                                     bigend[5]-smallend[5] + 1));
    }

    //! Return the length of the Box.
    AMREX_GPU_HOST_DEVICE
    IntVect length () const noexcept
    {
        return IntVect(AMREX_D6_DECL(bigend[0]-smallend[0] + 1,
                                     bigend[1]-smallend[1] + 1,
                                     bigend[2]-smallend[2] + 1,
                                     bigend[3]-smallend[3] + 1,
                                     bigend[4]-smallend[4] + 1,
                                     bigend[5]-smallend[5] + 1));
    }

    //! Return the length of the Box in given direction.
    AMREX_GPU_HOST_DEVICE
    int length (int dir) const noexcept { return bigend[dir] - smallend[dir] + 1; }

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,3> length3d () const noexcept {
#if (AMREX_SPACEDIM == 1)
        return {{bigend[0]-smallend[0]+1, 1, 1}};
#elif (AMREX_SPACEDIM == 2)
        return {{bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1, 1}};
#else
        return {{bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1, bigend[2]-smallend[2]+1}};
#endif
    }

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,AMREX_SPACEDIM> lengthnd () const noexcept {
#if (AMREX_SPACEDIM == 1)
        return {bigend[0]-smallend[0]+1};
#elif (AMREX_SPACEDIM == 2)
        return {bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1};
#elif (AMREX_SPACEDIM == 3)
        return {bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1, bigend[2]-smallend[2]+1};
#elif (AMREX_SPACEDIM == 4)
        return {bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1, bigend[2]-smallend[2]+1,
                bigend[3]-smallend[3]+1};
#elif (AMREX_SPACEDIM == 5)
        return {bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1, bigend[2]-smallend[2]+1,
                bigend[3]-smallend[3]+1, bigend[4]-smallend[4]+1};
#elif (AMREX_SPACEDIM == 6)
        return {bigend[0]-smallend[0]+1, bigend[1]-smallend[1]+1, bigend[2]-smallend[2]+1,
                bigend[3]-smallend[3]+1, bigend[4]-smallend[4]+1, bigend[5]-smallend[5]+1};
#endif
    }

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,3> loVect3d () const noexcept {
#if (AMREX_SPACEDIM == 1)
        return {{smallend[0], 0, 0}};
#elif (AMREX_SPACEDIM == 2)
        return {{smallend[0], smallend[1], 0}};
#else
        return {{smallend[0], smallend[1], smallend[2]}};
#endif
    }

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,AMREX_SPACEDIM> loVectnd () const noexcept {
#if (AMREX_SPACEDIM == 1)
        return {smallend[0]};
#elif (AMREX_SPACEDIM == 2)
        return {smallend[0], smallend[1]};
#elif (AMREX_SPACEDIM == 3)
        return {smallend[0], smallend[1], smallend[2]};
#elif (AMREX_SPACEDIM == 4)
        return {smallend[0], smallend[1], smallend[2], smallend[3]};
#elif (AMREX_SPACEDIM == 5)
        return {smallend[0], smallend[1], smallend[2], smallend[3], smallend[4]};
#elif (AMREX_SPACEDIM == 6)
        return {smallend[0], smallend[1], smallend[2], smallend[3], smallend[4], smallend[5]};
#endif
    }

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,3> hiVect3d () const noexcept {
#if (AMREX_SPACEDIM == 1)
        return {{bigend[0], 0, 0}};
#elif (AMREX_SPACEDIM == 2)
        return {{bigend[0], bigend[1], 0}};
#else
        return {{bigend[0], bigend[1], bigend[2]}};
#endif
    }

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,AMREX_SPACEDIM> hiVectnd () const noexcept {
#if (AMREX_SPACEDIM == 1)
        return {bigend[0]};
#elif (AMREX_SPACEDIM == 2)
        return {bigend[0], bigend[1]};
#elif (AMREX_SPACEDIM == 3)
        return {bigend[0], bigend[1], bigend[2]};
#elif (AMREX_SPACEDIM == 4)
        return {bigend[0], bigend[1], bigend[2], bigend[3]};
#elif (AMREX_SPACEDIM == 5)
        return {bigend[0], bigend[1], bigend[2], bigend[3], bigend[4]};
#elif (AMREX_SPACEDIM == 6)
        return {bigend[0], bigend[1], bigend[2], bigend[3], bigend[4], bigend[5]};
#endif
    }

    //! Returns a constant pointer the array of low end coordinates. Useful for calls to FORTRAN.
    AMREX_GPU_HOST_DEVICE
    const int* loVect () const& noexcept { return smallend.getVect(); }
    AMREX_GPU_HOST_DEVICE
    const int* loVect () && = delete;
    //! Returns a constant pointer the array of high end coordinates. Useful for calls to FORTRAN.
    AMREX_GPU_HOST_DEVICE
    const int* hiVect () const& noexcept { return bigend.getVect(); }
    AMREX_GPU_HOST_DEVICE
    const int* hiVect () && = delete;

    //! Returns the coordinate normal to given face.
    AMREX_GPU_HOST_DEVICE
    int operator[] (Orientation face) const noexcept {
        const int dir = face.coordDir();
        return face.isLow() ? smallend[dir] : bigend[dir];
    }

    //! Checks if it is an empty box.
    AMREX_GPU_HOST_DEVICE
    bool isEmpty () const noexcept { return !ok(); }

    //! Checks if it is a proper Box (including a valid type).
    AMREX_GPU_HOST_DEVICE
    bool ok () const noexcept {
        return bigend.allGE(smallend,dimen) && btype.ok();
    }

    //! Returns true if argument is contained within Box.
    AMREX_GPU_HOST_DEVICE
    bool contains (const IntVect& p) const noexcept {
        return p.allGE(smallend,dimen) && p.allLE(bigend,dimen);
    }

    //! Returns true if argument is contained within Box.
    AMREX_GPU_HOST_DEVICE
    bool contains (const Dim3& p) const noexcept {
        return AMREX_D_TERM(p.x >= smallend[0] && p.x <= bigend[0],
                         && p.y >= smallend[1] && p.y <= bigend[1],
                         && p.z >= smallend[2] && p.z <= bigend[2]);
    }

    //! Returns true if argument is contained within Box.
    AMREX_GPU_HOST_DEVICE
#if (AMREX_SPACEDIM == 1)
    bool contains (int i, int, int) const noexcept {
#elif (AMREX_SPACEDIM == 2)
    bool contains (int i, int j, int) const noexcept {
#else
    bool contains (int i, int j, int k) const noexcept {
#endif
        return AMREX_D_TERM(i >= smallend[0] && i <= bigend[0],
                         && j >= smallend[1] && j <= bigend[1],
                         && k >= smallend[2] && k <= bigend[2]);
    }

    /** \brief Returns true if argument is contained within Box.
    * It is an error if the Boxes have different types.
    */
    AMREX_GPU_HOST_DEVICE
    bool contains (const Box& b) const noexcept
    {
	BL_ASSERT(sameType(b) && sameDimension(b));
	return b.smallend.allGE(smallend,dimen) && b.bigend.allLE(bigend,dimen);
    }

    //! Returns true if argument is strictly contained within Box.
    AMREX_GPU_HOST_DEVICE
    bool strictly_contains (const IntVect& p) const noexcept {
        return p.allGT(smallend,dimen) && p.allLT(bigend,dimen);
    }

    /**
    * \brief Returns true if argument is strictly contained within Box.
    * It is an error if the Boxes have different types.
    */
    AMREX_GPU_HOST_DEVICE
    bool strictly_contains (const Box& b) const noexcept
    {
	BL_ASSERT(sameType(b) && sameDimension(b));
	return b.smallend.allGT(smallend,dimen) && b.bigend.allLT(bigend,dimen);
    }

    //! Returns true if argument is strictly contained within Box.
    AMREX_GPU_HOST_DEVICE
#if (AMREX_SPACEDIM == 1)
    bool strictly_contains (int i, int, int) const noexcept {
#elif (AMREX_SPACEDIM == 2)
    bool strictly_contains (int i, int j, int) const noexcept {
#else
    bool strictly_contains (int i, int j, int k) const noexcept {
#endif
        return AMREX_D_TERM(i > smallend[0] && i < bigend[0],
                         && j > smallend[1] && j < bigend[1],
                         && k > smallend[2] && k < bigend[2]);
    }

    /**
    * \brief Returns true if Boxes have non-null intersections.
    * It is an error if the Boxes have different types.
    */
    AMREX_GPU_HOST_DEVICE
    bool intersects (const Box& b) const noexcept {
        AMREX_ASSERT(sameType(b) && sameDimension(b));
        Box isect(*this);
        isect &= b;
        return isect.ok();
    }

    /**
    * \brief Returns true is Boxes same size, ie translates of each other,.
    * It is an error if they have different types.
    */
    AMREX_GPU_HOST_DEVICE
    bool sameSize (const Box& b) const noexcept {
        BL_ASSERT(sameType(b));
        return AMREX_D6_TERM(length(0) == b.length(0),
                          && length(1) == b.length(1),
                          && length(2) == b.length(2),
                          && length(3) == b.length(3),
                          && length(4) == b.length(4),
                          && length(5) == b.length(5));
    }

    //! Returns true if Boxes have same type.
    AMREX_GPU_HOST_DEVICE
    bool sameType (const Box &b) const noexcept { return btype == b.btype; }

    //! Returns true if Boxes have same dimension.
    AMREX_GPU_HOST_DEVICE
    bool sameDimension (const Box& b) const noexcept { return dimen == b.dimen; }

    //! Returns true if Boxes are identical (including type).
    AMREX_GPU_HOST_DEVICE
    bool operator== (const Box& b) const noexcept {
        return smallend == b.smallend && bigend == b.bigend && b.btype == btype && dimen == b.dimen;
    }

    //! Returns true if Boxes differ (including type).
    AMREX_GPU_HOST_DEVICE
    bool operator!= (const Box& b) const noexcept { return !operator==(b); }

    AMREX_GPU_HOST_DEVICE
    bool operator< (const Box& rhs) const noexcept
    {
        return (static_cast<short>(dimen) < static_cast<short>(rhs.dimen)) || ((dimen == rhs.dimen) &&
              ((btype    < rhs.btype   ) || ((btype == rhs.btype) &&
              ((smallend < rhs.smallend) || ((smallend == rhs.smallend) &&
              ((bigend   < rhs.bigend  )) )))));
    }
    AMREX_GPU_HOST_DEVICE
    bool operator <= (const Box& rhs) const noexcept {
        return !(rhs < *this);
    }
    AMREX_GPU_HOST_DEVICE
    bool operator> (const Box& rhs) const noexcept {
        return rhs < *this;
    }
    AMREX_GPU_HOST_DEVICE
    bool operator>= (const Box& rhs) const noexcept {
        return !(*this < rhs);
    }

    //! Returns true if Box is cell-centered in all indexing directions.
    AMREX_GPU_HOST_DEVICE
    bool cellCentered () const noexcept { return !btype.any(); }

    /**
    * \brief Returns the number of points contained in the Box.
    */
    AMREX_GPU_HOST_DEVICE
    Long numPts () const noexcept {
        return AMREX_D6_TERM( static_cast<Long>(length(0)),
                             *static_cast<Long>(length(1)),
                             *static_cast<Long>(length(2)),
                             *static_cast<Long>(length(3)),
                             *static_cast<Long>(length(4)),
                             *static_cast<Long>(length(5)) );
    }

    /**
    * \brief Returns the number of points contained in the Box.
    * This is intended for use only in diagnostic messages.
    */
    AMREX_GPU_HOST_DEVICE
    double d_numPts () const noexcept {
        BL_ASSERT(ok());
        return AMREX_D6_TERM(double(length(0)), *double(length(1)), *double(length(2)),
                            *double(length(3)), *double(length(4)), *double(length(5)));
    }

    /**
    * \brief Return the volume, in indexing space, of region enclosed by
    * this Box.   This is identical to numPts() for CELL centered
    * Box; otherwise, numPts() > volume().
    */
    AMREX_GPU_HOST_DEVICE
    Long volume () const noexcept {
        return AMREX_D6_TERM( static_cast<Long>(length(0)-btype[0]),
                             *static_cast<Long>(length(1)-btype[1]),
                             *static_cast<Long>(length(2)-btype[2]),
                             *static_cast<Long>(length(3)-btype[3]),
                             *static_cast<Long>(length(4)-btype[4]),
                             *static_cast<Long>(length(5)-btype[5]));
    }

    /**
    * \brief Returns length of longest side.  dir is modified to give
    * direction with longest side: 0...SPACEDIM-1. Ignores type.
    */
    AMREX_GPU_HOST_DEVICE
    int longside (int& dir) const noexcept {
        int maxlen = length(0);
        dir = 0;
        for (int i = 1; i < AMREX_SPACEDIM; i++)
        {
            if (length(i) > maxlen)
            {
                maxlen = length(i);
                dir = i;
            }
        }
        return maxlen;
    }

    //! Returns length of longest side.  Ignores type.
    AMREX_GPU_HOST_DEVICE
    int longside () const noexcept {
        int ignore = 0;
        return longside(ignore);
    }

    /**
    * \brief Returns length of shortest side.  dir is modified to give
    *  direction with shortest side: 0...SPACEDIM-1.  Ignores type.
    */
    AMREX_GPU_HOST_DEVICE
    int shortside (int& dir) const noexcept {
        int minlen = length(0);
        dir = 0;
        for (int i = 1; i < AMREX_SPACEDIM; i++)
        {
            if (length(i) < minlen)
            {
                minlen = length(i);
                dir = i;
            }
        }
        return minlen;
    }

    //! Returns length of shortest side.  Ignores type.
    AMREX_GPU_HOST_DEVICE
    int shortside () const noexcept {
        int ignore = 0;
        return shortside(ignore);
    }

    /**
    * \brief Returns offset of point from smallend; i.e.
    * index(smallend) -> 0, bigend would return numPts()-1.
    * Is used in accessing FArrayBox.
    */
    AMREX_GPU_HOST_DEVICE
    Long index (const IntVect& v) const noexcept;

    //! Given the offset, compute IntVect
    AMREX_GPU_HOST_DEVICE
    IntVect atOffset (Long offset) const noexcept;

    AMREX_GPU_HOST_DEVICE
    GpuArray<int,3> atOffset3d (Long offset) const noexcept;

    //! Redefine the small end of the Box.
    AMREX_GPU_HOST_DEVICE
    Box& setSmall (const IntVect& sm) noexcept { smallend = amrex::convert(sm,dimen); return *this; }

    //! Redefine the small end of the Box.
    AMREX_GPU_HOST_DEVICE
    Box& setSmall (int dir, int sm_index) noexcept { smallend.setVal(dir,sm_index); return *this; }

    //! Redefine the big end of the Box.
    AMREX_GPU_HOST_DEVICE
    Box& setBig (const IntVect& bg) noexcept { bigend = amrex::convert(bg,dimen); return *this; }

    //! Redefine the big end of the Box.
    AMREX_GPU_HOST_DEVICE
    Box& setBig (int dir, int bg_index) noexcept { bigend.setVal(dir,bg_index); return *this; }

    /**
    * \brief Set the entire range in a given direction, starting at
    * sm_index with length n_cells.  NOTE: This will yield an
    * illegal Box if n_cells <= 0.
    */
    AMREX_GPU_HOST_DEVICE
    Box& setRange (int dir,
                   int sm_index,
                   int n_cells = 1) noexcept;

    //! Set indexing type
    AMREX_GPU_HOST_DEVICE
    Box& setType (const IndexType& t) noexcept { btype = amrex::convert(t,dimen); return *this; }

    //! Shift this Box nzones indexing positions in coordinate direction dir.
    AMREX_GPU_HOST_DEVICE
    Box& shift (int dir, int nzones) noexcept {
        smallend.shift(dir,nzones);
        bigend.shift(dir,nzones);
        return *this;
    }

    //! Equivalent to b.shift(0,iv[0]).shift(1,iv[1]) ....
    AMREX_GPU_HOST_DEVICE
    Box& shift (const IntVect& iv) noexcept {
        for (int i = 0, n = static_cast<int>(dimen); i < n; ++i) {
            smallend[i] += iv[i];
            bigend[i] += iv[i];
        }
        return *this;
    }

    /**
    * \brief This member shifts the Box by "half" indicies, thereby
    * converting the Box from type CELL to NODE and visa-versa.
    * b.shiftHalf(0,1)  shifts b to the right by 1/2 cells.
    * b.shiftHalf(1,-3) shifts b in the -j direction by 3/2 cells.
    * NOTE: If num_halfs is EVEN the shift is num_halfs/2 full
    * zones and hence will not change the type.
    * This is: b.shifthalf(4) == b.shift(2).
    */
    AMREX_GPU_HOST_DEVICE
    Box& shiftHalf (int dir, int num_halfs) noexcept;

    //! Equivalent to b.shiftHalf(0,iv[0]).shiftHalf(1,iv[1]) ...
    AMREX_GPU_HOST_DEVICE
    Box& shiftHalf (const IntVect& iv) noexcept;

    /**
    * \brief Convert the Box from the current type into the
    * argument type.  This may change the Box coordinates:
    * type CELL -> NODE : increase coordinate by one on high end
    * type NODE -> CELL : reduce coordinate by one on high end
    * other type mappings make no change.
    */
    AMREX_GPU_HOST_DEVICE
    Box& convert (IndexType typ) noexcept;

    /**
    * \brief Convert the Box from the current type into the
    * argument type.  This may change the Box coordinates:
    * type CELL -> NODE : increase coordinate by one on high end
    * type NODE -> CELL : reduce coordinate by one on high end
    * other type mappings make no change.
    */
    AMREX_GPU_HOST_DEVICE
    Box& convert (const IntVect& typ) noexcept;

    //! Convert to NODE type in all directions.
    AMREX_GPU_HOST_DEVICE
    Box& surroundingNodes () noexcept;

    //! Convert to NODE type in given direction.
    AMREX_GPU_HOST_DEVICE
    Box& surroundingNodes (int dir) noexcept;

    AMREX_GPU_HOST_DEVICE
    Box& surroundingNodes (Direction d) noexcept { return surroundingNodes(static_cast<int>(d)); }

    //! Convert to CELL type in all directions.
    AMREX_GPU_HOST_DEVICE
    Box& enclosedCells () noexcept;

    //! Convert to CELL type in given direction.
    AMREX_GPU_HOST_DEVICE
    Box& enclosedCells (int dir) noexcept;

    AMREX_GPU_HOST_DEVICE
    Box& enclosedCells (Direction d) noexcept { return enclosedCells(static_cast<int>(d)); }

    /**
    * \brief Return Box that is intersection of this Box
    * and argument.  The Boxes MUST be of same type.
    */
    AMREX_GPU_HOST_DEVICE
    Box operator& (const Box& rhs) const noexcept { Box lhs(*this); lhs &= rhs; return lhs; }

    //! Intersect this Box with its argument. The Boxes MUST be of the same type.
    AMREX_GPU_HOST_DEVICE
    Box& operator&= (const Box& rhs) noexcept
    {
        BL_ASSERT(sameType(rhs));
        smallend.max(rhs.smallend);
        bigend.min(rhs.bigend);
        return *this;
    }

    /**
    * \brief Modify Box to that of the minimum Box containing both
    * the original Box and the argument.
    * Both Boxes must have identical type.
    */
    AMREX_GPU_HOST_DEVICE
    Box& minBox (const Box& b) noexcept {
        // BoxArray may call this with not ok boxes.  BL_ASSERT(b.ok() && ok());
        BL_ASSERT(sameType(b) && sameDimension(b));
        smallend.min(b.smallend);
        bigend.max(b.bigend);
        return *this;
    }

    //! Shift Box (relative) by given IntVect.
    AMREX_GPU_HOST_DEVICE
    Box& operator+= (const IntVect& v) noexcept {
        for (int i = 0, n = static_cast<int>(dimen); i < n; ++i) {
            smallend[i] += v[i];
            bigend[i] += v[i];
        }
        return *this;
    }

    //! Shift Box (relative) by given IntVect.
    AMREX_GPU_HOST_DEVICE
    Box operator+ (const IntVect& v) const noexcept { Box r(*this); r += v; return r; }

    //! Shift Box (relative) by given IntVect.
    AMREX_GPU_HOST_DEVICE
    Box& operator-= (const IntVect& v) noexcept {
        for (int i = 0, n = static_cast<int>(dimen); i < n; ++i) {
            smallend[i] -= v[i];
            bigend[i] -= v[i];
        }
        return *this;
    }

    //! Shift Box (relative) by given IntVect.
    AMREX_GPU_HOST_DEVICE
    Box operator- (const IntVect& v) const noexcept { Box r(*this); r -= v; return r; }

    /**
    * \brief Chop the Box at the chop_pnt in the dir direction
    * returns one Box, modifies the object Box.
    * The union of the two is the original Box.
    * The modified Box is the low end, the returned Box
    * is the high end.  If type(dir) = CELL, the Boxes are disjoint
    * with the chop_pnt included in the high end (new Box).
    * It is an ERROR if chop_pnt is the low end of the orig Box.
    * If type(dir) = NODE, the chop_pnt is included in both Boxes
    * but is the only point in common.  It is also an error if the
    * chop_pnt is an end node of the Box.
    */
    AMREX_GPU_HOST_DEVICE
    Box chop (int dir, int chop_pnt) noexcept;

    /*
    * \brief Grow Box in all directions by given amount.
    * NOTE: n_cell negative shrinks the Box by that number of cells.
    */
    AMREX_GPU_HOST_DEVICE
    Box& grow (int i) noexcept {
        for (int d = 0, D=static_cast<int>(dimen); d < D; ++d) {
            smallend[d] -= i;
            bigend[d] += i;
        }
        return *this;
    }

    //! Grow Box in each direction by specified amount.
    AMREX_GPU_HOST_DEVICE
    Box& grow (const IntVect& v) noexcept {
        for (int d = 0, D=static_cast<int>(dimen); d < D; ++d) {
            smallend[d] -= v[d];
            bigend[d] += v[d];
        }
        return *this;
    }

    /**
    * \brief Grow the Box on the low and high end by n_cell cells
    * in direction idir.
    */
    AMREX_GPU_HOST_DEVICE
    Box& grow (int idir, int n_cell) noexcept { smallend.shift(idir, -n_cell); bigend.shift(idir, n_cell); return *this; }

    AMREX_GPU_HOST_DEVICE
    Box& grow (Direction d, int n_cell) noexcept { return grow(static_cast<int>(d), n_cell); }

    /**
    * \brief Grow the Box on the low end by n_cell cells in direction idir.
    * NOTE: n_cell negative shrinks the Box by that number of cells.
    */
    AMREX_GPU_HOST_DEVICE
    Box& growLo (int idir, int n_cell = 1) noexcept { smallend.shift(idir, -n_cell); return *this; }

    AMREX_GPU_HOST_DEVICE
    Box& growLo (Direction d, int n_cell = 1) noexcept { return growLo(static_cast<int>(d), n_cell); }

    /**
    * \brief Grow the Box on the high end by n_cell cells in
    * direction idir.  NOTE: n_cell negative shrinks the Box by that
    * number of cells.
    */
    AMREX_GPU_HOST_DEVICE
    Box& growHi (int idir, int n_cell = 1) noexcept { bigend.shift(idir,n_cell); return *this; }

    AMREX_GPU_HOST_DEVICE
    Box& growHi (Direction d, int n_cell = 1) noexcept { return growHi(static_cast<int>(d), n_cell); }

    //! Grow in the direction of the given face.
    AMREX_GPU_HOST_DEVICE
    Box& grow (Orientation face, int n_cell = 1) noexcept {
        int idir = face.coordDir();
        if (face.isLow())
            smallend.shift(idir, -n_cell);
        else
            bigend.shift(idir,n_cell);
        return *this;
    }

    /**
    * \brief Refine Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
    * hi <- (hi+1)*ratio - 1.
    * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
    * hi <- hi*ratio.
    */
    AMREX_GPU_HOST_DEVICE
    Box& refine (int ref_ratio) noexcept {
        return this->refine(IntVect(ref_ratio));
    }

    /*
    * \brief Refine Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
    * hi <- (hi+1)*ratio - 1.
    * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
    * hi <- hi*ratio.
    */
    AMREX_GPU_HOST_DEVICE
    Box& refine (const IntVect& ref_ratio) noexcept;

    /**
    * \brief Coarsen Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
    * hi <- hi/ratio.
    * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
    * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
    * That is, refinement of coarsened Box must contain
    * the original Box.
    */
    AMREX_GPU_HOST_DEVICE
    Box& coarsen (int ref_ratio) noexcept {
        return this->coarsen(IntVect(ref_ratio));
    }

    /**
    * \brief Coarsen Box by given (positive) refinement ratio.
    * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
    * hi <- hi/ratio.
    * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
    * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
    * That is, refinement of coarsened Box must contain
    * the original Box.
    */
    AMREX_GPU_HOST_DEVICE
    Box& coarsen (const IntVect& ref_ratio) noexcept;

    /**
    * \brief Step through the rectangle.  It is a runtime error to give
    * a point not inside rectangle.  Iteration may not be efficient.
    */
    AMREX_GPU_HOST_DEVICE
    void next (IntVect &) const noexcept;

    /**
    * \brief This static member function returns a constant reference to
    * an object of type Box representing the unit box in
    * AMREX_SPACEDIM-dimensional space.
    */
    AMREX_GPU_HOST_DEVICE
    static Box TheUnitBox () noexcept {
        return Box(IntVect::TheZeroVector(),IntVect::TheZeroVector());
    }

    AMREX_GPU_HOST_DEVICE
    bool isSquare() const noexcept;

    AMREX_GPU_HOST_DEVICE
    bool coarsenable(const IntVect& refrat, const IntVect& min_width) const noexcept
    {
        if (!size().allGE(refrat*min_width,dimen)) {
            return false;
        } else {
            Box testBox = *this;
            testBox.coarsen(refrat);
            testBox.refine (refrat);
            return (*this == testBox);
        }
    }

    AMREX_GPU_HOST_DEVICE
    bool coarsenable(int refrat, int min_width=1) const noexcept {
        return coarsenable(IntVect(refrat), IntVect(min_width));
    }

    AMREX_GPU_HOST_DEVICE
    bool coarsenable(const IntVect& refrat, int min_width=1) const noexcept
    {
        return coarsenable(refrat, IntVect(min_width));
    }

    AMREX_GPU_HOST_DEVICE
    void normalize () noexcept
    {
        for (int idim=0, D=static_cast<int>(dimen); idim < D; ++idim) {
            if (this->length(idim) == 0) {
                this->growHi(idim,1);
            }
        }
    }

    AMREX_GPU_HOST_DEVICE
    void removeLastDimension () noexcept
    {
        --dimen;
        smallend[static_cast<int>(dimen)] = 0;
        bigend[static_cast<int>(dimen)] = 0;
    }

    AMREX_GPU_HOST_DEVICE
    Box& makeSlab (int direction, int slab_index) noexcept
    {
        smallend[direction] = slab_index;
        bigend[direction] = slab_index;
        return *this;
    }

    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 lbound (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 ubound (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 begin  (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 end    (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 length (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 max_lbound (Box const&, Box const&) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 max_lbound (Box const&, Dim3 const&) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 min_ubound (Box const&, Box const&) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Dim3 min_ubound (Box const&, Dim3 const&) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE Box minBox (Box const&, Box const&, IndexType) noexcept;

    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE DimN lboundn (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE DimN uboundn (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE DimN beginn  (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE DimN endn    (Box const& box) noexcept;
    AMREX_GPU_HOST_DEVICE friend AMREX_FORCE_INLINE DimN lengthn (Box const& box) noexcept;

private:
    IntVect         smallend;
    IntVect         bigend;
    IndexType       btype;
    Dimension       dimen = amrex::maxDim;
};

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::refine (const IntVect& ref_ratio) noexcept
{
    for (int d=0, D=static_cast<int>(dimen); d < D; ++d) {
        smallend[d] *= ref_ratio[d];
        int shft = 1 - btype[d];
        bigend[d] = (bigend[d]+shft)*ref_ratio[d] - shft;
    }
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::coarsen (const IntVect& ref_ratio) noexcept
{
    if (ref_ratio != 1)
    {
        smallend.coarsen(ref_ratio, dimen);

        if (btype.any())
        {
            for (int dir = 0, D=static_cast<int>(dimen); dir < D; dir++)
            {
                int off = btype[dir] && (bigend[dir]%ref_ratio[dir]);
                bigend[dir] = amrex::coarsen(bigend[dir],ref_ratio[dir]) + off;
            }
        }
        else
        {
            bigend.coarsen(ref_ratio,dimen);
        }
    }

    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::convert (const IntVect& typ) noexcept
{
    BL_ASSERT(typ.allGE(IntVect::TheZeroVector()) && typ.allLE(IntVect::TheUnitVector()));
    for (int d=0, D=static_cast<int>(dimen); d < D; ++d) {
        int shft = typ[d] - btype[d];
        bigend[d] += shft;
        btype.setType(d, static_cast<IndexType::CellIndex>(typ[d]));
    }
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::convert (IndexType t) noexcept
{
    for (int d=0, D=static_cast<int>(dimen); d < D; ++d) {
        int shft = t[d] - btype[d];
        bigend[d] += shft;
        btype.setType(d, static_cast<IndexType::CellIndex>(t[d]));
   }
   return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::surroundingNodes (int dir) noexcept
{
    if (!(btype[dir]))
    {
        bigend.shift(dir,1);
        //
        // Set dir'th bit to 1 = IndexType::NODE.
        //
        btype.set(dir);
    }
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::surroundingNodes () noexcept
{
    for (int d=0, D=static_cast<int>(dimen); d<D; ++d) {
        if (btype[d] == 0) {
            bigend[d] += 1;
            btype.set(d);
        }
    }
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::enclosedCells (int dir) noexcept
{
    if (btype[dir])
    {
        bigend.shift(dir,-1);
        //
        // Set dir'th bit to 0 = IndexType::CELL.
        //
        btype.unset(dir);
    }
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::enclosedCells () noexcept
{
    for (int d=0, D=static_cast<int>(dimen); d<D; ++d) {
        if (btype[d]) {
            bigend[d] -= 1;
            btype.unset(d);
        }
    }
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Long
Box::index (const IntVect& v) const noexcept
{
    Long result = v[0]-smallend[0];
    Long stride = 1L;
    for (int d=1, D=static_cast<int>(dimen); d<D; ++d) {
        stride *= length(d-1);
        result += stride * (v[d]-smallend[d]);
    }
    return result;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
IntVect
Box::atOffset (Long offset) const noexcept
{
#if (AMREX_SPACEDIM == 1)
    return IntVect{static_cast<int>(offset+smallend[0])};
#elif (AMREX_SPACEDIM == 2)
    int xlen = bigend[0]-smallend[0]+1;
    Long j = offset / xlen;
    Long i = offset - j*xlen;
    return IntVect{static_cast<int>(i+smallend[0]),
                   static_cast<int>(j+smallend[1])};
#elif (AMREX_SPACEDIM == 3)
    int xlen = bigend[0]-smallend[0]+1;
    int ylen = bigend[1]-smallend[1]+1;
    Long k = offset / (xlen*ylen);
    Long j = (offset - k*(xlen*ylen)) / xlen;
    Long i = (offset - k*(xlen*ylen)) - j*xlen;
    return IntVect{static_cast<int>(i+smallend[0]),
                   static_cast<int>(j+smallend[1]),
                   static_cast<int>(k+smallend[2])};
#elif (AMREX_SPACEDIM == 4)
    int xlen = bigend[0]-smallend[0]+1;
    int ylen = bigend[1]-smallend[1]+1;
    int zlen = bigend[2]-smallend[2]+1;
    Long jstride = xlen;
    Long kstride = jstride*ylen;
    Long lstride = kstride*zlen;
    Long l = offset / lstride;
    Long k = (offset - l*lstride) / kstride;
    Long j = (offset - l*lstride - k*kstride) / jstride;
    Long i =  offset - l*lstride - k*kstride - j*jstride;
    return {static_cast<int>(i+smallend[0]),
            static_cast<int>(j+smallend[1]),
            static_cast<int>(k+smallend[2]),
            static_cast<int>(l+smallend[3])};
#elif (AMREX_SPACEDIM == 5)
    int xlen = bigend[0]-smallend[0]+1;
    int ylen = bigend[1]-smallend[1]+1;
    int zlen = bigend[2]-smallend[2]+1;
    int ulen = bigend[3]-smallend[3]+1;
    Long jstride = xlen;
    Long kstride = jstride*ylen;
    Long lstride = kstride*zlen;
    Long mstride = lstride*ulen;
    Long m = offset / mstride;
    Long l = (offset - m*mstride) / lstride;
    Long k = (offset - m*mstride - l*lstride) / kstride;
    Long j = (offset - m*mstride - l*lstride - k*kstride) / jstride;
    Long i =  offset - m*mstride - l*lstride - k*kstride - j*jstride;
    return {static_cast<int>(i+smallend[0]),
            static_cast<int>(j+smallend[1]),
            static_cast<int>(k+smallend[2]),
            static_cast<int>(l+smallend[3]),
            static_cast<int>(m+smallend[4])};
#elif (AMREX_SPACEDIM == 6)
    int xlen = bigend[0]-smallend[0]+1;
    int ylen = bigend[1]-smallend[1]+1;
    int zlen = bigend[2]-smallend[2]+1;
    int ulen = bigend[3]-smallend[3]+1;
    int vlen = bigend[4]-smallend[4]+1;
    Long jstride = xlen;
    Long kstride = jstride*ylen;
    Long lstride = kstride*zlen;
    Long mstride = lstride*ulen;
    Long nstride = lstride*vlen;
    Long n = offset / nstride;
    Long m = (offset - n*nstride) / mstride;
    Long l = (offset - n*nstride - m*mstride) / lstride;
    Long k = (offset - n*nstride - m*mstride - l*lstride) / kstride;
    Long j = (offset - n*nstride - m*mstride - l*lstride - k*kstride) / jstride;
    Long i =  offset - n*nstride - m*mstride - l*lstride - k*kstride - j*jstride;
    return {static_cast<int>(i+smallend[0]),
            static_cast<int>(j+smallend[1]),
            static_cast<int>(k+smallend[2]),
            static_cast<int>(l+smallend[3]),
            static_cast<int>(m+smallend[4]),
            static_cast<int>(n+smallend[5])};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
GpuArray<int,3>
Box::atOffset3d (Long offset) const noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {{static_cast<int>(offset+smallend[0]),
             static_cast<int>(0),
             static_cast<int>(0)}};
#elif (AMREX_SPACEDIM == 2)
    int xlen = bigend[0]-smallend[0]+1;
    Long j = offset / xlen;
    Long i = offset - j*xlen;
    return {{static_cast<int>(i+smallend[0]),
             static_cast<int>(j+smallend[1]),
             static_cast<int>(0)}};
#else
    int xlen = bigend[0]-smallend[0]+1;
    int ylen = bigend[1]-smallend[1]+1;
    Long k = offset / (xlen*ylen);
    Long j = (offset - k*(xlen*ylen)) / xlen;
    Long i = (offset - k*(xlen*ylen)) - j*xlen;
    return {{static_cast<int>(i+smallend[0]),
             static_cast<int>(j+smallend[1]),
             static_cast<int>(k+smallend[2])}};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box&
Box::setRange (int dir,
               int sm_index,
               int n_cells) noexcept
{
    smallend.setVal(dir,sm_index);
    bigend.setVal(dir,sm_index+n_cells-1);
    return *this;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
Box::next (IntVect& p) const noexcept
{
    BL_ASSERT(contains(p));

    ++p[0];

#if (AMREX_SPACEDIM >= 2)
    if (p[0] > bigend[0])
    {
	p[0] = smallend[0];
	++p[1];
#if (AMREX_SPACEDIM >= 3)
	if (p[1] > bigend[1])
	{
	    p[1] = smallend[1];
	    ++p[2];
#if (AMREX_SPACEDIM >= 4)
            if (p[2] > bigend[2])
            {
                p[2] = smallend[2];
                ++p[3];
#if (AMREX_SPACEDIM >= 5)
                if (p[3] > bigend[3])
                {
                    p[3] = smallend[3];
                    ++p[4];
#if (AMREX_SPACEDIM == 6)
                    if (p[4] > bigend[4])
                    {
                        p[4] = smallend[4];
                        ++p[5];
                    }
#endif
                }
#endif
            }
#endif
	}
#endif
    }
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
bool
Box::isSquare () const noexcept
{
#if AMREX_SPACEDIM==1
    return false; // can't build a square in 1-D
#else
    const IntVect& sz = this->size();
#if AMREX_SPACEDIM==2
    return (sz[0] == sz[1]);
#elif AMREX_SPACEDIM==3
    return (sz[0] == sz[1] && (sz[1] == sz[2]));
#elif AMREX_SPACEDIM==4
    return (sz[0] == sz[1] && (sz[1] == sz[2]) && (sz[2] == sz[3]));
#elif AMREX_SPACEDIM==5
    return (sz[0] == sz[1] && (sz[1] == sz[2]) && (sz[2] == sz[3]) && sz[3] == sz[4]);
#elif AMREX_SPACEDIM==6
    return (sz[0] == sz[1] && (sz[1] == sz[2]) && (sz[2] == sz[3]) && sz[3] == sz[4] && sz[4] && sz[5]);
#endif
#endif
}

//
// Modified Box is low end, returned Box is high end.
// If CELL: chop_pnt included in high end.
// If NODE: chop_pnt included in both Boxes.
//

AMREX_GPU_HOST_DEVICE
inline
Box
Box::chop (int dir, int chop_pnt) noexcept
{
    //
    // Define new high end Box including chop_pnt.
    //
    IntVect sm(smallend);
    IntVect bg(bigend);
    sm.setVal(dir,chop_pnt);
    if (btype[dir])
    {
        //
        // NODE centered Box.
        //
        BL_ASSERT(chop_pnt > smallend[dir] && chop_pnt < bigend[dir]);
        //
        // Shrink original Box to just contain chop_pnt.
        //
        bigend.setVal(dir,chop_pnt);
    }
    else
    {
        //
        // CELL centered Box.
        //
        BL_ASSERT(chop_pnt > smallend[dir] && chop_pnt <= bigend[dir]);
        //
        // Shrink origional Box to one below chop_pnt.
        //
        bigend.setVal(dir,chop_pnt-1);
    }
    return Box(sm,bg,btype,dimen);
}

AMREX_GPU_HOST_DEVICE
inline
Box&
Box::shiftHalf (int dir, int nzones) noexcept
{
    const int nbit = (nzones<0 ? -nzones : nzones)%2;
    int nshift = nzones/2;
    //
    // Toggle btyp bit if nzones is odd.
    //
    const unsigned int bit_dir = btype[dir];
    if (nbit)
        btype.flip(dir);
    if (nzones < 0)
        nshift -= (bit_dir ? nbit : 0);
    else
        nshift += (bit_dir ? 0 : nbit);
    smallend.shift(dir,nshift);
    bigend.shift(dir,nshift);
    return *this;
}

AMREX_GPU_HOST_DEVICE
inline
Box&
Box::shiftHalf (const IntVect& nz) noexcept
{
    for (int d=0, D=static_cast<int>(dimen); d<D; ++d) {
        shiftHalf(d,nz[d]);
    }
    return *this;
}

class BoxCommHelper
{
public:

    explicit BoxCommHelper (const Box& bx, int* p_ = 0);

    int* data () const& noexcept { return p; }
    int* data () && = delete;

    Box make_box () const noexcept {
	return Box(IntVect(p), IntVect(p+AMREX_SPACEDIM), IntVect(p+2*AMREX_SPACEDIM),
                   static_cast<Dimension>(p[3*AMREX_SPACEDIM]));
    }

    static int size () noexcept { return 3*AMREX_SPACEDIM+1; }

private:
    int* p;
    std::vector<int> v;
};

class BoxConverter {
public:
    virtual Box doit (const Box& fine) const = 0;
    virtual BoxConverter* clone () const = 0;
    virtual ~BoxConverter () = 0;
};

inline BoxConverter::~BoxConverter () { }

void AllGatherBoxes (Vector<Box>& bxs, int n_extra_reserve=0);

/**
 * \brief Grow Box in all directions by given amount.
 * NOTE: n_cell negative shrinks the Box by that number of cells.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box grow (const Box& b, int i) noexcept
{
    Box result = b;
    result.grow(i);
    return result;
}

//! Grow Box in each direction by specified amount.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box grow (const Box& b, const IntVect& v) noexcept
{
    Box result = b;
    result.grow(v);
    return result;
}

//! Grow Box in direction idir be n_cell cells
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box grow (const Box& b, int idir, int n_cell) noexcept
{
    Box result = b;
    result.grow(idir, n_cell);
    return result;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box grow (const Box& b, Direction d, int n_cell) noexcept
{
    return grow(b, static_cast<int>(d), n_cell);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box growLo (const Box& b, int idir, int n_cell) noexcept
{
    Box result = b;
    result.growLo(idir, n_cell);
    return result;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box growLo (const Box& b, Direction d, int n_cell) noexcept
{
    return growLo(b, static_cast<int>(d), n_cell);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box growHi (const Box& b, int idir, int n_cell) noexcept
{
    Box result = b;
    result.growHi(idir, n_cell);
    return result;
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box growHi (const Box& b, Direction d, int n_cell) noexcept
{
    return growHi(b, static_cast<int>(d), n_cell);
}

/**
 * \brief Coarsen Box by given (positive) refinement ratio.
 * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
 * hi <- hi/ratio.
 * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
 * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
 * That is, refinement of coarsened Box must contain
 * the original Box.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box coarsen (const Box& b, int ref_ratio) noexcept
{
    Box result = b;
    result.coarsen(IntVect(ref_ratio));
    return result;
}

/**
 * \brief Coarsen Box by given (positive) refinement ratio.
 * NOTE: if type(dir) = CELL centered: lo <- lo/ratio and
 * hi <- hi/ratio.
 * NOTE: if type(dir) = NODE centered: lo <- lo/ratio and
 * hi <- hi/ratio + ((hi%ratio)==0 ? 0 : 1).
 * That is, refinement of coarsened Box must contain
 * the original Box.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box coarsen (const Box& b, const IntVect& ref_ratio) noexcept
{
    Box result = b;
    result.coarsen(ref_ratio);
    return result;
}

/**
 * \brief Refine Box by given (positive) refinement ratio.
 * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
 * hi <- (hi+1)*ratio - 1.
 * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
 * hi <- hi*ratio.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box refine (const Box& b, int ref_ratio) noexcept
{
    Box result = b;
    result.refine(IntVect(ref_ratio));
    return result;
}

/**
 * \brief Refine Box by given (positive) refinement ratio.
 * NOTE: if type(dir) = CELL centered: lo <- lo*ratio and
 * hi <- (hi+1)*ratio - 1.
 * NOTE: if type(dir) = NODE centered: lo <- lo*ratio and
 * hi <- hi*ratio.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box refine (const Box& b, const IntVect& ref_ratio) noexcept
{
    Box result = b;
    result.refine(ref_ratio);
    return result;
}

//! Return a Box with indices shifted by nzones in dir direction.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box shift (const Box& b, int dir, int nzones) noexcept
{
    Box result = b;
    result.shift(dir, nzones);
    return result;
}

//! Return a Box with indices shifted by nzones.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box shift (const Box& b, const IntVect& nzones) noexcept
{
    Box result = b;
    result.shift(nzones);
    return result;
}

/**
 * \brief Returns a Box with NODE based coordinates in direction dir
 * that encloses Box b.  NOTE: equivalent to b.convert(dir,NODE)
 * NOTE: error if b.type(dir) == NODE.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box surroundingNodes (const Box& b, int dir) noexcept
{
    Box bx(b);
    bx.surroundingNodes(dir);
    return bx;
}

/**
 * \brief Returns a Box with NODE based coordinates in direction dir
 * that encloses Box b.  NOTE: equivalent to b.convert(dir,NODE)
 * NOTE: error if b.type(dir) == NODE.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box surroundingNodes (const Box& b, Direction d) noexcept
{
    return surroundingNodes(b, static_cast<int>(d));
}

/**
 * \brief Returns a Box with NODE based coordinates in all
 * directions that encloses Box b.
*/
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box surroundingNodes (const Box& b) noexcept
{
    Box bx(b);
    bx.surroundingNodes();
    return bx;
}

//! Returns a Box with different type
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box convert (const Box& b, const IntVect& typ) noexcept
{
    Box bx(b);
    bx.convert(typ);
    return bx;
}

//! Returns a Box with different type
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box convert (const Box& b, const IndexType& typ) noexcept
{
    Box bx(b);
    bx.convert(typ);
    return bx;
}

/**
 * \brief Returns a Box with CELL based coordinates in
 * direction dir that is enclosed by b.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box enclosedCells (const Box& b, int dir) noexcept
{
    Box bx(b);
    bx.enclosedCells(dir);
    return bx;
}

/**
 * \brief Returns a Box with CELL based coordinates in
 * direction dir that is enclosed by b.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box enclosedCells (const Box& b, Direction d) noexcept
{
    return enclosedCells(b, static_cast<int>(d));
}

/**
 * \brief Returns a Box with CELL based coordinates in all
 * directions that is enclosed by b.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box enclosedCells (const Box& b) noexcept
{
    Box bx(b);
    bx.enclosedCells();
    return bx;
}

/**
 * \brief Returns the edge-centered Box (in direction dir) defining
 * the low side of Box b.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box bdryLo (const Box& b, int dir, int len=1) noexcept
{
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    int sm = low[dir];
    low.setVal(dir,sm-len+1);
    hi.setVal(dir,sm);
    //
    // set dir'th bit to 1 = IndexType::NODE.
    //
    IndexType typ(b.ixType());
    typ.set(dir);
    return Box(low,hi,typ,b.dimension());
}

/**
 * \brief Returns the edge-centered Box (in direction dir) defining
 * the high side of Box b.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box bdryHi (const Box& b, int dir, int len=1) noexcept
{
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    unsigned int bitval = b.type()[dir];
    int bg              = hi[dir]  + 1 - bitval%2;
    low.setVal(dir,bg);
    hi.setVal(dir,bg+len-1);
    //
    // Set dir'th bit to 1 = IndexType::NODE.
    //
    IndexType typ(b.ixType());
    typ.set(dir);
    return Box(low,hi,typ,b.dimension());
}

/**
 * \brief Similar to bdryLo and bdryHi except that it operates on the
 * given face of  box b.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box bdryNode (const Box& b, Orientation face, int len=1) noexcept
{
    int dir = face.coordDir();
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    if (face.isLow())
    {
        int sm = low[dir];
        low.setVal(dir,sm-len+1);
        hi.setVal(dir,sm);
    }
    else
    {
        unsigned int bitval = b.type()[dir];
        int bg              = hi[dir]  + 1 - bitval%2;
        low.setVal(dir,bg);
        hi.setVal(dir,bg+len-1);
    }
    //
    // Set dir'th bit to 1 = IndexType::NODE.
    //
    IndexType typ(b.ixType());
    typ.set(dir);
    return Box(low,hi,typ,b.dimension());
}

/**
 * \brief Returns the cell centered Box of length len adjacent
 * to b on the low end along the coordinate direction dir.
 * The return Box is identical to b in the other directions.
 * The return Box and b have an empty intersection.
 * NOTE:  len >= 1
 * NOTE:  Box retval = b.adjCellLo(b,dir,len)
 * is equivalent to the following set of operations:
 * Box retval(b);
 * retval.convert(dir,Box::CELL);
 * retval.setrange(dir,retval.smallEnd(dir)-len,len);
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box adjCellLo (const Box& b, int dir, int len=1) noexcept
{
    BL_ASSERT(len > 0);
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    int sm = low[dir];
    low.setVal(dir,sm - len);
    hi.setVal(dir,sm - 1);
    //
    // Set dir'th bit to 0 = IndexType::CELL.
    //
    IndexType typ(b.ixType());
    typ.unset(dir);
    return Box(low,hi,typ,b.dimension());
}

//! Similar to adjCellLo but builds an adjacent Box on the high end.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box adjCellHi (const Box& b, int dir, int len=1) noexcept
{
    BL_ASSERT(len > 0);
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    unsigned int bitval = b.type()[dir];
    int bg              = hi[dir]  + 1 - bitval%2;
    low.setVal(dir,bg);
    hi.setVal(dir,bg + len - 1);
    //
    // Set dir'th bit to 0 = IndexType::CELL.
    //
    IndexType typ(b.ixType());
    typ.unset(dir);
    return Box(low,hi,typ,b.dimension());
}

//! Similar to adjCellLo and adjCellHi; operates on given face.
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box adjCell (const Box& b, Orientation face, int len=1) noexcept
{
    BL_ASSERT(len > 0);
    IntVect low(b.smallEnd());
    IntVect hi(b.bigEnd());
    int dir = face.coordDir();
    if (face.isLow())
    {
        int sm = low[dir];
        low.setVal(dir,sm - len);
        hi.setVal(dir,sm - 1);
    }
    else
    {
        unsigned int bitval = b.type()[dir];
        int bg              = hi[dir]  + 1 - bitval%2;
        low.setVal(dir,bg);
        hi.setVal(dir,bg + len - 1);
    }
    //
    // Set dir'th bit to 0 = IndexType::CELL.
    //
    IndexType typ(b.ixType());
    typ.unset(dir);
    return Box(low,hi,typ,b.dimension());
}

/**
 * \brief Modify Box to that of the minimum Box containing both
 * the original Box and the argument.
 * Both Boxes must have identical type.
 */
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box minBox (const Box& b1, const Box& b2) noexcept
{
    Box result = b1;
    result.minBox(b2);
    return result;
}

//! Write an ASCII representation to the ostream.
std::ostream& operator<< (std::ostream& os, const Box& bx);

//! Read from istream.
std::istream& operator>> (std::istream& os, Box& bx);

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 lbound (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.smallend[0], 0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {box.smallend[0], box.smallend[1], 0};
#else
    return {box.smallend[0], box.smallend[1], box.smallend[2]};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 ubound (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.bigend[0], 0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {box.bigend[0], box.bigend[1], 0};
#else
    return {box.bigend[0], box.bigend[1], box.bigend[2]};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 begin (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.smallend[0], 0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {box.smallend[0], box.smallend[1], 0};
#else
    return {box.smallend[0], box.smallend[1], box.smallend[2]};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 end (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.bigend[0]+1, 1, 1};
#elif (AMREX_SPACEDIM == 2)
    return {box.bigend[0]+1, box.bigend[1]+1, 1};
#else
    return {box.bigend[0]+1, box.bigend[1]+1, box.bigend[2]+1};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 length (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.bigend[0]-box.smallend[0]+1, 1, 1};
#elif (AMREX_SPACEDIM == 2)
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1, 1};
#else
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1,
            box.bigend[2]-box.smallend[2]+1};
#endif
}

// Max of lower bound
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 max_lbound (Box const& b1, Box const& b2) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {amrex::max(b1.smallend[0], b2.smallend[0]),
            0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {amrex::max(b1.smallend[0], b2.smallend[0]),
            amrex::max(b1.smallend[1], b2.smallend[1]),
            0};
#else
    return {amrex::max(b1.smallend[0], b2.smallend[0]),
            amrex::max(b1.smallend[1], b2.smallend[1]),
            amrex::max(b1.smallend[2], b2.smallend[2])};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 max_lbound (Box const& b1, Dim3 const& lo) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {amrex::max(b1.smallend[0], lo.x),
            0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {amrex::max(b1.smallend[0], lo.x),
            amrex::max(b1.smallend[1], lo.y),
            0};
#else
    return {amrex::max(b1.smallend[0], lo.x),
            amrex::max(b1.smallend[1], lo.y),
            amrex::max(b1.smallend[2], lo.z)};
#endif
}

// Min of upper bound
AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 min_ubound (Box const& b1, Box const& b2) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {amrex::min(b1.bigend[0], b2.bigend[0]),
            0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {amrex::min(b1.bigend[0], b2.bigend[0]),
            amrex::min(b1.bigend[1], b2.bigend[1]),
            0};
#else
    return {amrex::min(b1.bigend[0], b2.bigend[0]),
            amrex::min(b1.bigend[1], b2.bigend[1]),
            amrex::min(b1.bigend[2], b2.bigend[2])};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Dim3 min_ubound (Box const& b1, Dim3 const& hi) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {amrex::min(b1.bigend[0], hi.x),
            0, 0};
#elif (AMREX_SPACEDIM == 2)
    return {amrex::min(b1.bigend[0], hi.x),
            amrex::min(b1.bigend[1], hi.y),
            0};
#else
    return {amrex::min(b1.bigend[0], hi.x),
            amrex::min(b1.bigend[1], hi.y),
            amrex::min(b1.bigend[2], hi.z)};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box minBox (Box const& b1, Box const& b2, IndexType typ) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return Box(IntVect(amrex::max(b1.smallend[0], b2.smallend[0])),
               IntVect(amrex::min(b1.bigend  [0], b2.bigend  [0])),
               typ, b1.dimension());
#elif (AMREX_SPACEDIM == 2)
    return Box(IntVect(amrex::max(b1.smallend[0], b2.smallend[0]),
                       amrex::max(b1.smallend[1], b2.smallend[1])),
               IntVect(amrex::min(b1.bigend  [0], b2.bigend  [0]),
                       amrex::min(b1.bigend  [1], b2.bigend  [1])),
               typ, b1.dimension());
#elif (AMREX_SPACEDIM == 3)
    return Box(IntVect(amrex::max(b1.smallend[0], b2.smallend[0]),
                       amrex::max(b1.smallend[1], b2.smallend[1]),
                       amrex::max(b1.smallend[2], b2.smallend[2])),
               IntVect(amrex::min(b1.bigend  [0], b2.bigend  [0]),
                       amrex::min(b1.bigend  [1], b2.bigend  [1]),
                       amrex::min(b1.bigend  [2], b2.bigend  [2])),
               typ, b1.dimension());
#elif (AMREX_SPACEDIM == 4)
    return Box(IntVect(amrex::max(b1.smallend[0], b2.smallend[0]),
                       amrex::max(b1.smallend[1], b2.smallend[1]),
                       amrex::max(b1.smallend[2], b2.smallend[2]),
                       amrex::max(b1.smallend[3], b2.smallend[3])),
               IntVect(amrex::min(b1.bigend  [0], b2.bigend  [0]),
                       amrex::min(b1.bigend  [1], b2.bigend  [1]),
                       amrex::min(b1.bigend  [2], b2.bigend  [2]),
                       amrex::min(b1.bigend  [3], b2.bigend  [3])),
               typ, b1.dimension());
#elif (AMREX_SPACEDIM == 5)
    return Box(IntVect(amrex::max(b1.smallend[0], b2.smallend[0]),
                       amrex::max(b1.smallend[1], b2.smallend[1]),
                       amrex::max(b1.smallend[2], b2.smallend[2]),
                       amrex::max(b1.smallend[3], b2.smallend[3]),
                       amrex::max(b1.smallend[4], b2.smallend[4])),
               IntVect(amrex::min(b1.bigend  [0], b2.bigend  [0]),
                       amrex::min(b1.bigend  [1], b2.bigend  [1]),
                       amrex::min(b1.bigend  [2], b2.bigend  [2]),
                       amrex::min(b1.bigend  [3], b2.bigend  [3]),
                       amrex::min(b1.bigend  [4], b2.bigend  [4])),
               typ, b1.dimension());
#elif (AMREX_SPACEDIM == 6)
    return Box(IntVect(amrex::max(b1.smallend[0], b2.smallend[0]),
                       amrex::max(b1.smallend[1], b2.smallend[1]),
                       amrex::max(b1.smallend[2], b2.smallend[2]),
                       amrex::max(b1.smallend[3], b2.smallend[3]),
                       amrex::max(b1.smallend[4], b2.smallend[4]),
                       amrex::max(b1.smallend[5], b2.smallend[5])),
               IntVect(amrex::min(b1.bigend  [0], b2.bigend  [0]),
                       amrex::min(b1.bigend  [1], b2.bigend  [1]),
                       amrex::min(b1.bigend  [2], b2.bigend  [2]),
                       amrex::min(b1.bigend  [3], b2.bigend  [3]),
                       amrex::min(b1.bigend  [4], b2.bigend  [4]),
                       amrex::min(b1.bigend  [5], b2.bigend  [5])),
               typ, b1.dimension());
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
DimN lboundn (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.smallend[0]};
#elif (AMREX_SPACEDIM == 2)
    return {box.smallend[0], box.smallend[1]};
#elif (AMREX_SPACEDIM == 3)
    return {box.smallend[0], box.smallend[1], box.smallend[2]};
#elif (AMREX_SPACEDIM == 4)
    return {box.smallend[0], box.smallend[1], box.smallend[2], box.smallend[3]};
#elif (AMREX_SPACEDIM == 5)
    return {box.smallend[0], box.smallend[1], box.smallend[2], box.smallend[3], box.smallend[4]};
#elif (AMREX_SPACEDIM == 6)
    return {box.smallend[0], box.smallend[1], box.smallend[2], box.smallend[3], box.smallend[4], box.smallend[5]};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
DimN uboundn (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.bigend[0]};
#elif (AMREX_SPACEDIM == 2)
    return {box.bigend[0], box.bigend[1]};
#elif (AMREX_SPACEDIM == 3)
    return {box.bigend[0], box.bigend[1], box.bigend[2]};
#elif (AMREX_SPACEDIM == 4)
    return {box.bigend[0], box.bigend[1], box.bigend[2], box.bigend[3]};
#elif (AMREX_SPACEDIM == 5)
    return {box.bigend[0], box.bigend[1], box.bigend[2], box.bigend[3], box.bigend[4]};
#elif (AMREX_SPACEDIM == 6)
    return {box.bigend[0], box.bigend[1], box.bigend[2], box.bigend[3], box.bigend[4], box.bigend[5]};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
DimN beginn (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.smallend[0]};
#elif (AMREX_SPACEDIM == 2)
    return {box.smallend[0], box.smallend[1]};
#elif (AMREX_SPACEDIM == 3)
    return {box.smallend[0], box.smallend[1], box.smallend[2]};
#elif (AMREX_SPACEDIM == 4)
    return {box.smallend[0], box.smallend[1], box.smallend[2], box.smallend[3]};
#elif (AMREX_SPACEDIM == 5)
    return {box.smallend[0], box.smallend[1], box.smallend[2], box.smallend[3], box.smallend[4]};
#elif (AMREX_SPACEDIM == 6)
    return {box.smallend[0], box.smallend[1], box.smallend[2], box.smallend[3], box.smallend[4], box.smallend[5]};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
DimN endn (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.bigend[0]+1};
#elif (AMREX_SPACEDIM == 2)
    return {box.bigend[0]+1, box.bigend[1]+1};
#elif (AMREX_SPACEDIM == 3)
    return {box.bigend[0]+1, box.bigend[1]+1, box.bigend[2]+1};
#elif (AMREX_SPACEDIM == 4)
    return {box.bigend[0]+1, box.bigend[1]+1, box.bigend[2]+1, box.bigend[3]+1};
#elif (AMREX_SPACEDIM == 5)
    return {box.bigend[0]+1, box.bigend[1]+1, box.bigend[2]+1, box.bigend[3]+1, box.bigend[4]+1};
#elif (AMREX_SPACEDIM == 6)
    return {box.bigend[0]+1, box.bigend[1]+1, box.bigend[2]+1, box.bigend[3]+1, box.bigend[4]+1, box.bigend[5]+1};
#endif
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
DimN lengthn (Box const& box) noexcept
{
#if (AMREX_SPACEDIM == 1)
    return {box.bigend[0]-box.smallend[0]+1};
#elif (AMREX_SPACEDIM == 2)
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1};
#elif (AMREX_SPACEDIM == 3)
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1,
            box.bigend[2]-box.smallend[2]+1};
#elif (AMREX_SPACEDIM == 4)
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1,
            box.bigend[2]-box.smallend[2]+1,
            box.bigend[3]-box.smallend[3]+1};
#elif (AMREX_SPACEDIM == 5)
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1,
            box.bigend[2]-box.smallend[2]+1,
            box.bigend[3]-box.smallend[3]+1,
            box.bigend[4]-box.smallend[4]+1};
#elif (AMREX_SPACEDIM == 6)
    return {box.bigend[0]-box.smallend[0]+1,
            box.bigend[1]-box.smallend[1]+1,
            box.bigend[2]-box.smallend[2]+1,
            box.bigend[3]-box.smallend[3]+1,
            box.bigend[4]-box.smallend[4]+1,
            box.bigend[5]-box.smallend[5]+1};
#endif
}

// Returns a Box that covers all the argument Boxes in index
// space. The types are ignored. Thus, the arguments can have
// different index types, and the returned Box's index type has no
// meaning.
AMREX_FORCE_INLINE
Box getIndexBounds (Box const& b1) noexcept
{
    return b1;
}

AMREX_FORCE_INLINE
Box getIndexBounds (Box const& b1, Box const& b2) noexcept
{
    Box b = b1;
    b.setType(b2.ixType());
    b.minBox(b2);
    return b;
}

template <class T, class ... Ts>
AMREX_FORCE_INLINE
Box getIndexBounds (T const& b1, T const& b2, Ts const& ... b3) noexcept
{
    return getIndexBounds(getIndexBounds(b1,b2),b3...);
}


AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
IntVect getCell (Box const* boxes, int nboxes, Long icell) noexcept
{
    int ibox;
    Long ncells_subtotal = 0;
    Long offset = 0;
    for (ibox = 0; ibox < nboxes; ++ibox) {
        const Long n = boxes[ibox].numPts();
        ncells_subtotal += n;
        if (icell < ncells_subtotal) {
            offset = icell - (ncells_subtotal - n);
            break;
        }
    }
    return boxes[ibox].atOffset(offset);
}

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
Box makeSlab (Box const& b, int direction, int slab_index) noexcept
{
    Box r = b;
    r.makeSlab(direction,slab_index);
    return r;
}

}

#endif /*BL_BOX_H*/
