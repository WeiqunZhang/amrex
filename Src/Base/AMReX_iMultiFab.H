
#ifndef BL_IMULTIFAB_H
#define BL_IMULTIFAB_H
#include <AMReX_Config.H>

#include <AMReX_BLassert.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabArrayUtility.H>
#include <AMReX_Geometry.H>
#include <memory>

/**
 * \file AMReX_iMultiFab.H
 * \brief Integer-valued MultiFab container and helpers.
 */

namespace amrex {

/**
* \class amrex::iMultiFab
* \ingroup amrex_multifab_containers
* \brief A Collection of IArrayBoxes
*
* The iMultiFab class is publicly derived from the
* FabArray<int,IArrayBox> class.  It is a collection (stored as an array) of
* IArrayBoxes useful for storing integer data on a domain defined by
* a union of rectangular regions embedded in a uniform index space.  The
* iMultiFab class extends the function of the underlying FabArray class just
* as the IArrayBox class extends the function of BaseFab<int>.  Additional
* member functions are defined for I/O and simple arithmetic operations on
* these aggregate objects.
*
* This class does NOT provide a copy constructor or assignment operator.
*/
class iMultiFab
    :
    public FabArray<IArrayBox>
{
public:

    /**
    * \brief Constructs an empty iMultiFab.  Data can be defined at a later
    * time using the define member functions inherited
    * from FabArray.
    */
    iMultiFab () noexcept = default;

    /**
    * \brief Constructs an empty iMultiFab.  Data can be defined at a later
    * time using the define member functions inherited from FabArray.  If
    * `define` is called later with a nullptr as MFInfo's arena, the default
    * Arena `a` will be used.  If the arena in MFInfo is not a nullptr, the
    * MFInfo's arena will be used.
    */
    explicit iMultiFab (Arena* a) noexcept;

    /**
    * \brief Constructs a iMultiFab with a valid region defined by bxs and
    * a region of definition defined by the grow factor ngrow.
    */
    iMultiFab (const BoxArray&            bxs,
               const DistributionMapping& dm,
               int                        ncomp,
               int                        ngrow,
#ifdef AMREX_STRICT_MODE
               const MFInfo&              info,
               const FabFactory<IArrayBox>& factory);
#else
               const MFInfo&              info = MFInfo(),
               const FabFactory<IArrayBox>& factory = DefaultFabFactory<IArrayBox>());
#endif

    iMultiFab (const BoxArray&            bxs,
               const DistributionMapping& dm,
               int                        ncomp,
               const IntVect&             ngrow,
#ifdef AMREX_STRICT_MODE
               const MFInfo&              info,
               const FabFactory<IArrayBox>& factory);
#else
               const MFInfo&              info = MFInfo(),
               const FabFactory<IArrayBox>& factory = DefaultFabFactory<IArrayBox>());
#endif

    /**
    * \brief Make an alias iMultiFab. maketype must be
    * amrex::make_alias.  scomp is the starting component of the
    * alias and ncomp is the number of components in the new aliasing
    * iMultiFab.
    *
    * \param rhs
    * \param maketype
    * \param scomp
    * \param ncomp
    */
    iMultiFab (const iMultiFab& rhs, MakeType maketype, int scomp, int ncomp);

    ~iMultiFab () = default;

    iMultiFab (iMultiFab&& rhs) noexcept = default;
    iMultiFab& operator= (iMultiFab&& rhs) noexcept = default;

    iMultiFab (const iMultiFab& rhs) = delete;
    iMultiFab& operator= (const iMultiFab& rhs) = delete;

    iMultiFab& operator= (int r);

    /**
    * \brief Returns the minimum value contained in component comp of the
    * iMultiFab.  The parameter nghost determines the number of
    * boundary cells to search for the minimum.  The default is to
    * search only the valid regions of the IArrayBoxes.
    *
    * \param comp
    * \param nghost
    * \param local
    */
    [[nodiscard]] int min (int comp,
             int nghost = 0,
             bool local = false) const;

    /**
    * \brief Identical to the previous min() function, but confines its
    * search to intersection of Box b and the iMultiFab.
    *
    * \param region
    * \param comp
    * \param nghost
    * \param local
    */
    [[nodiscard]] int min (const Box& region,
             int        comp,
             int        nghost = 0,
             bool       local = false) const;

    /**
    * \brief Returns the maximum value contained in component comp of the
    * iMultiFab.  The parameter nghost determines the number of
    * boundary cells to search for the maximum.  The default is to
    * search only the valid regions of the IArrayBoxes.
    *
    * \param comp
    * \param nghost
    * \param local
    */
    [[nodiscard]] int max (int comp,
             int nghost = 0,
             bool local = false) const;

    /**
    * \brief Identical to the previous max() function, but confines its
    * search to intersection of Box b and the iMultiFab.
    *
    * \param region
    * \param comp
    * \param nghost
    * \param local
    */
    [[nodiscard]] int max (const Box& region,
             int        comp,
             int        nghost = 0,
             bool       local = false) const;

    /**
    * \brief Returns the sum in component comp
    *
    * \param comp
    * \param nghost
    * \param local
    */
    [[nodiscard]] Long sum (int comp, int nghost = 0, bool local = false) const;

    /**
    * \brief Sum component \p comp over \p region (no ghost cells are included).
    *
    * \param region Box defining the region of interest (intersected with the valid domain).
    * \param comp   Component index to accumulate.
    * \param local  When true, skip the parallel reduction and return the local rank sum only.
    */
    [[nodiscard]] Long sum (Box const& region, int comp = 0, bool local = false) const;

    /**
    * \brief Adds the scalar value val to the value of each cell in the
    * specified subregion of the iMultiFab.  The subregion consists
    * of the num_comp components starting at component comp.
    * The value of nghost specifies the number of cells in the
    * boundary region of each IArrayBox in the subregion that should
    * be modified.
    *
    * \param val
    * \param comp
    * \param num_comp
    * \param nghost
    */
    void plus (int val,
               int  comp,
               int  num_comp,
               int  nghost = 0);

    /**
    * \brief Identical to the previous version of plus(), with the
    * restriction that the subregion is further constrained to
    * the intersection with Box region.
    *
    * \param val
    * \param region
    * \param comp
    * \param num_comp
    * \param nghost
    */
    void plus (int       val,
               const Box& region,
               int        comp,
               int        num_comp,
               int        nghost = 0);

    /**
    * \brief Adds the scalar value val to the value of each cell in the
    * valid region of each component of the iMultiFab.  The value
    * of nghost specifies the number of cells in the boundary
    * region that should be modified.
    *
    * \param val
    * \param nghost
    */
    void plus (int val,
               int  nghost);

    /**
    * \brief Adds the scalar value val to the value of each cell in the
    * valid region of each component of the iMultiFab, that also
    * intersects the Box region.  The value of nghost specifies the
    * number of cells in the boundary region of each IArrayBox in
    * the subregion that should be modified.
    *
    * \param val
    * \param region
    * \param nghost
    */
    void plus (int       val,
               const Box& region,
               int        nghost);

    /**
    * \brief Scales the value of each cell in the specified subregion of the
    * iMultiFab by the scalar val (a[i] \<- a[i]*val). The subregion
    * consists of the num_comp components starting at component comp.
    * The value of nghost specifies the number of cells in the
    * boundary region of each IArrayBox in the subregion that should
    * be modified.
    *
    * \param val
    * \param comp
    * \param num_comp
    * \param nghost
    */
    void mult (int val,
               int  comp,
               int  num_comp,
               int  nghost = 0);

    /**
    * \brief Identical to the previous version of mult(), with the
    * restriction that the subregion is further constrained to the
    * intersection with Box region.  The value of nghost specifies the
    * number of cells in the boundary region of each IArrayBox in
    * the subregion that should be modified.
    *
    * \param val
    * \param region
    * \param comp
    * \param num_comp
    * \param nghost
    */
    void mult (int       val,
               const Box& region,
               int        comp,
               int        num_comp,
               int        nghost = 0);

    /**
    * \brief Scales the value of each cell in the valid region of each
    * component of the iMultiFab by the scalar val (a[i] \<- a[i]*val).
    * The value of nghost specifies the number of cells in the
    * boundary region that should be modified.
    *
    * \param val
    * \param nghost
    */
    void mult (int val,
               int  nghost = 0);

    /**
    * \brief Scales the value of each cell in the valid region of each
    * component of the iMultiFab by the scalar val (a[i] \<- a[i]*val),
    * that also intersects the Box region.  The value of nghost
    * specifies the number of cells in the boundary region of each
    * IArrayBox in the subregion that should be modified.
    *
    * \param val
    * \param region
    * \param nghost
    */
    void mult (int       val,
               const Box& region,
               int        nghost = 0);

    /**
    * \brief Negates the value of each cell in the specified subregion of
    * the iMultiFab.  The subregion consists of the num_comp
    * components starting at component comp.  The value of nghost
    * specifies the number of cells in the boundary region of each
    * IArrayBox in the subregion that should be modified.
    *
    * \param comp
    * \param num_comp
    * \param nghost
    */
    void negate (int comp,
                 int num_comp,
                 int nghost = 0);

    /**
    * \brief Identical to the previous version of negate(), with the
    * restriction that the subregion is further constrained to
    * the intersection with Box region.
    *
    * \param region
    * \param comp
    * \param num_comp
    * \param nghost
    */
    void negate (const Box& region,
                 int        comp,
                 int        num_comp,
                 int        nghost = 0);

    /**
    * \brief Negates the value of each cell in the valid region of
    * the iMultiFab.  The value of nghost specifies the number of
    * cells in the boundary region that should be modified.
    *
    * \param nghost
    */
    void negate (int nghost = 0);

    /**
    * \brief Negates the value of each cell in the valid region of
    * the iMultiFab that also intersects the Box region.  The value
    * of nghost specifies the number of cells in the boundary region
    * that should be modified.
    *
    * \param region
    * \param nghost
    */
    void negate (const Box& region,
                 int        nghost = 0);

    [[nodiscard]] IntVect minIndex (int comp,
                      int nghost = 0) const;

    [[nodiscard]] IntVect maxIndex (int comp,
                      int nghost = 0) const;

    /**
    * \brief This function adds the values of the cells in mf to the corresponding
    * cells of this iMultiFab.  mf is required to have the same BoxArray or
    * "valid region" as this iMultiFab.  The addition is done only to num_comp
    * components, starting with component number strt_comp.  The parameter
    * nghost specifies the number of boundary cells that will be modified.
    * If nghost == 0, only the valid region of each IArrayBox will be
    * modified.
    *
    * \param mf
    * \param strt_comp
    * \param num_comp
    * \param nghost
    */
    void plus (const iMultiFab& mf,
               int             strt_comp,
               int             num_comp,
               int             nghost);

    /**
    * \brief This function subtracts the values of the cells in mf from the
    * corresponding cells of this iMultiFab.  mf is required to have the
    * same BoxArray or "valid region" as this iMultiFab.  The subtraction is
    * done only to num_comp components, starting with component number
    * strt_comp.  The parameter nghost specifies the number of boundary
    * cells that will be modified.  If nghost == 0, only the valid region of
    * each IArrayBox will be modified.
    *
    * \param mf
    * \param strt_comp
    * \param num_comp
    * \param nghost
    */
    void minus (const iMultiFab& mf,
                int             strt_comp,
                int             num_comp,
                int             nghost);

    /**
    * \brief This function divides the values of the cells in mf from the
    * corresponding cells of this iMultiFab.  mf is required to have the
    * same BoxArray or "valid region" as this iMultiFab.  The division is
    * done only to num_comp components, starting with component number
    * strt_comp.  The parameter nghost specifies the number of boundary
    * cells that will be modified.  If nghost == 0, only the valid region of
    * each IArrayBox will be modified.  Note, nothing is done to protect
    * against divide by zero.
    *
    * \param mf
    * \param strt_comp
    * \param num_comp
    * \param nghost
    */
    void divide (const iMultiFab& mf,
                 int             strt_comp,
                 int             num_comp,
                 int             nghost);

    /**
    * \brief Add \p src into \p dst over the specified component range.
    *
    * Both iMultiFabs must share the same BoxArray. Components
    * `srccomp..srccomp+numcomp-1` are added into the corresponding destination
    * components `dstcomp..dstcomp+numcomp-1`, including \p nghost ghost cells.
    *
    * \param dst     Destination MultiFab updated in place.
    * \param src     Source MultiFab providing the values to add.
    * \param srccomp First component in \p src to read.
    * \param dstcomp First component in \p dst to update.
    * \param numcomp Number of components to process.
    * \param nghost  Number of ghost cells to include.
    */
    static void Add (iMultiFab&       dst,
                     const iMultiFab& src,
                     int              srccomp,
                     int              dstcomp,
                     int              numcomp,
                     int              nghost);

    static void Add (iMultiFab&       dst,
                     const iMultiFab& src,
                     int              srccomp,
                     int              dstcomp,
                     int              numcomp,
                     const IntVect&   nghost);

    /**
    * \brief Copy components from \p src to \p dst, including \p nghost ghost cells.
    *
    * Both iMultiFabs must share the same BoxArray so their valid regions align.
    *
    * \param dst     Destination MultiFab updated in place.
    * \param src     Source MultiFab providing the values to copy.
    * \param srccomp First component in \p src to read.
    * \param dstcomp First component in \p dst to update.
    * \param numcomp Number of components to process.
    * \param nghost  Number of ghost cells to include.
    */
    static void Copy (iMultiFab&       dst,
                      const iMultiFab& src,
                      int             srccomp,
                      int             dstcomp,
                      int             numcomp,
                      int             nghost);

    static void Copy (iMultiFab&       dst,
                      const iMultiFab& src,
                      int             srccomp,
                      int             dstcomp,
                      int             numcomp,
                      const IntVect&  nghost);

    /**
    * \brief Swap component ranges between two iMultiFabs, including \p nghost ghost cells.
    *
    * Both arrays must share the same BoxArray, and the swap is purely local
    * (no MPI communication).
    */
    static void Swap (iMultiFab&      dst,
                      iMultiFab&      src,
                      int             srccomp,
                      int             dstcomp,
                      int             numcomp,
                      int             nghost);

    static void Swap (iMultiFab&      dst,
                      iMultiFab&      src,
                      int             srccomp,
                      int             dstcomp,
                      int             numcomp,
                      const IntVect&  nghost);

    /**
    * \brief Subtract \p src from \p dst across \p numcomp components and \p nghost ghosts.
    *
    * The operands must have identical BoxArrays.
    */
    static void Subtract (iMultiFab&       dst,
                          const iMultiFab& src,
                          int              srccomp,
                          int              dstcomp,
                          int              numcomp,
                          int              nghost);

    static void Subtract (iMultiFab&       dst,
                          const iMultiFab& src,
                          int              srccomp,
                          int              dstcomp,
                          int              numcomp,
                          const IntVect&   nghost);

    /**
    * \brief Multiply \p dst by \p src component-wise, including \p nghost ghosts.
    *
    * The operands must have identical BoxArrays.
    */
    static void Multiply (iMultiFab&       dst,
                          const iMultiFab& src,
                          int              srccomp,
                          int              dstcomp,
                          int              numcomp,
                          int              nghost);

    static void Multiply (iMultiFab&       dst,
                          const iMultiFab& src,
                          int              srccomp,
                          int              dstcomp,
                          int              numcomp,
                          const IntVect&   nghost);

    /**
    * \brief Divide \p dst by \p src component-wise, including \p nghost ghosts.
    *
    * The operands must have identical BoxArrays. No protection against divide-by-zero is provided.
    */
    static void Divide (iMultiFab&       dst,
                        const iMultiFab& src,
                        int              srccomp,
                        int              dstcomp,
                        int              numcomp,
                        int              nghost);

    static void Divide (iMultiFab&       dst,
                        const iMultiFab& src,
                        int              srccomp,
                        int              dstcomp,
                        int              numcomp,
                        const IntVect&   nghost);

    /**
    * \brief Define/resize the iMultiFab with explicit grow vector.
    *
    * \param bxs    BoxArray describing the valid region.
    * \param dm     Distribution mapping assigning boxes to ranks.
    * \param nvar   Number of components.
    * \param ngrow  Grow vector (per-direction ghost cells).
    * \param info   Optional MFInfo controlling allocation options.
    * \param factory Fab factory used to instantiate IArrayBoxes.
    */
    void define (const BoxArray&            bxs,
                 const DistributionMapping& dm,
                 int                        nvar,
                 const IntVect&             ngrow,
#ifdef AMREX_STRICT_MODE
                 const MFInfo&              info,
                 const FabFactory<IArrayBox>& factory);
#else
                 const MFInfo&              info = MFInfo(),
                 const FabFactory<IArrayBox>& factory = DefaultFabFactory<IArrayBox>());
#endif

    /**
    * \brief Define/resize the iMultiFab with an isotropic grow value.
    *
    * \param bxs    BoxArray describing the valid region.
    * \param dm     Distribution mapping assigning boxes to ranks.
    * \param nvar   Number of components.
    * \param ngrow  Number of ghost cells in every direction.
    * \param info   Optional MFInfo controlling allocation options.
    * \param factory Fab factory used to instantiate IArrayBoxes.
    */
    void define (const BoxArray&            bxs,
                 const DistributionMapping& dm,
                 int                        nvar,
                 int                        ngrow,
#ifdef AMREX_STRICT_MODE
                 const MFInfo&              info,
                 const FabFactory<IArrayBox>& factory);
#else
                 const MFInfo&              info = MFInfo(),
                 const FabFactory<IArrayBox>& factory = DefaultFabFactory<IArrayBox>());
#endif

    //! Initialize iMultiFab static state (allocator tracking).
    static void Initialize ();
    //! Tear down any static state created by Initialize().
    static void Finalize ();
};

// ngrow != IntVect{0} is a special case that should not be used in normal cases,
// because it may mark valid cells as non-owner and ghost cells as owners.
/**
 * \brief Build a mask identifying the owner of each cell across overlapping boxes.
 *
 * Cells marked 1 belong to the canonical owner, whereas 0 marks non-owners.
 * Ghost-cell grow vectors other than zero are discouraged because they can flip
 * ownership.
 *
 * \param mf     MultiFab-like object whose ownership is being queried.
 * \param period Domain periodicity used for ghost resolution.
 * \param ngrow  Optional extra grow cells for the mask (default zero).
 */
std::unique_ptr<iMultiFab>
OwnerMask (FabArrayBase const& mf, const Periodicity& period, const IntVect& ngrow=IntVect{0});

}

#endif /*BL_IMULTIFAB_H*/
