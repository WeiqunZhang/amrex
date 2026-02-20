#ifndef AMREX_CUDA_GRAPH_H_
#define AMREX_CUDA_GRAPH_H_
#include <AMReX_Config.H>

#if defined(__CUDACC__) && defined(AMREX_USE_CUDA)

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_GpuDevice.H>

/**
 * \file AMReX_CudaGraph.H
 * \brief Helpers for building and launching CUDA Graph executions in AMReX.
 */

namespace amrex {

/** \brief Describes a strided copy region captured inside a CUDA graph. */
struct CopyMemory
{
    void* src = nullptr;
    void* dst = nullptr;
    Dim3 src_begin = {0,0,0};
    Dim3 src_end = {0,0,0};
    Dim3 dst_begin = {0,0,0};
    Dim3 dst_end = {0,0,0};
    int scomp = 0;
    int ncomp = 0;

    template <class T>
    AMREX_GPU_HOST_DEVICE
    Array4<T const> getSrc () { return Array4<T const>(static_cast<T const*>(src), src_begin, src_end, scomp+ncomp); }

    template <class T>
    AMREX_GPU_HOST_DEVICE
    Array4<T> getDst () { return Array4<T>(static_cast<T*>(dst), dst_begin, dst_end, scomp+ncomp); }
};

/**
 * \brief Build a CopyMemory descriptor from Array4 views.
 *
 * \tparam T Source Array4 value type.
 * \tparam U Destination Array4 value type.
 * \param src Source Array4 view.
 * \param dst Destination Array4 view.
 * \param scomp Starting component index to copy.
 * \param ncomp Number of components to copy.
 * \return Populated CopyMemory ready for capture into a CUDA graph.
 */
template <typename T, typename U>
CopyMemory
makeCopyMemory (Array4<T> const& src, Array4<U> const& dst, int scomp, int ncomp)
{
    return CopyMemory{ (void*)(src.p), (void*)(dst.p), IntVectShrink<3>(src.begin).dim3(), IntVectShrink<3>(src.end).dim3(),
        IntVectShrink<3>(dst.begin).dim3(), IntVectShrink<3>(dst.end).dim3(), scomp, ncomp };
}

// ======================================================================================

/**
 * \brief Lightweight RAII wrapper over a cudaGraphExec_t plus parameter storage.
 * \tparam T trivially-copyable struct describing kernel parameters.
 */
template <typename T>
struct CudaGraph
{
    cudaGraphExec_t m_graph;
    Vector<T> m_parms;
    T* m_parms_d = nullptr;
    bool graph_is_ready = false;

    //! Construct an empty wrapper; call resize() before setting parameters.
    CudaGraph()
        : m_parms(0)
    {
        static_assert(std::is_trivially_copyable_v<T>, "CudaGraph's T must be trivially copyable");
    }
    /**
     * \brief Construct a wrapper that stores \p num parameter blocks.
     * \param num Number of kernels represented in the captured graph.
     */
    CudaGraph(int num)
        : m_parms(num)
    {
        static_assert(std::is_trivially_copyable_v<T>, "CudaGraph's T must be trivially copyable");
        m_parms_d = static_cast<T*>( The_Arena()->alloc(sizeof(T)*m_parms.size()) );
    }
    //! Destructor releases the device buffer and destroys the graph exec if present.
    ~CudaGraph() {
        The_Arena()->free(m_parms_d);

        if (graph_is_ready)
        {
            AMREX_CUDA_SAFE_CALL(cudaGraphExecDestroy(m_graph));
        }
    }

    /**
     * \brief Resize the parameter array and reallocate the device buffer.
     * \param num New number of parameter entries.
     */
    void resize(Long num) {
        m_parms.resize(num);
        if (m_parms_d != nullptr)
        {
            The_Arena()->free(m_parms_d);
        }
        m_parms_d = static_cast<T*>( The_Arena()->alloc(sizeof(T)*m_parms.size()) );
    }
    /**
     * \brief Install an executable graph handle captured elsewhere.
     * \param graph Executable graph returned by cudaGraphInstantiate.
     */
    void setGraph(cudaGraphExec_t const& graph) {
        m_graph = graph;
        graph_is_ready = true;
    }
    /**
     * \brief Populate the host-side parameter entry at \p idx.
     * \param idx Kernel index converted during capture.
     * \param a_parm Parameter payload copied into m_parms.
     */
    void setParams(int idx, T const& a_parm) { m_parms[idx] = a_parm; }

    //! Return a pointer to the host parameter slot for kernel \p idx.
    T* getHostPtr (int idx) { return (m_parms.data() + idx); }
    //! Return a pointer to the device parameter slot for kernel \p idx.
    T* getDevicePtr (int idx) const { return (m_parms_d + idx); }
    //! Whether setGraph() has been called and the graph can be launched.
    bool ready() const { return graph_is_ready; }

    /**
     * \brief Launch the captured graph on the current stream.
     * \param synch When true, call cudaStreamSynchronize after submitting.
     */
    void executeGraph (bool synch = true) const {
        Gpu::Device::executeGraph(m_graph, synch);
    }
};

}

#endif
#endif
