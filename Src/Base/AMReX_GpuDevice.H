#ifndef AMREX_GPU_DEVICE_H_
#define AMREX_GPU_DEVICE_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_GpuDevice.H
 * \brief GPU device management, stream control, and memory-copy helpers.
 */

#include <AMReX.H>
#include <AMReX_Extension.H>
#include <AMReX_Utility.H>
#include <AMReX_GpuTypes.H>
#include <AMReX_GpuError.H>
#include <AMReX_GpuControl.H>
#include <AMReX_OpenMP.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

#define AMREX_GPU_MAX_STREAMS 8

/// \cond DOXYGEN_IGNORE
#ifdef AMREX_USE_GPU
namespace amrex {
#ifdef AMREX_USE_HIP
using gpuDeviceProp_t = hipDeviceProp_t;
#elif defined(AMREX_USE_CUDA)
using gpuDeviceProp_t = cudaDeviceProp;
#elif defined(AMREX_USE_SYCL)
    struct gpuDeviceProp_t {
        std::string name;
        std::string vendor; // SYCL only (inferred for CUDA and HIP)
        std::size_t totalGlobalMem;
        std::size_t sharedMemPerBlock;
        int multiProcessorCount;
        int maxThreadsPerMultiProcessor;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int warpSize;
        Long maxMemAllocSize; // SYCL only
        int managedMemory;
        int concurrentManagedAccess;
        int maxParameterSize;
    };
#endif
}
#endif
/// \endcond

namespace amrex {
    class Arena;
}

namespace amrex::Gpu {

/// \cond DOXYGEN_IGNORE
#ifdef AMREX_USE_GPU
enum class ExternalStreamSync {
    Yes,
    No
};
#endif

#ifdef AMREX_USE_GPU
class StreamManager {
    gpuStream_t m_stream;
    std::mutex m_mutex;
    Vector<std::pair<Arena*, void*>> m_free_wait_list;
public:
    //! Return the underlying GPU stream.
    [[nodiscard]] gpuStream_t& getStream ();
    //! Return the underlying GPU stream.
    [[nodiscard]] gpuStream_t const& getStream () const;
    void sync ();
    void free_async (Arena* arena, void* mem);
    std::size_t wait_list_size ();
};
#endif
/// \endcond

/**
 * \class amrex::Gpu::Device
 * \ingroup amrex_execution
 * \brief Manages GPU device selection, stream pools, memory-copy helpers, and device queries.
 *
 * All member functions are static; there is no instance of this class.
 */
class Device
{

public:

    //! Initialize the GPU device with the given \p a_device_id; call before any GPU work.
    static void Initialize (bool minimal, int a_device_id);
    //! Finalize and release GPU device resources.
    static void Finalize ();

#if defined(AMREX_USE_GPU)
    //! Return the GPU stream for the current OpenMP thread.
    static gpuStream_t gpuStream () noexcept {
        if (!external_stream_stack.empty()) {
            AMREX_ASSERT(external_stream_stack.back().manager != nullptr);
            return external_stream_stack.back().manager->getStream();
        } else {
            int tid = OpenMP::get_thread_num();
            return gpu_stream_pool[gpu_stream_index[tid]].getStream();
        }
    }
#ifdef AMREX_USE_CUDA
    //! Return the CUDA stream for the current thread (for backward compatibility).
    static cudaStream_t cudaStream () noexcept { return gpuStream(); }
#endif
#ifdef AMREX_USE_SYCL
    //! Return the SYCL queue for the current thread's stream.
    static sycl::queue& streamQueue () noexcept { return *(gpuStream().queue); }
    //! Return the SYCL queue for stream \p i.
    static sycl::queue& streamQueue (int i) noexcept { return *(gpu_stream_pool[i].getStream().queue); }
#endif
#endif

    //! Return the number of active GPU streams (1 in a SingleStreamRegion).
    static int numGpuStreams () noexcept {
#ifdef AMREX_USE_GPU
        return (inSingleStreamRegion() || usingExternalStream()) ? 1 : max_gpu_streams;
#else
        return 1;
#endif
    }

    //! Set the current GPU stream index to \p idx for this thread.
    static void setStreamIndex (int idx) noexcept;
    //! Reset the current GPU stream index to 0 for this thread.
    static void resetStreamIndex () noexcept { setStreamIndex(0); }

#ifdef AMREX_USE_GPU
    //! Return the index of stream \p s in the stream pool.
    static int streamIndex (gpuStream_t s = gpuStream()) noexcept;

    //! Activate stream \p s as the current GPU stream; returns the previously active stream.
    static gpuStream_t setStream (gpuStream_t s) noexcept;
    //! Reset the current GPU stream to stream 0; returns the previously active stream.
    static gpuStream_t resetStream () noexcept;

    /** Override the current GPU stream with a user supplied stream. */
    static void setExternalStream (gpuStream_t s);
    /** Pop the current external stream and optionally synchronize it. */
    static void resetExternalStream (ExternalStreamSync sync_stream = ExternalStreamSync::Yes) noexcept;
    static bool usingExternalStream () noexcept;
#endif

    //! Return the device ID of the GPU currently in use.
    static int deviceId () noexcept;
    //! Return the total number of GPU devices used across all MPI ranks.
    static int numDevicesUsed () noexcept;
    //! Return the number of MPI ranks sharing this GPU device.
    static int numDevicePartners () noexcept;

    /**
     * \ingroup amrex_execution
     * \brief Halt execution until the GPU has finished all previously requested tasks.
     */
    static void synchronize () noexcept;

    /**
     * \ingroup amrex_execution
     * \brief Halt execution until the current AMReX GPU stream has finished all pending tasks.
     */
    static void streamSynchronize () noexcept;
#ifdef AMREX_USE_GPU
    static void streamSynchronize (gpuStream_t s) noexcept;
#endif

    /**
     * \ingroup amrex_execution
     * \brief Halt execution until all active AMReX GPU streams have finished all pending tasks.
     *
     * If we are in single stream region, only stream 0 is considered
     * active. Otherwise all streams are active.
     */
    static void streamSynchronizeActive () noexcept;

    /**
     * \ingroup amrex_execution
     * \brief Halt execution until all AMReX GPU streams have finished all pending tasks.
     */
    static void streamSynchronizeAll () noexcept;

    //! Schedule \p mem (from \p arena) for deallocation after the current stream completes.
    static void freeAsync (Arena* arena, void* mem) noexcept;
#ifdef AMREX_USE_GPU
    static void streamOrderedFreeAsync (Arena* arena, void* mem, gpuStream_t stream) noexcept;
#endif

    //! Flush the deferred-free buffer; returns true if any memory was freed.
    static bool clearFreeAsyncBuffer () noexcept;

#if defined(__CUDACC__)
    //! Begin CUDA graph recording; call with \p first_iter=true on the first iteration.
    static void startGraphRecording(bool first_iter, void* h_ptr, void* d_ptr, size_t sz);
    //! End CUDA graph recording and return the executable graph; call with \p last_iter=true on the last iteration.
    static cudaGraphExec_t stopGraphRecording(bool last_iter);

    //! Instantiate a CUDA graph into an executable form.
    static cudaGraphExec_t instantiateGraph(cudaGraph_t graph);

    //! Execute an instantiated CUDA graph, optionally synchronizing afterwards.
    static void executeGraph(const cudaGraphExec_t &graphExec, bool synch = true);

#endif

    //! Advise the driver to migrate pages of \p sz bytes at \p p to \p device.
    static void mem_advise_set_preferred (void* p, std::size_t sz, int device);
    //! Mark \p sz bytes at \p p as read-only for migration advice.
    static void mem_advise_set_readonly (void* p, std::size_t sz);

#ifdef AMREX_USE_GPU
    //! Set the minimum number of threads per block in each dimension.
    static void setNumThreadsMin (int nx, int ny, int nz) noexcept;
    //! Compute grid and block dimensions for a 1-D loop of \p N iterations.
    static void n_threads_and_blocks (const Long N, dim3& numBlocks, dim3& numThreads) noexcept;
    //! Compute grid/block dims for a box [\p lo, \p hi] with \p comps components.
    static void c_comps_threads_and_blocks (const int* lo, const int* hi, const int comps,
                                            dim3& numBlocks, dim3& numThreads) noexcept;
    //! Compute grid/block dims for a box [\p lo, \p hi].
    static void c_threads_and_blocks (const int* lo, const int* hi, dim3& numBlocks, dim3& numThreads) noexcept;
    //! Compute grid/block dims for a grid-stride loop.
    static void grid_stride_threads_and_blocks (dim3& numBlocks, dim3& numThreads) noexcept;

    //! Return total global device memory in bytes.
    static std::size_t totalGlobalMem () noexcept { return device_prop.totalGlobalMem; }
    //! Return shared memory per block in bytes.
    static std::size_t sharedMemPerBlock () noexcept { return device_prop.sharedMemPerBlock; }
    //! Return the number of multiprocessors (SMs) on the device.
    static int numMultiProcessors () noexcept { return device_prop.multiProcessorCount; }
    //! Return the maximum number of threads per multiprocessor.
    static int maxThreadsPerMultiProcessor () noexcept { return device_prop.maxThreadsPerMultiProcessor; }
    //! Return the maximum number of threads per block.
    static int maxThreadsPerBlock () noexcept { return device_prop.maxThreadsPerBlock; }
    //! Return the maximum number of threads per block along dimension \p dir.
    static int maxThreadsPerBlock (int dir) noexcept { return device_prop.maxThreadsDim[dir]; }
    //! Return the maximum number of blocks in the grid along dimension \p dir.
    static int maxBlocksPerGrid (int dir) noexcept { return device_prop.maxGridSize[dir]; }
    //! Return the device name string.
    static std::string deviceName () noexcept { return std::string(device_prop.name); }
#endif

#ifdef AMREX_USE_CUDA
    //! Return the CUDA compute capability major version.
    static int devicePropMajor () noexcept { return device_prop.major; }
    //! Return the CUDA compute capability minor version.
    static int devicePropMinor () noexcept { return device_prop.minor; }
#endif

    //! Return a string identifying the GPU vendor (AMD, NVIDIA, or the SYCL vendor string).
    static std::string deviceVendor() noexcept
    {
#if defined(AMREX_USE_HIP) && defined(__HIP_PLATFORM_AMD__)
        return std::string("AMD");
#elif defined(AMREX_USE_CUDA) || (defined(AMREX_USE_HIP) && defined(__HIP_PLATFORM_NVIDIA__))
        // Using HIP on NVIDIA GPUs isn't currently supported by AMReX
        return std::string("NVIDIA");
#elif defined(AMREX_USE_SYCL)
        return device_prop.vendor;
#else
        return std::string("Unknown");
#endif
    }

    //! Return the amount of free device memory in bytes.
    static std::size_t freeMemAvailable ();
    //! Start the GPU profiler (e.g., nvtx range).
    static void profilerStart ();
    //! Stop the GPU profiler.
    static void profilerStop ();

#ifdef AMREX_USE_GPU

    //! Return non-zero if managed memory is supported on this device.
    static int managedMemorySupported () noexcept { return device_prop.managedMemory; }

    //! Return non-zero if CUDA/HIP memory pools are supported on this device.
    static int memoryPoolsSupported () noexcept { return memory_pools_supported; }

#if defined(AMREX_USE_HIP)
    static AMREX_EXPORT constexpr int warp_size = AMREX_AMDGCN_WAVEFRONT_SIZE;
#elif defined(AMREX_USE_SYCL)
    static AMREX_EXPORT constexpr int warp_size = AMREX_SYCL_SUB_GROUP_SIZE;
#else
    static AMREX_EXPORT constexpr int warp_size = AMREX_HIP_OR_CUDA(64,32);
#endif

    //! Return the maximum number of thread blocks per kernel launch.
    static unsigned int maxBlocksPerLaunch () noexcept { return max_blocks_per_launch; }

#ifdef AMREX_USE_SYCL
    //! Return the maximum single allocation size in bytes (SYCL only).
    static Long maxMemAllocSize () noexcept { return device_prop.maxMemAllocSize; }
    //! Return a reference to the SYCL context.
    static sycl::context& syclContext () { return *sycl_context; }
    //! Return a reference to the SYCL device.
    static sycl::device& syclDevice () { return *sycl_device; }
#endif
#endif

private:

    static void initialize_gpu (bool minimal);

    static AMREX_EXPORT int device_id;
    static AMREX_EXPORT int num_devices_used;
    static AMREX_EXPORT int num_device_partners;
    static AMREX_EXPORT int verbose;
    static AMREX_EXPORT int max_gpu_streams;

#ifdef AMREX_USE_GPU
    static AMREX_EXPORT dim3 numThreadsMin;
    static AMREX_EXPORT dim3 numBlocksOverride, numThreadsOverride;

    static AMREX_EXPORT Vector<StreamManager> gpu_stream_pool; // The size of this is max_gpu_stream
    // The non-owning gpu_stream_index is used to store the current stream index that will be used.
    // gpu_stream_index is a vector so that it's thread safe to write to it.
    static AMREX_EXPORT Vector<int> gpu_stream_index; // The size of this is omp_max_threads
    static AMREX_EXPORT gpuDeviceProp_t device_prop;
    static AMREX_EXPORT int memory_pools_supported;
    static AMREX_EXPORT unsigned int max_blocks_per_launch;

    struct ExternalStream
    {
        std::unique_ptr<StreamManager> manager;
        int saved_stream_index = 0;
    };
    static AMREX_EXPORT Vector<ExternalStream> external_stream_stack;

#ifdef AMREX_USE_SYCL
    static AMREX_EXPORT std::unique_ptr<sycl::context> sycl_context;
    static AMREX_EXPORT std::unique_ptr<sycl::device>  sycl_device;
#endif

    friend StreamManager;
    friend struct ExternalGpuStreamRegion;
#endif
};

// Put these in amrex::Gpu

#if defined(AMREX_USE_GPU)
/// \ingroup amrex_execution
//! Return the GPU stream for the current thread.
inline gpuStream_t
gpuStream () noexcept
{
    return Device::gpuStream();
}
#endif

/// \ingroup amrex_execution
//! Return the number of active GPU streams (1 in a SingleStreamRegion).
inline int
numGpuStreams () noexcept
{
    return Device::numGpuStreams();
}

/// \ingroup amrex_execution
//! Synchronize the device; wait for all pending GPU work to complete.
inline void
synchronize () noexcept
{
    Device::synchronize();
}

/// \ingroup amrex_execution
//! Synchronize the current GPU stream; wait for all work on it to complete.
inline void
streamSynchronize () noexcept
{
    Device::streamSynchronize();
}

#ifdef AMREX_USE_GPU
/// \ingroup amrex_execution
//! Synchronize the given GPU stream; wait for all work on it to complete.
inline void
streamSynchronize (gpuStream_t s) noexcept
{
    Device::streamSynchronize(s);
}
#endif

/// \ingroup amrex_execution
//! Synchronize all GPU streams; wait for all work on every stream to complete.
inline void
streamSynchronizeActive () noexcept
{
    Device::streamSynchronizeActive();
}

inline void
streamSynchronizeAll () noexcept
{
    Device::streamSynchronizeAll();
}

/**
 * \ingroup amrex_execution
 * rief Deallocate memory belonging to an arena asynchronously.
 *
 * Memory held in a pool will not be reused until the next
 * amrex::Gpu::streamSynchronize(). GPU kernels already launched on the
 * currently active stream may still access the memory after this call.
 * Not needed for CPU-only memory or with The_Async_Arena.
 *
 * \param[in] arena the arena the memory belongs to
 * \param[in] mem pointer to the memory to be freed
 */
inline void
freeAsync (Arena* arena, void* mem) noexcept
{
    Device::freeAsync(arena, mem);
}

/**
 * \ingroup amrex_execution
 * \brief Clear the temporary buffer used by freeAsync.
 *
 * Synchronizes the stream if memory is waiting to be freed. Returns true if
 * some memory was freed.
 */
inline bool
clearFreeAsyncBuffer () noexcept
{
    return Device::clearFreeAsyncBuffer();
}

#ifdef AMREX_USE_GPU

/** \brief Provide a user-supplied GPU stream.
 *
 *  Nested calls form a stack.  Each new call makes the supplied stream
 *  current for AMReX GPU launches until resetExternalGpuStream pops it and
 *  restores the previous external stream, if any.  Calling this function
 *  inside an OpenMP parallel region will trigger an assertion.
 */
inline void
setExternalGpuStream (gpuStream_t stream)
{
    Device::setExternalStream(stream);
}

/** \brief Pop the current user-supplied GPU stream.
 *
 *  Must be called once for every preceding successful setExternalGpuStream
 *  invocation (RAII helpers do this automatically).  When ``sync_stream``
 *  is :cpp:`ExternalStreamSync::No` AMReX skips the stream-wide synchronize
 *  unless deferred frees recorded by The_Async_Arena are pending, in which
 *  case a synchronization is still forced for correctness.  If an older
 *  external stream was active before the current one, it becomes current
 *  again after the pop.  Not supported inside OpenMP parallel regions.
 */
inline void
resetExternalGpuStream (ExternalStreamSync sync_stream = ExternalStreamSync::Yes) noexcept
{
    Device::resetExternalStream(sync_stream);
}

/** RAII helper that overrides the GPU stream for the current scope (optionally
 *  skipping the synchronization on destruction when ``sync_on_exit`` is
 *  :cpp:`ExternalStreamSync::No`).
 */
struct [[nodiscard]] ExternalGpuStreamRegion
{
    explicit ExternalGpuStreamRegion (gpuStream_t stream,
                                      ExternalStreamSync sync_on_exit = ExternalStreamSync::Yes) noexcept
        : m_sync_on_exit(sync_on_exit)
    {
        Device::setExternalStream(stream);
    }

    ExternalGpuStreamRegion (ExternalGpuStreamRegion&& rhs) noexcept = delete;
    ExternalGpuStreamRegion& operator= (ExternalGpuStreamRegion&& rhs) noexcept = delete;
    ExternalGpuStreamRegion (ExternalGpuStreamRegion const&) = delete;
    ExternalGpuStreamRegion& operator= (ExternalGpuStreamRegion const&) = delete;

    ~ExternalGpuStreamRegion ()
    {
        Device::resetExternalStream(m_sync_on_exit);
    }

private:
    ExternalStreamSync m_sync_on_exit = ExternalStreamSync::Yes;
};

/// \ingroup amrex_execution
//! Asynchronously copy \p sz bytes from host pointer \p p_h to device pointer \p p_d.
inline void
htod_memcpy_async (void* p_d, const void* p_h, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
#ifdef AMREX_USE_SYCL
    auto& q = Device::streamQueue();
    q.submit([&] (sycl::handler& h) { h.memcpy(p_d, p_h, sz); });
#else
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpyAsync(p_d, p_h, sz, hipMemcpyHostToDevice, gpuStream()));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpyAsync(p_d, p_h, sz, cudaMemcpyHostToDevice, gpuStream())); )
#endif
}

/// \ingroup amrex_execution
//! Asynchronously copy \p sz bytes from device pointer \p p_d to host pointer \p p_h.
inline void
dtoh_memcpy_async (void* p_h, const void* p_d, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
#ifdef AMREX_USE_SYCL
    auto& q = Device::streamQueue();
    q.submit([&] (sycl::handler& h) { h.memcpy(p_h, p_d, sz); });
#else
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpyAsync(p_h, p_d, sz, hipMemcpyDeviceToHost, gpuStream()));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpyAsync(p_h, p_d, sz, cudaMemcpyDeviceToHost, gpuStream())); )
#endif
}

/// \ingroup amrex_execution
//! Asynchronously copy \p sz bytes from device pointer \p p_d_src to device pointer \p p_d_dst.
inline void
dtod_memcpy_async (void* p_d_dst, const void* p_d_src, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
#ifdef AMREX_USE_SYCL
    auto& q = Device::streamQueue();
    q.submit([&] (sycl::handler& h) { h.memcpy(p_d_dst, p_d_src, sz); });
#else
    AMREX_HIP_OR_CUDA(
        AMREX_HIP_SAFE_CALL(hipMemcpyAsync(p_d_dst, p_d_src, sz, hipMemcpyDeviceToDevice, gpuStream()));,
        AMREX_CUDA_SAFE_CALL(cudaMemcpyAsync(p_d_dst, p_d_src, sz, cudaMemcpyDeviceToDevice, gpuStream())); )
#endif
}

/// \cond DOXYGEN_IGNORE
#else // AMREX_USE_GPU

inline void
htod_memcpy_async (void* p_d, const void* p_h, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
    std::memcpy(p_d, p_h, sz);
}

inline void
dtoh_memcpy_async (void* p_h, const void* p_d, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
    std::memcpy(p_h, p_d, sz);
}

inline void
dtod_memcpy_async (void* p_d_dst, const void* p_d_src, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
    std::memcpy(p_d_dst, p_d_src, sz);
}

#endif // AMREX_USE_GPU
/// \endcond

/// \ingroup amrex_execution
//! Synchronously copy \p sz bytes from host pointer \p p_h to device pointer \p p_d.
inline void
htod_memcpy (void* p_d, const void* p_h, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
    htod_memcpy_async(p_d, p_h, sz);
    Gpu::streamSynchronize();
}

/// \ingroup amrex_execution
//! Synchronously copy \p sz bytes from device pointer \p p_d to host pointer \p p_h.
inline void
dtoh_memcpy (void* p_h, const void* p_d, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
    dtoh_memcpy_async(p_h, p_d, sz);
    Gpu::streamSynchronize();
}

/// \ingroup amrex_execution
//! Synchronously copy \p sz bytes from device pointer \p p_d_src to device pointer \p p_d_dst.
inline void
dtod_memcpy (void* p_d_dst, const void* p_d_src, const std::size_t sz) noexcept
{
    if (sz == 0) { return; }
    dtod_memcpy_async(p_d_dst, p_d_src, sz);
    Gpu::streamSynchronize();
}

#ifdef AMREX_USE_HYPRE
/// \ingroup amrex_execution
//! Synchronize HYPRE operations with the GPU.
void hypreSynchronize ();
#endif

/// \ingroup amrex_execution
//! Copy \p nbytes bytes from host to device global variable \p dg starting at byte \p offset.
template <typename T>
void memcpy_from_host_to_device_global_async (T& dg, const void* src,
                                              std::size_t nbytes,
                                              std::size_t offset = 0)
{
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaMemcpyToSymbolAsync(dg, src, nbytes, offset,
                                                 cudaMemcpyHostToDevice,
                                                 Device::gpuStream()));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipMemcpyToSymbolAsync(dg, src, nbytes, offset,
                                               hipMemcpyHostToDevice,
                                               Device::gpuStream()));
#elif defined(AMREX_USE_SYCL)
    Device::streamQueue().memcpy(dg, src, nbytes, offset);
#else
    auto* p = (char*)(&dg);
    std::memcpy(p+offset, src, nbytes);
#endif
}

/// \ingroup amrex_execution
//! Copy \p nbytes bytes from device global variable \p dg to host pointer \p dst starting at byte \p offset.
template <typename T>
void memcpy_from_device_global_to_host_async (void* dst, T const& dg,
                                              std::size_t nbytes,
                                              std::size_t offset = 0)
{
#if defined(AMREX_USE_CUDA)
    AMREX_CUDA_SAFE_CALL(cudaMemcpyFromSymbolAsync(dst, dg, nbytes, offset,
                                                   cudaMemcpyDeviceToHost,
                                                   Device::gpuStream()));
#elif defined(AMREX_USE_HIP)
    AMREX_HIP_SAFE_CALL(hipMemcpyFromSymbolAsync(dst, dg, nbytes, offset,
                                                  hipMemcpyDeviceToHost,
                                                  Device::gpuStream()));
#elif defined(AMREX_USE_SYCL)
    Device::streamQueue().memcpy(dst, dg, nbytes, offset);
#else
    auto const* p = (char const*)(&dg);
    std::memcpy(dst, p+offset, nbytes);
#endif
}

#ifdef AMREX_USE_GPU

//! Check if the given stream is active.
bool isStreamActive (gpuStream_t s);

/**
 * This struct provides a RAII-style mechanism for disabling GPU
 * synchronization and calling Gpu::streamSynchronizeActive when going out
 * of scope if we are in sync region previously. Note that explicit calls to
 * Gpu::streamSynchronize and Gpu::synchronize still work.
 */
struct [[nodiscard]] SyncAtExitOnly
    : public NoSyncRegion
{
    ~SyncAtExitOnly () {
        if (! inNoSyncRegionPreviously()) {
            Gpu::streamSynchronizeActive();
        }
    }
};

#else

struct [[nodiscard]] SyncAtExitOnly {};

#endif

}

#endif
