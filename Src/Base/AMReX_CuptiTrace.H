#ifndef AMREX_CUPTI_TRACE_H_
#define AMREX_CUPTI_TRACE_H_
#include <AMReX_Config.H>

#ifdef AMREX_USE_CUPTI
#include <stdio.h>
#include <vector>
#include <memory>
#include <cuda.h>
#include <cupti.h>

/**
 * \file AMReX_CuptiTrace.H
 * \brief CUPTI-based kernel tracing hooks used by the AMReX profiler.
 */

namespace amrex {

void CUPTIAPI bfrRequestCallback (uint8_t* *bfr, size_t* size,
                                  size_t* maxNumRecords) noexcept;
void CUPTIAPI bfrCompleteCallback (CUcontext ctx, uint32_t streamId, uint8_t* bfr,
                                   size_t size, size_t validSize) noexcept;

/** \brief Initialize CUPTI tracing buffers and callbacks. */
void initCuptiTrace () noexcept;
/** \brief Enable CUPTI callbacks and begin tracing GPU activity. */
void cuptiTraceStart () noexcept;
/** \brief Stop tracing and process collected records. */
void cuptiTraceStop () noexcept;
/** \brief Stop tracing for a specific box identifier. */
void cuptiTraceStop (unsigned boxUintID) noexcept;

/**
 * \class amrex::CUpti_Activity_Userdata
 * \brief Stores metadata for a single CUPTI activity record.
 */
class CUpti_Activity_Userdata {
private:
    unsigned uintID_;
    std::string charID_;
    unsigned long long startTime_;
    unsigned long long endTime_;
    unsigned long long timeElapsed_;
    int streamID_;
    std::string name_;

public:
    //! Set the numeric identifier associated with the record (e.g., MFIter index).
    void setUintID (unsigned) noexcept;
    //! Set the human-readable identifier string for this record.
    void setCharID (std::string) noexcept;
    //! Record the start timestamp (in CUPTI timestamp units).
    void setStartTime (unsigned long long) noexcept;
    //! Record the end timestamp (in CUPTI timestamp units).
    void setEndTime (unsigned long long) noexcept;
    //! Cache the elapsed time computed from start/end.
    void setTimeElapsed (unsigned long long) noexcept;
    //! Set the CUDA stream id that executed the activity.
    void setStreamID (int) noexcept;
    //! Set a descriptive kernel or API name.
    void setName (std::string) noexcept;
    //! Retrieve the numeric identifier.
    unsigned getUintID () const noexcept;
    //! Retrieve the human-readable identifier.
    std::string const& getCharID () const noexcept;
    //! Retrieve the start timestamp.
    unsigned long long getStartTime () const noexcept;
    //! Retrieve the end timestamp.
    unsigned long long getEndTime () const noexcept;
    //! Retrieve the elapsed time (end-start).
    unsigned long long getTimeElapsed () const noexcept;
    //! Retrieve the CUDA stream id.
    int getStreamID () const noexcept;
    //! Retrieve the descriptive name.
    std::string const& getName () const noexcept;
};

/**
 * \class amrex::CuptiTrace
 * \brief RAII wrapper that manages CUPTI lifetime for AMReX profiling.
 */
class CuptiTrace {
public:
    //! Construct the tracer and allocate CUPTI resources.
    CuptiTrace () noexcept;
    //! Destructor releases callbacks and CUPTI buffers.
    ~CuptiTrace () noexcept;
    //! Enable CUPTI tracing globally.
    void start () noexcept;
    //! Disable tracing and flush accumulated records.
    void stop () noexcept;
    //! Disable tracing for a specific box identifier.
    void stop (unsigned boxUintID) noexcept;
};

extern std::vector<std::unique_ptr<CUpti_Activity_Userdata>> activityRecordUserdata;
/**
 * \brief Sum the elapsed time stored in each userdata record.
 * \param activityRecordUserdata Collection of activity metadata.
 * \return Total time (microseconds per CUPTI units) spent across records.
 */
double computeElapsedTimeUserdata(const std::vector<std::unique_ptr<CUpti_Activity_Userdata>>&
                                  activityRecordUserdata) noexcept;

//! Initialize CUPTI (called once per process).
void CuptiInitialize ();
//! Tear down CUPTI resources (matching CuptiInitialize()).
void CuptiFinalize ();
}

#endif
#endif
