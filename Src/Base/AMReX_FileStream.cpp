#include <AMReX_FileStream.H>
#include <AMReX_ParmParse.H>

#ifndef _WIN32

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <pthread.h>
#include <thread>
#include <chrono>

namespace amrex
{

struct WriteInfo {
    int fd;
    const char* buf;
    size_t count;
    ssize_t result;
    bool done = false;
};

void* write_thread_func(void* arg)
{
    // Enable cancellation
    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, nullptr);
    pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, nullptr);

    auto* info = static_cast<WriteInfo*>(arg);
    ssize_t n = ::write(info->fd, info->buf, info->count);
    info->result = n;
    info->done = true;
    return nullptr;
}

ssize_t timed_write(int fd, const char* buf, size_t count, int timeout_sec)
{
    WriteInfo info = {fd, buf, count, -1, false};
    pthread_t tid;

    if (pthread_create(&tid, nullptr, write_thread_func, &info) != 0) {
        std::cerr << "Failed to create thread\n";
        return -1;
    }

    // Wait for completion or timeout
    for (int i = 0; i < timeout_sec * 10; ++i) {
        if (info.done) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!info.done) {
        std::cerr << "Timeout occurred, cancelling thread\n";
        pthread_cancel(tid);
        pthread_join(tid, nullptr);
        errno = ETIMEDOUT;
        return -1;
    } else {
        pthread_join(tid, nullptr);  // clean up
        return info.result;
    }
}

FileStream::FileStream (char const* filename, std::ios_base::openmode mode)
{
    open(filename, mode);
}

FileStream::FileStream (std::string const& filename, std::ios_base::openmode mode)
{
    open(filename.c_str(), mode);
}

FileStream::FileStream (FileStream&& rhs) noexcept
    : m_base_delay   (rhs.m_base_delay),
      m_pos          (rhs.m_pos),
      m_size         (rhs.m_size),
      m_max_retries  (rhs.m_max_retries),
      m_fd           (std::exchange(rhs.m_fd,-1)),
      m_good         (std::exchange(rhs.m_good,false)),
      m_binary       (rhs.m_binary),
      m_append       (rhs.m_append),
      m_buffer       (std::exchange(rhs.m_buffer,nullptr)),
      m_buffer_size  (rhs.m_buffer_size),
      m_buffer_begin (rhs.m_buffer_begin),
      m_buffer_end   (rhs.m_buffer_end),
      m_buffer_mode  (rhs.m_buffer_mode)
{}

FileStream& FileStream::operator= (FileStream&& rhs) noexcept
{
    if (this != &rhs) {
        std::swap(m_base_delay  , rhs.m_base_delay);
        std::swap(m_pos         , rhs.m_pos);
        std::swap(m_size        , rhs.m_size);
        std::swap(m_max_retries , rhs.m_max_retries);
        std::swap(m_good        , rhs.m_good);
        std::swap(m_binary      , rhs.m_binary);
        std::swap(m_append      , rhs.m_append);
        std::swap(m_fd          , rhs.m_fd);
        std::swap(m_buffer      , rhs.m_buffer);
        std::swap(m_buffer_size , rhs.m_buffer_size);
        std::swap(m_buffer_begin, rhs.m_buffer_begin);
        std::swap(m_buffer_end  , rhs.m_buffer_end);
        std::swap(m_buffer_mode , rhs.m_buffer_mode);
    }
    return *this;
}

FileStream::~FileStream ()
{
    try {
        close();
    } catch (std::runtime_error& e) {
        amrex::Error_host("Error", e.what(), false);
    }
}

void FileStream::open (char const* filename, std::ios_base::openmode mode)
{
    int flags = 0; // O_SYNC;
    if ((mode & std::ios_base::in ) != 0 &&
        (mode & std::ios_base::out) != 0) {
        flags |= O_RDWR | O_CREAT;
    } else if ((mode & std::ios_base::in) != 0) {
        flags |= O_RDONLY;
    } else if ((mode & std::ios_base::out) != 0) {
        flags |= O_WRONLY | O_CREAT;
    }
    if ((mode & std::ios_base::app) != 0) {
        flags |= O_APPEND;
    } else if ((mode & std::ios_base::trunc) != 0) {
        flags |= O_TRUNC;
    } else if (((mode & std::ios_base::out) != 0) && ((mode & std::ios_base::in) == 0)) {
        flags |= O_TRUNC;
    }
    m_binary = (mode & std::ios_base::binary) != 0;
    m_append = (mode & std::ios_base::app) != 0;

//    amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                      << ": before ::open\n";

    mode_t mod = 0666;
    execute_with_retry([&]() {
        m_fd = ::open(filename, flags, mod);
        if (m_fd != -1) {
            m_good = true;
            m_pos = 0;
            return true;
        } else {
            m_good = false;
            return false;
        }
    }, "File open");

//    amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                      << ": after ::open before flock\n";

    if (m_good) {

        execute_with_retry([&]() {
            auto r = ::flock(m_fd, LOCK_EX);
            if (r == 0) {
                return true;
            } else {
                m_good = false;
                return false;
            }
        }, "Lock");

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": after flock\n";
    }


    if (m_good && (((mode & std::ios_base::ate) != 0) || m_append)) {

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": before ::lseed\n";

        execute_with_retry([&]() {
            auto end_pos = ::lseek(m_fd, 0, SEEK_END);
            if (end_pos >= 0) {
                m_pos = end_pos;
                m_size = end_pos;
                return true;
            } else {
                m_good = false;
                return false;
            }
        }, "Seek to end in ate or app mode");

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": after ::lseek\n";
    }

    if (m_buffer == nullptr) {
        m_buffer.reset(new char[m_buffer_size]); // Do NOT use std::make_unique for performance reasons
    }
    m_buffer_mode = BufferMode::None;
    m_buffer_begin = m_buffer_end = 0;
}

void FileStream::open (std::string const& filename, std::ios_base::openmode mode)
{
    open(filename.c_str(), mode);
}

void FileStream::close ()
{
    if (m_fd != -1) {
        try {
            flush();
        } catch (std::exception const& e) {
            if (amrex::Verbose()) {
                amrex::Warning(std::string("FileStream::close: flush failed before close: ")
                               + e.what());
            }
        }

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": before ::flock\n";

        if (::flock(m_fd, LOCK_UN) != 0) {
            amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
                              << ": unlock failed "
                              << std::strerror(errno) << "\n";
        }

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": after::flock before ::close\n";

        int r = ::close(m_fd);

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": after ::close\n";

        m_fd = -1;
        if (r != 0) {
            m_good = false;
            if (errno != EINTR) { // Could be harmless
                throw std::runtime_error(std::string("FileStream::close failed: ")
                                         + std::strerror(errno));
            }
        }
    }
    m_good = false;
}

// xxxx TODO: not tested yet
FileStream& FileStream::read (char* s, std::streamsize count)
{
    if (count == 0) { return *this; }

    if (m_fd == -1 || !m_good) {
        throw std::runtime_error("FileStream::read: bad file descriptor or bad state");
    }

    if (m_buffer_mode == BufferMode::Write) {
        try {
            flush_write_buffer(); // This sets buffer mode to None
        } catch (...) {
            m_good = false;
            throw;
        }
    }

    std::streamsize total_read = 0;
    while (total_read < count)
    {
        if (m_buffer_mode == BufferMode::None) {
            fill_read_buffer(); // This set buffer mode to Read
        }

        // Now the buffer mode is Read.
        if (m_buffer_end == m_buffer_begin) { break; } // EOF

        auto available = m_buffer_end - m_buffer_begin;
        auto to_copy = std::min(available, count-total_read);

        std::memcpy(s + total_read, m_buffer.get() + m_buffer_begin, to_copy);
        m_buffer_begin += to_copy;
        m_pos          += to_copy;
        total_read     += to_copy;

        if (m_buffer_begin == m_buffer_end) {
            m_buffer_mode = BufferMode::None;
            m_buffer_begin = m_buffer_end = 0;
        }
    }

    return *this;
}

FileStream& FileStream::write (char const* s, std::streamsize count)
{
    if (count <= 0) { return *this; }

    if (m_fd == -1 || !m_good) {
        throw std::runtime_error("FileStream::write: bad file descriptor or bad state");
    }

//    amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                      << ": beg of write()\n";

    if (m_append && (m_pos != m_size)) {
        this->seekp(0, std::ios_base::end);
    }

    if (m_buffer_mode == BufferMode::Read) {
        if (m_buffer_begin < m_buffer_end) {
            off_type unread = m_buffer_end - m_buffer_begin;
            m_buffer_mode = BufferMode::None;
            m_buffer_begin = m_buffer_end = 0;
            this->seekp(-unread, std::ios_base::cur); // go back
        } else {
            m_buffer_mode = BufferMode::None;
            m_buffer_begin = m_buffer_end = 0;
        }
    }

    if (m_buffer_mode == BufferMode::None) {
        if (count <= m_buffer_size) {
            fill_write_buffer(s, count);
        } else {
            file_write(s, count);
        }
    } else { // Write mode
        std::streamsize total_written = 0;
        char const* src = s;
        while (total_written < count) {
            std::streamsize remaining = count - total_written;

            auto space_left = m_buffer_size - m_buffer_end;
            if (space_left == 0) {
                try {
                    flush_write_buffer(); // this sets buffer mode to None
                } catch (...) {
                    m_good = false;
                    throw;
                }
                space_left = m_buffer_size;
            }

#if 0
// xxxxx
            if ((m_buffer_mode == BufferMode::None) && (remaining > 2*m_buffer_size)) {
                file_write(src, remaining);
                // no need to update total_written due to break
                break;
            } else
#endif
            {
                auto to_buffer = std::min(remaining, space_left);
                fill_write_buffer(src, to_buffer);
                src           += to_buffer;
                total_written += to_buffer;
            }
        }
    }

    m_pos += count;
    m_size += count;

//    amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                      << ": end of write()\n";

    return *this;
}

FileStream& FileStream::flush ()
{
    if (m_fd != -1 && m_good) {
        if (m_buffer_mode == BufferMode::Write) {
            try {
                flush_write_buffer();
            } catch (...) {
                m_good = false;
                throw;
            }
        }
    }
    return *this;
}

FileStream& FileStream::seekp (pos_type off)
{
    return seekp(off, std::ios_base::beg);
}

FileStream& FileStream::seekp (off_type off, std::ios_base::seekdir dir)
{
    if (m_fd != -1 && m_good)
    {
        // reset buffer
        if (m_buffer_mode == BufferMode::Write) {
            try {
                flush_write_buffer();
            } catch (...) {
                m_good = false;
                throw;
            }
        } else if (m_buffer_mode == BufferMode::Read) {
            if ((dir == std::ios_base::cur) && (m_buffer_begin < m_buffer_end)) {
                off_type unread = m_buffer_end - m_buffer_begin;
                off -= unread;
            }
            m_buffer_mode = BufferMode::None;
            m_buffer_begin = m_buffer_end = 0;
        }

        int whence;
        switch (dir) {
            case std::ios_base::beg: { whence = SEEK_SET; break; }
            case std::ios_base::cur: { whence = SEEK_CUR; break; }
            case std::ios_base::end: { whence = SEEK_END; break; }
            default: {
                m_good = false;
                return *this;
            }
        }

        execute_with_retry([&]() {
            off_t new_pos = ::lseek(m_fd, off, whence);
            if (new_pos != -1) {
                m_pos = new_pos;
                return true;
            } else {
                m_good = false;
                return false;
            }
        }, "Seek position");
    }
    return *this;
}

FileStream& FileStream::seekg (pos_type off)
{
    return seekg(off, std::ios_base::beg);
}

FileStream& FileStream::seekg (off_type off, std::ios_base::seekdir dir)
{
    return seekp(off,dir);
}

FileStream::pos_type FileStream::tellp () const
{
    return m_pos;
}

FileStream::pos_type FileStream::tellg () const
{
    return m_pos;
}

bool FileStream::good () const
{
    return m_good && m_fd != -1;
}

bool FileStream::bad () const
{
    return !m_good || m_fd == -1;
}

bool FileStream::fail () const
{
    return !m_good || m_fd == -1;
}

void FileStream::file_write (char const* s, Long count)
{
    if (count == 0) { return; }

    if (m_fd == -1 || !m_good) {
        throw std::runtime_error("FileStream::file_write: bad file descriptor or bad state");
    }

    static int write_timeout = 600;
    static std::string command;
    if (command.empty()) {
        ParmParse pp;
        pp.get("write_timeout", write_timeout);
        std::string job_id;
        pp.get("job_id", job_id);
        command. append("scancel ").append(job_id);
    }

    static double twrite_max = 0;

    double t0 = amrex::second();

    Long total_written = 0;
    while (total_written < count) {
        ssize_t nbytes_written = -1;

        struct stat sb;
        if (::fstat(m_fd, &sb) != 0) {
            amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
                              << ": ::fstat failed " << std::strerror(errno) << "\n";
        }

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": before ::write " << count-total_written << "\n";

        execute_with_retry([&]() {
            nbytes_written = timed_write(m_fd, s + total_written, count - total_written, write_timeout);
            if (nbytes_written >= 0) {
                return true;
            } else {
                amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
                                  << ": ::write failed. About to run "
                                  << command <<"\n";
                std::system(command.c_str());
                amrex::Sleep(300);
                amrex::Abort("timed out");
                return false;
            }
        }, "Write");

//        ::fsync(m_fd);

//        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
//                          << ": after ::write " << nbytes_written << "\n";

        if (nbytes_written >= 0) {
            total_written += nbytes_written;
        } else {
            throw std::runtime_error("FileStream: write failed");
        }
    }

    double t1 = amrex::second();

    if (t1-t0 > twrite_max) {
        amrex::AllPrint() << "xxxxx Proc. " << ParallelDescriptor::MyProc()
                          << ": write time max: " << t1-t0 << "\n";
        twrite_max = t1-t0;
    }
}

void FileStream::flush_write_buffer ()
{
    if (m_buffer_end == 0) { return; }
    file_write(m_buffer.get(), m_buffer_end);
    m_buffer_mode = BufferMode::None;
    m_buffer_end = 0;
}

void FileStream::fill_write_buffer (char const* s, Long count)
{
    std::memcpy(m_buffer.get()+m_buffer_end, s, count);
    m_buffer_mode = BufferMode::Write;
    m_buffer_end += count;
}

void FileStream::fill_read_buffer ()
{
    if (m_fd == -1 || !m_good) {
        throw std::runtime_error("FileStream::fill_read_buffer: bad file descriptor or bad state");
    }

    m_buffer_begin = 0;
    m_buffer_end = 0;
    while (m_buffer_end < m_buffer_size) {
        ssize_t nbytes_read = -1;
        execute_with_retry([&]() {
            nbytes_read = ::read(m_fd, m_buffer.get() + m_buffer_end,
                                 m_buffer_size - m_buffer_end);
            if (nbytes_read >= 0) {
                return true;
            } else {
                return false;
            }
        }, "Read to buffer");
        if (nbytes_read > 0) {
            m_buffer_end += nbytes_read;
        } else if (nbytes_read == 0) {
            break; // EOF
        } else {
            throw std::runtime_error("FileStream::fill_read_buffer failed");
        }
    }
    m_buffer_mode = BufferMode::Read;
}

}

#endif
