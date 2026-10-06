#ifndef AMREX_FB_VARIANTS_H_
#define AMREX_FB_VARIANTS_H_
#include <AMReX_Config.H>

/**
 * \file AMReX_FBVariants.H
 * \brief Experimental FillBoundary code path for GPU-aware MPI, selected by
 *        the `amrex.fb.*` parameters (see FabArrayBase::FBVariants).
 */

#ifdef AMREX_FB_VARIANTS

namespace amrex {

/// \cond DOXYGEN_IGNORE
namespace detail {

template <typename BUF, typename T>
void fbv_pack (TagVector<CommSendBufTag<T>> const& tv, char* dbase, char* hbase,
               int scomp, int ncomp)
{
    detail::ParallelFor_doit(tv,
        [=] AMREX_GPU_DEVICE (
            int icell, int ncells, int i, int j, int k, CommSendBufTag<T> const& tag) noexcept
    {
        if (icell < ncells) {
            char* p = (tag.poff >= 0) ? dbase + tag.poff : hbase + (~tag.poff);
            Array4<BUF> dfab{(BUF*)p, amrex::begin(tag.bx), amrex::end(tag.bx), ncomp};
            for (int n = 0; n < ncomp; ++n) {
                dfab(i,j,k,n) = (BUF)tag.sfab(i,j,k,n+scomp);
            }
        }
    });
}

template <typename BUF, typename T>
void fbv_unpack (TagVector<CommRecvBufTag<T>> const& tv, char const* dbase,
                 char const* hbase, int dcomp, int ncomp)
{
    detail::ParallelFor_doit(tv,
        [=] AMREX_GPU_DEVICE (
            int icell, int ncells, int i, int j, int k, CommRecvBufTag<T> const& tag) noexcept
    {
        if (icell < ncells) {
            char const* p = (tag.poff >= 0) ? dbase + tag.poff : hbase + (~tag.poff);
            Array4<BUF const> sfab{(BUF const*)p, amrex::begin(tag.bx), amrex::end(tag.bx), ncomp};
            for (int n = 0; n < ncomp; ++n) {
                tag.dfab(i,j,k,n+dcomp) = (T)sfab(i,j,k,n);
            }
        }
    });
}

}
/// \endcond

template <class FAB>
template <typename BUF>
FBVEntry<FAB>*
FabArray<FAB>::fbv_get_entry (const FB& TheFB, int ncomp)
{
    RecvSendCopyHandlerKey key{TheFB.m_id, sizeof(BUF), ncomp};
    if (auto it = m_fbv_cache.find(key); it != m_fbv_cache.end()) {
        return it->second.get();
    }

    if (m_fbv_cache.size() >= 32) { m_fbv_cache.clear(); }

    auto const& fbv = FabArrayBase::m_fbv;
    auto e = std::make_unique<FBVEntry<FAB>>();

    auto make_layout = [&] (MapOfCopyComTagContainers const& tags, bool is_send,
                            typename FBVEntry<FAB>::Layout& lo)
    {
        for (auto const& [rank, cctc] : tags) {
            std::size_t nbytes = 0;
            for (auto const& cct : cctc) {
                nbytes += (is_send ? cct.sbox : cct.dbox).numPts() * ncomp * sizeof(BUF);
            }
            std::size_t acd = ParallelDescriptor::sizeof_selected_comm_data_type(nbytes);
            nbytes = amrex::aligned_size(acd, nbytes);
            bool h = (fbv.host_off_node && fbv.off_node(rank))
                || static_cast<Long>(nbytes) < fbv.host_threshold;
            auto& total = h ? lo.host_bytes : lo.dev_bytes;
            total = amrex::aligned_size(std::max(alignof(BUF), acd), total);
            lo.rank.push_back(rank);
            lo.size.push_back(nbytes);
            lo.offset.push_back(total);
            lo.on_host.push_back(h);
            total += nbytes;
        }
    };

    make_layout(*TheFB.m_SndTags, true, e->snd);
    make_layout(*TheFB.m_RcvTags, false, e->rcv);

    auto encode = [] (std::size_t off, bool h) {
        auto poff = static_cast<std::ptrdiff_t>(off);
        return h ? ~poff : poff;
    };

    {
        Vector<CommSendBufTag<value_type>> tags;
        int m = 0;
        for (auto const& kv : *TheFB.m_SndTags) {
            std::size_t off = e->snd.offset[m];
            for (auto const& cct : kv.second) {
                tags.push_back({.sfab = this->const_array(cct.srcIndex),
                                .poff = encode(off, e->snd.on_host[m]), .bx = cct.sbox});
                off += cct.sbox.numPts() * ncomp * sizeof(BUF);
            }
            ++m;
        }
        e->send_tv.define(tags);
    }

    {
        Vector<CommRecvBufTag<value_type>> tags;
        int m = 0;
        for (auto const& kv : *TheFB.m_RcvTags) {
            std::size_t off = e->rcv.offset[m];
            Vector<CommRecvBufTag<value_type>> mtags;
            for (auto const& cct : kv.second) {
                const int li = this->localindex(cct.dstIndex);
                mtags.push_back({.dfab = this->atLocalIdx(li).array(),
                                 .poff = encode(off, e->rcv.on_host[m]), .bx = cct.dbox});
                off += cct.dbox.numPts() * ncomp * sizeof(BUF);
            }
            tags.insert(tags.end(), mtags.begin(), mtags.end());
            if (fbv.unpack_on_arrival) {
                e->recv_tv_msg.emplace_back(mtags);
            }
            ++m;
        }
        e->recv_tv.define(tags);
    }

    if (fbv.persistent) {
        e->unpack_done = detail::fbv_event_create();
    }

    auto* r = e.get();
    m_fbv_cache[key] = std::move(e);
    return r;
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::fbv_nowait (const FB& TheFB, int scomp, int ncomp, int SeqNum)
{
    auto const& fbv = FabArrayBase::m_fbv;
    auto* e = fbv_get_entry<BUF>(TheFB, ncomp);

    const int N_locs = TheFB.m_LocTags->size();
    const auto N_rcvs = static_cast<int>(e->rcv.rank.size());
    const auto N_snds = static_cast<int>(e->snd.rank.size());

    fbd = std::make_unique<FBData<FAB>>();
    fbd->fb    = &TheFB;
    fbd->scomp = scomp;
    fbd->ncomp = ncomp;
    fbd->tag   = SeqNum;
    fbd->fbv_entry = e;

    auto alloc = [] (Arena* ar, std::size_t n) {
        return n > 0 ? static_cast<char*>(ar->alloc(n)) : nullptr;
    };

    if (fbv.persistent) {
        if (e->unpack_pending) {
            BL_PROFILE("FB::wait_prev_unpack");
            detail::fbv_event_sync(e->unpack_done);
            e->unpack_pending = false;
        }
        if (!e->rcv_d) { e->rcv_d = alloc(The_Comms_Arena(), e->rcv.dev_bytes); }
        if (!e->rcv_h) { e->rcv_h = alloc(The_Pinned_Arena(), e->rcv.host_bytes); }
        if (!e->snd_d) { e->snd_d = alloc(The_Comms_Arena(), e->snd.dev_bytes); }
        if (!e->snd_h) { e->snd_h = alloc(The_Pinned_Arena(), e->snd.host_bytes); }
        fbd->fbv_rcv_d = e->rcv_d;
        fbd->fbv_rcv_h = e->rcv_h;
        fbd->fbv_snd_d = e->snd_d;
        fbd->fbv_snd_h = e->snd_h;
    } else {
        fbd->fbv_rcv_d = alloc(The_Comms_Arena(), e->rcv.dev_bytes);
        fbd->fbv_rcv_h = alloc(The_Pinned_Arena(), e->rcv.host_bytes);
        fbd->fbv_snd_d = alloc(The_Comms_Arena(), e->snd.dev_bytes);
        fbd->fbv_snd_h = alloc(The_Pinned_Arena(), e->snd.host_bytes);
    }

    MPI_Comm comm = ParallelContext::CommunicatorSub();

    if (N_rcvs > 0) {
        BL_PROFILE("FB::post_recvs");
        fbd->recv_reqs.assign(N_rcvs, MPI_REQUEST_NULL);
        fbd->recv_stat.resize(N_rcvs);
        fbd->recv_size.resize(N_rcvs);
        for (int i = 0; i < N_rcvs; ++i) {
            fbd->recv_size[i] = e->rcv.size[i];
            char* p = (e->rcv.on_host[i] ? fbd->fbv_rcv_h : fbd->fbv_rcv_d) + e->rcv.offset[i];
            if (e->rcv.size[i] > 0) {
                const int rank = ParallelContext::global_to_local_rank(e->rcv.rank[i]);
                fbd->recv_reqs[i] = ParallelDescriptor::Arecv
                    (p, e->rcv.size[i], rank, SeqNum, comm).req();
            }
        }
    }

    bool local_done = false;

    if (N_snds > 0) {
        {
            BL_PROFILE("FB::pack");
            detail::fbv_pack<BUF>(e->send_tv, fbd->fbv_snd_d, fbd->fbv_snd_h, scomp, ncomp);
        }
        if (fbv.local_first && N_locs > 0) {
            auto ev = detail::fbv_pack_event();
            detail::fbv_event_record(ev);
            {
                BL_PROFILE("FB::local_copy");
                FB_local_copy_gpu(TheFB, scomp, ncomp);
            }
            local_done = true;
            BL_PROFILE("FB::pack_sync");
            detail::fbv_event_sync(ev);
        } else {
            BL_PROFILE("FB::pack_sync");
            Gpu::streamSynchronize();
        }

        BL_PROFILE("FB::post_sends");
        fbd->send_reqs.assign(N_snds, MPI_REQUEST_NULL);
        for (int j = 0; j < N_snds; ++j) {
            char* p = (e->snd.on_host[j] ? fbd->fbv_snd_h : fbd->fbv_snd_d) + e->snd.offset[j];
            if (e->snd.size[j] > 0) {
                const int rank = ParallelContext::global_to_local_rank(e->snd.rank[j]);
                fbd->send_reqs[j] = ParallelDescriptor::Asend
                    (p, e->snd.size[j], rank, SeqNum, comm).req();
            }
        }
    }

    if (N_rcvs > 0) { FillBoundary_test(); }

    if (N_locs > 0 && !local_done) {
        BL_PROFILE("FB::local_copy");
        FB_local_copy_gpu(TheFB, scomp, ncomp);
        if (N_rcvs > 0) { FillBoundary_test(); }
    }
}

template <class FAB>
template <typename BUF>
void
FabArray<FAB>::fbv_finish ()
{
    auto const& fbv = FabArrayBase::m_fbv;
    auto* e = fbd->fbv_entry;
    const auto N_rcvs = static_cast<int>(e->rcv.rank.size());
    const auto N_snds = static_cast<int>(e->snd.rank.size());

    if (N_rcvs > 0) {
        if (fbv.unpack_on_arrival) {
            // Messages completed by an earlier MPI_Testall have null
            // requests, so Waitsome never returns them.
            Vector<char> unpacked(N_rcvs, 0);
            Vector<int> indx(N_rcvs);
            auto unpack = [&] (int m) {
                BL_PROFILE("FB::unpack");
                detail::fbv_unpack<BUF>(e->recv_tv_msg[m], fbd->fbv_rcv_d,
                                        fbd->fbv_rcv_h, fbd->scomp, fbd->ncomp);
                unpacked[m] = 1;
            };
            for (;;) {
                int completed = 0;
                {
                    BL_PROFILE("FB::wait_recvs");
                    ParallelDescriptor::Waitsome(fbd->recv_reqs, completed, indx, fbd->recv_stat);
                }
                if (completed == MPI_UNDEFINED || completed == 0) { break; }
                for (int c = 0; c < completed; ++c) { unpack(indx[c]); }
            }
            for (int m = 0; m < N_rcvs; ++m) {
                if (!unpacked[m]) { unpack(m); }
            }
        } else {
            {
                BL_PROFILE("FB::wait_recvs");
                ParallelDescriptor::Waitall(fbd->recv_reqs, fbd->recv_stat);
            }
#ifdef AMREX_DEBUG
            if (!CheckRcvStats(fbd->recv_stat, fbd->recv_size, fbd->tag)) {
                amrex::Abort("FillBoundary_finish failed with wrong message size");
            }
#endif
            BL_PROFILE("FB::unpack");
            detail::fbv_unpack<BUF>(e->recv_tv, fbd->fbv_rcv_d, fbd->fbv_rcv_h,
                                    fbd->scomp, fbd->ncomp);
        }
    }

    // Without a host sync, the recv buffers are released in stream order.
    const bool sync = !fbv.async_unpack && !Gpu::inNoSyncRegion();
    if (sync) {
        BL_PROFILE("FB::unpack_sync");
        Gpu::streamSynchronize();
    }

    if (fbv.persistent) {
        if (!sync && N_rcvs > 0) {
            detail::fbv_event_record(e->unpack_done);
            e->unpack_pending = true;
        }
    } else {
        auto release = [&] (Arena* ar, char* p) {
            if (p) {
                if (sync) { ar->free(p); } else { Gpu::freeAsync(ar, p); }
            }
        };
        release(The_Comms_Arena(), fbd->fbv_rcv_d);
        release(The_Pinned_Arena(), fbd->fbv_rcv_h);
    }

    if (N_snds > 0) {
        BL_PROFILE("FB::wait_sends");
        Vector<MPI_Status> stats(fbd->send_reqs.size());
        ParallelDescriptor::Waitall(fbd->send_reqs, stats);
    }

    if (!fbv.persistent) {
        // The pack kernel finished before the sends were posted.
        if (fbd->fbv_snd_d) { The_Comms_Arena()->free(fbd->fbv_snd_d); }
        if (fbd->fbv_snd_h) { The_Pinned_Arena()->free(fbd->fbv_snd_h); }
    }

    fbd.reset();
}

}

#endif
#endif
