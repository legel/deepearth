"""Dynamics and autoregressive probe modes. NONE of these can set a record.

Every mode here either measures a target that is not on the scorecard (occupancy, richness,
community-activity, abundance, first-arrival) or is quarantined for origin-crossing risk
(--ar_rollout, --ar_cond_lead). All of them evaluate on RAW positional features, so Earth4D is not in
the comparison and their numbers cannot speak to the encoder under any reading.

They live apart from the recording modes for exactly that reason: ~600 lines that an agent working a
scorecard capability never needs to read. Each declares itself a diagnostic, so trace.py refuses to
record them even if invoked under a legal --metric.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepearth.autoresearch.programs.spacetime.probe_emit import RAW_PE_REASON, declare


def ar_rollout_mode(ctx):
    """Selected by `a.ar_rollout`. Diagnostic: quarantined."""
    a, dev, t0 = ctx.a, ctx.dev, ctx.t0
    lat, lon, days, test = ctx.lat, ctx.lon, ctx.days, ctx.test
    load_species = ctx.load_species
    obs_index = ctx.obs_index
    assert a.forecast, "--ar_rollout requires --forecast"
    import numpy as _np
    from deepearth.autoresearch.programs.spacetime.dyntargets import (
        _windows, _assemble, _reg_skill, _community_activity_target, _abundance_target, _richness_target)
    lat_a = lat.astype(_np.float32); lon_a = lon.astype(_np.float32)
    rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
    raw_feat = torch.tensor(rn_sp)                                 # SPACE-ONLY query feature
    fdim = raw_feat.shape[1]; K = a.rec_k; H = a.rec_hidden; out_dim = 1
    win = a.abund_win                                              # overlap regime: win = step + 180 (set by caller)
    n_steps = max(1, int(round(a.ar_final / a.ar_step)))
    leads = [float(j * a.ar_step) for j in range(1, n_steps + 1)]  # intermediate + final leads
    assert abs(leads[-1] - a.ar_final) < 1e-6, f"ar_final {a.ar_final} not a multiple of ar_step {a.ar_step}"
    _sp_arr = load_species(a.cache_dir, a.n_shards)[obs_index] if a.ar_target == "richness" else None

    def _tgt_at(lead):
        if a.ar_target == "abundance":
            return _abundance_target(lat_a, lon_a, days, win=win, lead=lead, delta=False).astype(_np.float32)
        if a.ar_target == "richness":
            return _richness_target(lat_a, lon_a, days, _sp_arr, win=win, lead=lead)[0].astype(_np.float32)
        return _community_activity_target(lat_a, lon_a, days, win=win, lead=lead)[0].astype(_np.float32)

    # neighbour PAST state = its OWN target-lead activity (SAME convention as the settled breadth/direct
    # baseline, whose leak guard passes): each neighbour carries only its own observed quantity, never
    # anything relative to the query. This makes AR-vs-direct a fair matched comparison; the ONLY AR
    # addition is the fed-back query-state channel.
    past_state = _tgt_at(a.ar_final).reshape(-1, 1).astype(_np.float32)
    S = past_state.shape[1]

    _test = test
    tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
    _rng = _np.random.default_rng(0)
    q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
    g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
    # assemble the neighbour tensors ONCE (windows are lead-independent: causal past<=d); only the target
    # scalar changes per lead. Use a dummy target to grab the leak-safe (nfeat,nstate,edge,mask,len) tensors.
    dummy = _np.zeros(len(lat_a), _np.float32)
    tr0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, q_tr, g_tr, v_tr, dummy, K, out_dim)
    te0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, te_idx, g_te, v_te, dummy, K, out_dim)
    _to = lambda ts: [t.to(dev) for t in ts]
    nftr, nstr, qftr, etr, mtr, ltr, _, _ = _to(tr0)
    nfte, nste, qfte, ete, mte, lte, _, _ = _to(te0)
    # per-lead targets aligned to the SAME ok-masked rows _assemble kept (ok = window has >=1 valid nb)
    ok_tr = torch.tensor(v_tr).any(1).numpy(); ok_te = torch.tensor(v_te).any(1).numpy()
    Ytr = {ld: torch.tensor(_tgt_at(ld)[q_tr][ok_tr]).unsqueeze(-1).to(dev) for ld in leads}
    Yte = {ld: torch.tensor(_tgt_at(ld)[te_idx][ok_te]).unsqueeze(-1).to(dev) for ld in leads}
    Btr = int(nftr.shape[0]); n_te = int(nfte.shape[0])
    if Btr == 0 or n_te == 0:
        print("=== ar_rollout: EMPTY window set, abort ==="); return {"ar_rollout": True, "n_te": n_te}
    bs = min(2048, Btr)

    # single-step propagator g: (neighbour window, edge, query-prev-state ŷ) -> level at this step's lead.
    # The +1 input dim is the fed-back query state channel (0 at step 1, then the model's own prediction).
    class _ARStep(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
            s.head = nn.Sequential(nn.Linear(H + 1, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, nf, ns, edge, lengths, qprev):
            x = torch.cat([nf, ns, edge], -1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            _, (h, _) = s.lstm(packed)
            return s.head(torch.cat([h[-1], qprev], -1))
    # direct single-shot: identical capacity, no fed-back channel (qprev fixed 0), trained ONLY on final lead
    class _Direct(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
            s.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, nf, ns, edge, lengths):
            x = torch.cat([nf, ns, edge], -1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            _, (h, _) = s.lstm(packed)
            return s.head(h[-1])

    loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
    # ---- train the AR step model: shared weights over ALL steps; teacher-forced qprev = prev-lead target,
    #      plus scheduled feed-back of the model's own prediction (0.5 mix) so eval-time rollout matches.
    arm = _ARStep().to(dev); opt = torch.optim.Adam(arm.parameters(), lr=a.lr); arm.train()
    for it in range(a.steps):
        sidx = torch.randint(0, Btr, (bs,), device=dev)
        qprev = torch.zeros(bs, 1, device=dev)
        loss = 0.0
        for j, ld in enumerate(leads):
            pred = arm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx], qprev)
            loss = loss + loss_fn(pred, Ytr[ld][sidx])
            # feed BACK: mix model prediction with teacher (prev-lead truth) -> AR consistency w/o drift
            tf = Ytr[ld][sidx]
            qprev = (0.5 * pred.detach() + 0.5 * tf).detach()
        opt.zero_grad(); (loss / len(leads)).backward(); opt.step()
    arm.eval()
    # ---- ROLLOUT at eval: seed qprev=0, feed the model's OWN prediction forward. NO future obs consumed.
    roll_r2 = {}; qprev = torch.zeros(n_te, 1, device=dev)
    with torch.no_grad():
        for ld in leads:
            pred = arm(nfte, nste, ete, lte, qprev)
            roll_r2[ld] = _reg_skill(pred, Yte[ld], Yte[ld])
            qprev = pred                                          # pure AR: its own prediction only
    # ---- DIRECT single-shot at the final horizon (matched capacity, one prediction) ----
    dm = _Direct().to(dev); opt = torch.optim.Adam(dm.parameters(), lr=a.lr); dm.train()
    yfin_tr = Ytr[leads[-1]]
    for it in range(a.steps):
        sidx = torch.randint(0, Btr, (bs,), device=dev)
        loss = loss_fn(dm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx]), yfin_tr[sidx])
        opt.zero_grad(); loss.backward(); opt.step()
    dm.eval()
    with torch.no_grad():
        direct_fin = _reg_skill(dm(nfte, nste, ete, lte), Yte[leads[-1]], Yte[leads[-1]])

    # ---- STATIC FLOOR at final horizon (query space-only feat, no propagation) for dR2 context ----
    class _StaticH(nn.Module):
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(fdim, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, qf): return s.net(qf)
    sh = _StaticH().to(dev); opt = torch.optim.Adam(sh.parameters(), lr=a.lr); sh.train()
    for _ in range(a.steps):
        sidx = torch.randint(0, Btr, (bs,), device=dev)
        loss = loss_fn(sh(qftr[sidx]), yfin_tr[sidx]); opt.zero_grad(); loss.backward(); opt.step()
    sh.eval()
    with torch.no_grad(): static_fin = _reg_skill(sh(qfte), Yte[leads[-1]], Yte[leads[-1]])

    # ---- LEAK-GUARD: time-only static head at the FINAL horizon must NOT solve it ----
    tnorm = ((days - days.min()) / (days.max() - days.min() + 1e-9)).astype(_np.float32)
    tq_tr = torch.tensor(tnorm[q_tr][ok_tr]).unsqueeze(-1).to(dev)[: Btr]
    tq_te = torch.tensor(tnorm[te_idx][ok_te]).unsqueeze(-1).to(dev)[: n_te]
    yl_tr = Ytr[leads[-1]][: tq_tr.shape[0]]; yl_te = Yte[leads[-1]][: tq_te.shape[0]]
    class _TimeOnly(nn.Module):
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(1, H), nn.GELU(), nn.Linear(H, 1))
        def forward(s, x): return s.net(x)
    tom = _TimeOnly().to(dev); opt = torch.optim.Adam(tom.parameters(), lr=a.lr)
    for _ in range(a.steps):
        sidx = torch.randint(0, tq_tr.shape[0], (min(2048, tq_tr.shape[0]),), device=dev)
        loss = loss_fn(tom(tq_tr[sidx]), yl_tr[sidx]); opt.zero_grad(); loss.backward(); opt.step()
    tom.eval()
    with torch.no_grad(): leak_mae, leak_r2 = _reg_skill(tom(tq_te), yl_te, yl_te)

    print(f"=== SPACETIME AR-ROLLOUT | target={a.ar_target} | raw PE | obs={len(lat_a)} q={n_te} K={K} win={win:.0f}d step={a.ar_step:.0f}d final={a.ar_final:.0f}d n_steps={n_steps} Sdim={S} ===")
    for ld in leads:
        m, r2 = roll_r2[ld]
        tag = " <FINAL>" if abs(ld - leads[-1]) < 1e-6 else ""
        print(f"  rollout  lead {ld:6.0f}d | MAE {m:7.4f}  absR2 {r2:+.4f}{tag}")
    dmae, dr2 = direct_fin
    rmae, rr2 = roll_r2[leads[-1]]
    smae, sr2 = static_fin
    print(f"  DIRECT   lead {leads[-1]:6.0f}d | MAE {dmae:7.4f}  absR2 {dr2:+.4f}  (single-shot, matched horizon; dR2 vs static {dr2 - sr2:+.4f})")
    print(f"  STATIC   lead {leads[-1]:6.0f}d | MAE {smae:7.4f}  absR2 {sr2:+.4f}  (no-propagation floor)")
    print(f"  AR final dR2 vs static           | {rr2 - sr2:+.4f}")
    print(f"  AR - DIRECT (final absR2)        | {rr2 - dr2:+.4f}   (POSITIVE = rollout holds skill better)")
    print(f"  LEAK-GUARD time-only (final)     | MAE {leak_mae:7.4f}  absR2 {leak_r2:+.4f}  (must be ~0/neg = no time leak)")
    dt = time.time() - t0
    print(f"  [profile] q={n_te} K={K} hidden={H} steps={a.steps} n_steps={n_steps}")
    print(f"  {len(lat_a)} obs, {a.steps}-step AR-rollout in {dt:.1f}s")
    return {"ar_rollout": True, "target": a.ar_target, "final_lead": leads[-1], "step": a.ar_step,
            "n_steps": n_steps, "win": win, "K": K, "n_te": n_te,
            "rollout_absR2": {ld: roll_r2[ld][1] for ld in leads},
            "direct_final_absR2": dr2, "static_final_absR2": sr2,
            "ar_minus_direct": rr2 - dr2, "ar_dR2_vs_static": rr2 - sr2,
            "leak_absR2": leak_r2, "seconds": dt}

def ar_cond_lead_mode(ctx):
    """Selected by `a.ar_cond_lead`. Diagnostic: quarantined."""
    a, dev, t0 = ctx.a, ctx.dev, ctx.t0
    lat, lon, days, test = ctx.lat, ctx.lon, ctx.days, ctx.test
    load_species = ctx.load_species
    assert a.forecast, "--ar_cond_lead requires --forecast"
    import numpy as _np
    from deepearth.autoresearch.programs.spacetime.dyntargets import (
        _windows, _assemble, _reg_skill, _community_activity_target)
    lat_a = lat.astype(_np.float32); lon_a = lon.astype(_np.float32)
    rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
    raw_feat = torch.tensor(rn_sp); fdim = raw_feat.shape[1]
    K = a.rec_k; H = a.rec_hidden; out_dim = 1; win = a.abund_win
    n_steps = max(1, int(round(a.ar_final / a.ar_step)))
    leads = [float(j * a.ar_step) for j in range(1, n_steps + 1)]
    past_state = _community_activity_target(lat_a, lon_a, days, win=win, lead=0.0)[0].reshape(-1, 1).astype(_np.float32)
    S = past_state.shape[1]
    _test = test
    tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
    _rng = _np.random.default_rng(0)
    q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
    g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
    dummy = _np.zeros(len(lat_a), _np.float32)
    tr0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, q_tr, g_tr, v_tr, dummy, K, out_dim)
    te0 = _assemble(raw_feat, past_state, days, lat_a, lon_a, te_idx, g_te, v_te, dummy, K, out_dim)
    _to = lambda ts: [t.to(dev) for t in ts]
    nftr, nstr, qftr, etr, mtr, ltr, _, _ = _to(tr0)
    nfte, nste, qfte, ete, mte, lte, _, _ = _to(te0)
    ok_tr = torch.tensor(v_tr).any(1).numpy(); ok_te = torch.tensor(v_te).any(1).numpy()
    Ytr = {ld: torch.tensor(_community_activity_target(lat_a, lon_a, days, win=win, lead=ld)[0][q_tr][ok_tr]).unsqueeze(-1).to(dev) for ld in leads}
    Yte = {ld: torch.tensor(_community_activity_target(lat_a, lon_a, days, win=win, lead=ld)[0][te_idx][ok_te]).unsqueeze(-1).to(dev) for ld in leads}
    Btr = int(nftr.shape[0]); n_te = int(nfte.shape[0])
    if Btr == 0 or n_te == 0:
        print("=== ar_cond_lead: EMPTY window set, abort ==="); return {"ar_cond_lead": True, "n_te": n_te}
    bs = min(2048, Btr); lscale = float(max(leads))

    class _CondLead(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
            s.head = nn.Sequential(nn.Linear(H + 1, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, nf, ns, edge, lengths, lead_scalar):
            x = torch.cat([nf, ns, edge], -1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            _, (h, _) = s.lstm(packed)
            return s.head(torch.cat([h[-1], lead_scalar], -1))
    class _Direct(nn.Module):
        def __init__(s):
            super().__init__()
            s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=2, batch_first=True)
            s.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, nf, ns, edge, lengths):
            x = torch.cat([nf, ns, edge], -1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            _, (h, _) = s.lstm(packed)
            return s.head(h[-1])
    loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
    # one conditioned model over ALL leads (lead sampled each step)
    cm = _CondLead().to(dev); opt = torch.optim.Adam(cm.parameters(), lr=a.lr); cm.train()
    for it in range(a.steps):
        ld = leads[torch.randint(0, len(leads), (1,)).item()]
        sidx = torch.randint(0, Btr, (bs,), device=dev)
        ls = torch.full((bs, 1), ld / lscale, device=dev)
        loss = loss_fn(cm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx], ls), Ytr[ld][sidx])
        opt.zero_grad(); loss.backward(); opt.step()
    cm.eval()
    cond_r2 = {}
    with torch.no_grad():
        for ld in leads:
            ls = torch.full((n_te, 1), ld / lscale, device=dev)
            cond_r2[ld] = _reg_skill(cm(nfte, nste, ete, lte, ls), Yte[ld], Yte[ld])
    # per-lead DIRECT specialists
    spec_r2 = {}
    for ld in leads:
        dm = _Direct().to(dev); opt = torch.optim.Adam(dm.parameters(), lr=a.lr); dm.train()
        for it in range(a.steps):
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            loss = loss_fn(dm(nftr[sidx], nstr[sidx], etr[sidx], ltr[sidx]), Ytr[ld][sidx])
            opt.zero_grad(); loss.backward(); opt.step()
        dm.eval()
        with torch.no_grad(): spec_r2[ld] = _reg_skill(dm(nfte, nste, ete, lte), Yte[ld], Yte[ld])

    print(f"=== SPACETIME CONTINUOUS-LEAD conditioning | community-activity | raw PE | obs={len(lat_a)} q={n_te} K={K} win={win:.0f}d leads={[int(l) for l in leads]} ===")
    for ld in leads:
        cm_, cr2 = cond_r2[ld]; sm_, sr2 = spec_r2[ld]
        print(f"  lead {ld:6.0f}d | 1-model-cond absR2 {cr2:+.4f}  vs  specialist absR2 {sr2:+.4f}  (cond-spec {cr2 - sr2:+.4f})")
    mean_gap = float(_np.mean([cond_r2[ld][1] - spec_r2[ld][1] for ld in leads]))
    print(f"  mean (cond - specialist) absR2 over leads | {mean_gap:+.4f}   (>=0 = one model matches per-lead specialists)")
    dt = time.time() - t0
    print(f"  [profile] q={n_te} K={K} hidden={H} steps={a.steps} n_leads={len(leads)}")
    print(f"  {len(lat_a)} obs, {a.steps}-step cond-lead in {dt:.1f}s")
    return {"ar_cond_lead": True, "leads": leads, "win": win, "K": K, "n_te": n_te,
            "cond_absR2": {ld: cond_r2[ld][1] for ld in leads},
            "spec_absR2": {ld: spec_r2[ld][1] for ld in leads},
            "mean_cond_minus_spec": mean_gap, "seconds": dt}

def breadth_mode(ctx):
    """Selected by `a.breadth_target`. Diagnostic: target is not a scorecard capability."""
    a, dev, t0 = ctx.a, ctx.dev, ctx.t0
    lat, lon, days, test = ctx.lat, ctx.lon, ctx.days, ctx.test
    load_species = ctx.load_species
    obs_index = ctx.obs_index
    assert a.forecast, "--breadth_target requires --forecast"
    import numpy as _np
    from deepearth.autoresearch.programs.spacetime.dyntargets import (
        _windows, _assemble, _reg_skill, _occupancy_target, _richness_target, _community_activity_target)
    coords_ll = torch.tensor(_np.stack([lat, lon], 1).astype(_np.float32))
    lat_a = coords_ll[:, 0].numpy(); lon_a = coords_ll[:, 1].numpy()
    rn_sp = _np.stack([lat / 90.0, lon / 180.0], 1).astype(_np.float32)
    raw_feat = torch.tensor(rn_sp)                                 # SPACE-ONLY query feature (t stripped)
    fdim = raw_feat.shape[1]
    sp_arr = load_species(a.cache_dir, a.n_shards)[obs_index]
    win, lead = a.abund_win, a.abund_lead
    if a.breadth_target == "occupancy":
        tgt, past = _occupancy_target(lat_a, lon_a, days, sp_arr, win=win, lead=lead, sub=a.breadth_sub)
        _tn = "OCCUPANCY-LEVEL(detect-frac)"
    elif a.breadth_target == "richness":
        tgt, past = _richness_target(lat_a, lon_a, days, sp_arr, win=win, lead=lead)
        _tn = "COMMUNITY-RICHNESS-LEVEL(log-nspp)"
    else:
        tgt, past = _community_activity_target(lat_a, lon_a, days, win=win, lead=lead)
        _tn = "COMMUNITY-ACTIVITY-LEVEL(log-count-all)"
    tgt = tgt.astype(_np.float32)
    nstate = past.astype(_np.float32) if past is not None else tgt.reshape(-1, 1).astype(_np.float32)
    S = nstate.shape[1]; K = a.rec_k; H = a.rec_hidden; out_dim = 1

    _test = test
    tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
    _rng = _np.random.default_rng(0)
    q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
    g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
    tr = _assemble(raw_feat, nstate, days, lat_a, lon_a, q_tr, g_tr, v_tr, tgt, K, out_dim)
    te = _assemble(raw_feat, nstate, days, lat_a, lon_a, te_idx, g_te, v_te, tgt, K, out_dim)
    _to = lambda ts: [t.to(dev) for t in ts]
    nftr, nstr, qftr, etr, mtr, ltr, ytr, _ = _to(tr)
    nfte, nste, qfte, ete, mte, lte, yte, _ = _to(te)
    n_te = int(nfte.shape[0]); Btr = int(nftr.shape[0])
    if Btr == 0 or n_te == 0:
        print("=== breadth_target: EMPTY window set, abort ==="); return {"breadth_target": a.breadth_target, "n_te": n_te}
    bs = min(2048, Btr)

    class _StaticH(nn.Module):
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(fdim, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, qf): return s.net(qf)

    class _DeepLSTM(nn.Module):
        def __init__(s, layers):
            super().__init__()
            s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=layers, batch_first=True)
            s.head = nn.Linear(H, out_dim)
        def forward(s, nf, ns, edge, lengths):
            x = torch.cat([nf, ns, edge], -1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            _, (h, _) = s.lstm(packed)
            return s.head(h[-1])

    loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
    def _train(model, fwd):
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        model.train()
        for _ in range(a.steps):
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            loss = loss_fn(fwd(model, sidx), ytr[sidx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

    results = {}
    sh = _StaticH().to(dev); _train(sh, lambda m, s: m(qftr[s]))
    with torch.no_grad(): results["static"] = _reg_skill(sh(qfte), yte, yte)
    lstm = _DeepLSTM(2).to(dev)
    _train(lstm, lambda m, s: m(nftr[s], nstr[s], etr[s], ltr[s]))
    with torch.no_grad(): results["lstm2"] = _reg_skill(lstm(nfte, nste, ete, lte), yte, yte)

    # LEAK-GUARD SMOKE TEST: a static head reading ONLY the query's own normalized time coord must NOT
    # solve it. Low absR2 => the propagator result is not a time-arithmetic leak. Uses the SAME ok-masked
    # test rows so it is directly comparable to the propagator absR2 above.
    tnorm = ((days - days.min()) / (days.max() - days.min() + 1e-9)).astype(_np.float32)
    okmask_te = torch.tensor(v_te).any(1).numpy()
    tq_te = torch.tensor(tnorm[te_idx][okmask_te]).unsqueeze(-1).to(dev)
    okmask_tr = torch.tensor(v_tr).any(1).numpy()
    tq_tr = torch.tensor(tnorm[q_tr][okmask_tr]).unsqueeze(-1).to(dev)
    n_lk = min(tq_tr.shape[0], ytr.shape[0]); tq_tr = tq_tr[:n_lk]; yl_tr = ytr[:n_lk]
    n_lke = min(tq_te.shape[0], yte.shape[0]); tq_te = tq_te[:n_lke]; yl_te = yte[:n_lke]
    class _TimeOnly(nn.Module):
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(1, H), nn.GELU(), nn.Linear(H, 1))
        def forward(s, x): return s.net(x)
    to_m = _TimeOnly().to(dev); opt = torch.optim.Adam(to_m.parameters(), lr=a.lr)
    for _ in range(a.steps):
        sidx = torch.randint(0, tq_tr.shape[0], (min(2048, tq_tr.shape[0]),), device=dev)
        loss = loss_fn(to_m(tq_tr[sidx]), yl_tr[sidx]); opt.zero_grad(); loss.backward(); opt.step()
    to_m.eval()
    with torch.no_grad(): leak_mae, leak_r2 = _reg_skill(to_m(tq_te), yl_te, yl_te)

    s_mae, s_r2 = results["static"]; l_mae, l_r2 = results["lstm2"]
    print(f"  static-floor        | MAE {s_mae:7.4f}  absR2 {s_r2:+.4f}")
    print(f"  deepLSTM-2L         | MAE {l_mae:7.4f}  absR2 {l_r2:+.4f}  (dR2 vs static {l_r2 - s_r2:+.4f})")
    print(f"  LEAK-GUARD time-only| MAE {leak_mae:7.4f}  absR2 {leak_r2:+.4f}  (must be ~0/negative = no time leak)")
    dt = time.time() - t0
    print(f"  {len(lat_a)} obs, {a.steps}-step breadth in {dt:.1f}s")
    declare(
        capability="", mode=f"BREADTH({_tn})", metric="absR2", value=s_r2,
        diagnostic=True,
        diagnostic_reason=f"{_tn} is not a scorecard capability; " + RAW_PE_REASON,
        obs=len(lat_a), queries=n_te, K=K, win=win, lead=lead, seconds=dt,
        static_mae=s_mae, lstm_absR2=l_r2, leak_guard_absR2=leak_r2,
    )
    return {"breadth_target": a.breadth_target, "target": _tn, "static_absR2": s_r2,
            "lstm2_absR2": l_r2, "leak_absR2": leak_r2, "win": win, "lead": lead, "K": K,
            "n_te": n_te, "seconds": dt}

def prop_arch_mode(ctx):
    """Selected by `a.abund_prop_arch`. Diagnostic: target is not a scorecard capability."""
    a, dev, t0 = ctx.a, ctx.dev, ctx.t0
    lat, lon, days, test = ctx.lat, ctx.lon, ctx.days, ctx.test
    load_species = ctx.load_species
    import numpy as _np0
    _rn = _np0.stack([lat / 90.0, lon / 180.0], 1).astype(_np0.float32)
    raw_sp = torch.tensor(_rn)
    coords_ll = torch.tensor(_np0.stack([lat, lon], 1).astype(_np0.float32))
    # LOOP-spacetime propagator-ARCHITECTURE probe on the LEVEL abundance target (settled forecastable:
    # LSTM abs R2 up to +0.76). ONE structural change: swap the causal propagator head. Reuse dyntargets'
    # leak-guarded window builder + target; define deeper/attention heads LOCALLY (additive, probe-only).
    assert a.forecast, "--abund_prop_arch requires --forecast"
    import numpy as _np
    from deepearth.autoresearch.programs.spacetime.dyntargets import (
        _abundance_target, _windows, _assemble, _reg_skill, doy_of, doy_to_vec)
    lat_a = coords_ll[:, 0].numpy(); lon_a = coords_ll[:, 1].numpy()
    raw_feat = raw_sp                                              # SPACE-ONLY query feature (t stripped)
    fdim = raw_feat.shape[1]
    tgt = _abundance_target(lat_a, lon_a, days, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta)
    # neighbour PAST state: abundance-only, or joint multivariate [abund || DOY sin,cos || occupancy bit]
    if a.abund_multivar:
        doyv = doy_to_vec(doy_of(days))                           # [N,2] each neighbour's own past DOY phase
        occ = (tgt > 0).astype(_np.float32).reshape(-1, 1)        # past occupancy (was cell active)
        nstate = _np.concatenate([tgt.reshape(-1, 1), doyv, occ], 1).astype(_np.float32)
    else:
        nstate = tgt.reshape(-1, 1).astype(_np.float32)
    S = nstate.shape[1]; K = a.rec_k; H = a.rec_hidden; out_dim = 1

    # build the SAME leak-guarded train/test window tensors used by dyntargets._fit_eval
    _test = test
    tr_idx = _np.where(~_test)[0]; te_idx = _np.where(_test)[0]
    _rng = _np.random.default_rng(0)
    q_tr = tr_idx if len(tr_idx) <= 6000 else _rng.choice(tr_idx, 6000, replace=False)
    g_tr, v_tr = _windows(lat_a, lon_a, days, q_tr, tr_idx, K)
    g_te, v_te = _windows(lat_a, lon_a, days, te_idx, tr_idx, K)
    tr = _assemble(raw_feat, nstate, days, lat_a, lon_a, q_tr, g_tr, v_tr, tgt, K, out_dim)
    te = _assemble(raw_feat, nstate, days, lat_a, lon_a, te_idx, g_te, v_te, tgt, K, out_dim)
    _to = lambda ts: [t.to(dev) for t in ts]
    nftr, nstr, qftr, etr, mtr, ltr, ytr, _ = _to(tr)
    nfte, nste, qfte, ete, mte, lte, yte, _ = _to(te)
    n_te = int(nfte.shape[0]); Btr = int(nftr.shape[0])
    if Btr == 0 or n_te == 0:
        print("=== abund_prop_arch: EMPTY window set, abort ==="); return {"abund_prop_arch": True, "n_te": n_te}
    bs = min(2048, Btr)

    class _StaticH(nn.Module):
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(fdim, H), nn.GELU(), nn.Linear(H, out_dim))
        def forward(s, qf): return s.net(qf)

    class _DeepLSTM(nn.Module):
        def __init__(s, layers):
            super().__init__()
            s.lstm = nn.LSTM(fdim + S + 2, H, num_layers=layers, batch_first=True,
                             dropout=0.0)
            s.head = nn.Linear(H, out_dim)
        def forward(s, nf, ns, edge, lengths):
            x = torch.cat([nf, ns, edge], -1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1),
                                                       batch_first=True, enforce_sorted=False)
            _, (h, _) = s.lstm(packed)
            return s.head(h[-1])

    class _AttnH(nn.Module):
        # attention-over-neighbour-history: query token attends over K past-neighbour tokens
        def __init__(s, heads, layers):
            super().__init__()
            s.tok = nn.Linear(fdim + S + 2, H)
            s.q = nn.Linear(fdim, H)
            enc = nn.TransformerEncoderLayer(H, heads, H * 2, batch_first=True, activation="gelu")
            s.tr = nn.TransformerEncoder(enc, layers)
            s.head = nn.Linear(H, out_dim)
        def forward(s, nf, ns, edge, mask):
            x = s.tok(torch.cat([nf, ns, edge], -1))              # [B,K,H]
            pad = ~mask.bool()                                    # True where padded
            x = s.tr(x, src_key_padding_mask=pad)
            x = x.masked_fill(pad.unsqueeze(-1), 0.0)
            pooled = x.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            return s.head(pooled)

    loss_fn = lambda p, y: F.smooth_l1_loss(p, y)
    def _train(model, fwd):
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
        model.train()
        for _ in range(a.steps):
            sidx = torch.randint(0, Btr, (bs,), device=dev)
            loss = loss_fn(fwd(model, sidx), ytr[sidx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

    want = set(x for x in a.prop_arch.split(",") if x)
    results = {}
    # static floor (shared)
    sh = _StaticH().to(dev); _train(sh, lambda m, s: m(qftr[s]))
    with torch.no_grad():
        results["static"] = _reg_skill(sh(qfte), yte, yte)
    arch_defs = {
        "lstm1": ("deepLSTM-1L", lambda: _DeepLSTM(1), "seq"),
        "lstm2": ("deepLSTM-2L", lambda: _DeepLSTM(2), "seq"),
        "lstm3": ("deepLSTM-3L", lambda: _DeepLSTM(3), "seq"),
        "lstm4": ("deepLSTM-4L", lambda: _DeepLSTM(4), "seq"),
        "attn":  ("attn-hist",   lambda: _AttnH(a.prop_attn_heads, a.prop_attn_layers), "attn"),
        "mv":    ("deepLSTM-2L", lambda: _DeepLSTM(2), "seq"),
    }
    for key in ("lstm1", "lstm2", "lstm3", "lstm4", "attn", "mv"):
        if key not in want: continue
        nm, ctor, kind = arch_defs[key]
        model = ctor().to(dev)
        if kind == "seq":
            _train(model, lambda m, s: m(nftr[s], nstr[s], etr[s], ltr[s]))
            with torch.no_grad():
                results[key] = _reg_skill(model(nfte, nste, ete, lte), yte, yte)
        else:
            _train(model, lambda m, s: m(nftr[s], nstr[s], etr[s], mtr[s]))
            with torch.no_grad():
                results[key] = _reg_skill(model(nfte, nste, ete, mte), yte, yte)

    _tgtn = "ABUND-DELTA(dlog)" if a.abund_delta else "ABUND-LEVEL(log-count)"
    _mv = " MULTIVAR-nstate[abund|doy|occ]" if a.abund_multivar else ""
    s_mae, s_r2 = results["static"]
    print(f"  static-floor        | MAE {s_mae:7.3f}  absR2 {s_r2:+.4f}")
    for key in ("lstm1", "lstm2", "lstm3", "lstm4", "attn", "mv"):
        if key in results and key != "static":
            nm = arch_defs[key][0]
            mae, r2 = results[key]
            print(f"  {nm:<18}| MAE {mae:7.3f}  absR2 {r2:+.4f}  (dR2 vs static {r2 - s_r2:+.4f})")
    dt = time.time() - t0
    print(f"  {len(lat_a)} obs, {a.steps}-step prop-arch in {dt:.1f}s")
    declare(
        capability="", mode=f"PROPAGATOR-ARCH({_tgtn})", metric="absR2", value=s_r2,
        diagnostic=True,
        diagnostic_reason="propagator-architecture comparison; " + RAW_PE_REASON,
        obs=len(lat_a), queries=n_te, K=K, hidden=H, seconds=dt,
        static_mae=s_mae, prop_arch=a.prop_arch,
    )
    return {"abund_prop_arch": True, "target": _tgtn, "static_absR2": s_r2,
            "results": {k: {"mae": v[0], "absR2": v[1]} for k, v in results.items()},
            "abund_lead": a.abund_lead, "abund_win": a.abund_win, "abund_delta": a.abund_delta,
            "multivar": a.abund_multivar, "K": K, "n_te": n_te, "seconds": dt}

def arrival_abundance_mode(ctx):
    """Selected by `a.first_arrival or a.abundance`. Diagnostic: target is not a scorecard capability."""
    a, dev, t0 = ctx.a, ctx.dev, ctx.t0
    lat, lon, days, test = ctx.lat, ctx.lon, ctx.days, ctx.test
    load_species = ctx.load_species
    obs_index = ctx.obs_index
    enc, e4d = ctx.enc, ctx.e4d
    assert a.forecast, "--first_arrival/--abundance require --forecast (needs live event-time + past->future split)"
    rn_sp = np.stack([lat / 90.0, lon / 180.0], 1).astype(np.float32)
    raw_sp = torch.tensor(rn_sp)
    _rng = np.random.default_rng(0)
    _proj = rn_sp @ (_rng.normal(0, 8.0, (2, e4d.shape[1] // 2)).astype(np.float32))
    rff_sp = torch.tensor(np.concatenate([np.sin(_proj), np.cos(_proj)], 1).astype(np.float32))
    coords_sp = torch.tensor(np.stack([lat, lon, np.zeros_like(lat), np.zeros_like(lat)], 1).astype(np.float32))  # t=0: no time leak
    with torch.no_grad():
        e4d_sp = enc(coords_sp.to(dev)).cpu()
    fd = {"e4d": e4d_sp.shape[1], "rff": rff_sp.shape[1], "raw": raw_sp.shape[1]}
    coords_ll = torch.tensor(np.stack([lat, lon], 1).astype(np.float32))

    def _report(name, unit, r, tol_line):
        n_te = r["raw"]["n_te"]
        def pg(ft, prop):
            return r[ft]["static_mae"] - r[ft][prop + "_mae"]
        g_raw_gnn = pg("raw", "gnn"); g_raw_lstm = pg("raw", "lstm")
        g_e4d_gnn = pg("e4d", "gnn"); g_rff_gnn = pg("rff", "gnn")
        best = max(g_raw_gnn, g_raw_lstm)
        for ft in ("raw", "rff", "e4d"):
            d = r[ft]
            print(f"  {ft:>4} | static {unit} {d['static_mae']:7.3f} acc/R2 {d['static_acc']:+.4f} -> GNN {unit} {d['gnn_mae']:7.3f} acc/R2 {d['gnn_acc']:+.4f} (prop {d['static_mae']-d['gnn_mae']:+.3f}) | LSTM {unit} {d['lstm_mae']:7.3f} acc/R2 {d['lstm_acc']:+.4f} (prop {d['static_mae']-d['lstm_mae']:+.3f})")
        print(f"  BEST propagator_gain (raw features, {unit} reduction; POSITIVE=propagation helps) GNN {g_raw_gnn:+.3f}  LSTM {g_raw_lstm:+.3f}  best {best:+.3f}")
        print(f"  ENCODER control (GNN {unit} reduction vs static, per PE): raw {g_raw_gnn:+.3f} | RFF {g_rff_gnn:+.3f} | Earth4D {g_e4d_gnn:+.3f}  (Earth4D-vs-raw GNN {unit} {r['raw']['gnn_mae']-r['e4d']['gnn_mae']:+.3f}: +=E4D better)")
        return {"target": name, "static_mae_raw": r["raw"]["static_mae"], "gnn_mae_raw": r["raw"]["gnn_mae"],
                "lstm_mae_raw": r["raw"]["lstm_mae"], "propagator_gain_mae": best,
                "propagator_gain_gnn_mae": g_raw_gnn, "propagator_gain_lstm_mae": g_raw_lstm,
                "propagator_gain_e4d_mae": g_e4d_gnn, "propagator_gain_rff_mae": g_rff_gnn,
                "static_acc_raw": r["raw"]["static_acc"], "gnn_acc_raw": r["raw"]["gnn_acc"],
                "lstm_acc_raw": r["raw"]["lstm_acc"], "obs": len(lat), "n_te": n_te}

    if a.first_arrival:
        import glob as _glob
        from pathlib import Path as _Path
        _sp = []
        for _f in sorted(_glob.glob(str(_Path(a.cache_dir) / "gbif_tokens/*.npz")))[:a.n_shards]:
            _sp.append(np.load(_f)["species_local"])
        sp_all = np.concatenate(_sp).astype(np.int64)[obs_index]
        from deepearth.autoresearch.programs.spacetime.dyntargets import run_first_arrival_all
        r = run_first_arrival_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, sp_all, test, dev,
                                  feats=tuple(x for x in a.pheno_feats.split(",") if x),
                                  K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, tol_days=a.pheno_tol)
        res = _report("FIRST_ARRIVAL(onset-DOY)", "MAEd", r, f"tol=+/-{a.pheno_tol:.0f}d")
        dt = time.time() - t0
        print(f"  {len(lat)} obs, {a.steps}-step first-arrival in {dt:.1f}s")
        declare(
            capability="", mode="FIRST-ARRIVAL(onset-DOY)", metric="MAE",
            value=res.get("static_mae_raw", float("nan")),
            diagnostic=True,
            diagnostic_reason="first-arrival is not a scorecard capability; " + RAW_PE_REASON,
            obs=len(lat), seconds=dt, win=a.abund_win, lead=a.abund_lead,
        )
        return res | {"seconds": dt, "first_arrival": True}

    if a.abundance:
        from deepearth.autoresearch.programs.spacetime.dyntargets import run_abundance_all
        r = run_abundance_all(e4d_sp, rff_sp, raw_sp, fd, days, coords_ll, test, dev,
                              feats=tuple(x for x in a.pheno_feats.split(",") if x),
                              K=a.rec_k, steps=a.steps, lr=a.lr, hidden=a.rec_hidden, hops=a.gnn_hops, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta)
        _nm = "ABUNDANCE-DELTA(dlog)" if a.abund_delta else "ABUNDANCE(log-count)"
        res = _report(_nm, "MAE", r, f"win={a.abund_win:.0f}d lead={a.abund_lead:.0f}d delta={a.abund_delta}")
        res = res | {"abund_lead": a.abund_lead, "abund_win": a.abund_win, "abund_delta": a.abund_delta}
        dt = time.time() - t0
        print(f"  {len(lat)} obs, {a.steps}-step abundance in {dt:.1f}s")
        declare(
            capability="", mode=_nm,   # _nm already reads "ABUNDANCE(...)"
            metric="MAE",
            value=res.get("static_mae_raw", float("nan")),
            diagnostic=True,
            diagnostic_reason="abundance is not a scorecard capability; " + RAW_PE_REASON,
            obs=len(lat), seconds=dt, win=a.abund_win, lead=a.abund_lead, delta=a.abund_delta,
        )
        return res | {"seconds": dt, "abundance": True}
