"""
동시 추정 모형: β 식을 SIR 안에 넣고 θ·S0·I0를 ILI 우도 하나로 함께 추정 (IF2, Ionides et al. 2015)
-----------------------------------------------------------------------------------------------
상태      S, I, η                     (하루 단위 7스텝 적분, γ = 1/3 고정)
전파      logβ_t = θ0 + θ_T·Tz_{t-L} + η_t,   η_t = φ·η_{t-1} + σ_η·ε_t      (AR(1), φ=0.9)
관측지연  ILI_t = κ·[w·c_t + (1-w)·c_{t-1}] + b_s + ε_t                     (감염→진료 지연)
관측오차  ε_t ~ N(0, a + (σ_b·flu_t)²),  연휴 주는 분산 ×9
절기(구간) s마다 S0_s, I0_s가 따로 있고, θ0·θ_T·σ_η·w·σ_b는 모든 구간이 공유한다.
기온 시차 L ∈ {0,1,2}는 L별로 최대우도를 비교해 고른다(프로파일 우도).
"""
import numpy as np, pandas as pd
from datetime import date
import core

KAPPA, GAMMA, PHI, OEV_A, IOTA, HOL = 1000.0, 1 / 3.0, 0.9, 0.3, 2e-6, 9.0
SHARED = ["theta0", "thetaT", "thetaS", "log_sig", "logit_w", "log_sigb"]
USE_SCHOOL = True   # 학교 수업 여부(방학·연휴=0) 이진 공변량을 β에 넣을지
RW_SD = dict(theta0=0.02, thetaT=0.02, thetaS=0.02, log_sig=0.02, logit_w=0.02, log_sigb=0.02)
IVP_SD = dict(S0=0.3, I0=0.5)
lgt = lambda p: np.log(p / (1 - p)); expit = lambda x: 1 / (1 + np.exp(-x))

# ---------------------------------------------------------------- 기온
def load_temp(path):
    """일별 서울 평균기온 → 질병청 주차 평균. Tz = (T - 15)/10 (고정 상수로 표준화: 정보누수 없음)."""
    rows = []
    for line in open(path, encoding="cp949"):
        p = line.strip().replace("\t", "").split(",")
        if len(p) >= 3 and p[0][:2] == "20" and p[2] not in ("", None):
            try:
                rows.append((date.fromisoformat(p[0]), float(p[2])))
            except ValueError:
                pass
    d = pd.DataFrame(rows, columns=["date", "T"])
    d["yw"] = d["date"].map(core.date_to_yw)
    w = d.groupby("yw").agg(T=("T", "mean"), n=("T", "size")).reset_index()
    w = w[w["n"] == 7]                       # 7일이 다 찬 주만
    w["Tz"] = (w["T"] - 15.0) / 10.0
    return w[["yw", "T", "Tz"]]

def load_ah(path):
    """일별 서울 절대습도(g/m³) → 질병청 주차 평균. 표준화 Tz = (AH - 10)/5 (고정 상수, 정보누수 없음).
    열 이름을 기온과 같게(Tz) 두어 나머지 코드를 그대로 쓴다."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d.rename(columns={d.columns[0]: "date", d.columns[-1]: "AH"})[["date", "AH"]].dropna()
    d["date"] = pd.to_datetime(d["date"]).dt.date
    d["yw"] = d["date"].map(core.date_to_yw)
    w = d.groupby("yw").agg(T=("AH", "mean"), n=("AH", "size")).reset_index()
    w = w[w["n"] == 7]
    w["Tz"] = (w["T"] - 10.0) / 5.0
    return w[["yw", "T", "Tz"]]

def temp_series(temp, weeks, origin, L):
    """weeks 각각의 (t-L) 주 기온. 기점 이후 주차는 기점까지 자료의 같은 주차 평균(기후값)."""
    hist = temp[temp["yw"] <= origin]
    known = dict(zip(hist["yw"], hist["Tz"]))
    clim = hist.groupby(hist["yw"] % 100)["Tz"].mean()
    out = []
    for w in weeks:
        lw = shift_week(w, -L)
        out.append(known[lw] if lw in known else clim.get(lw % 100, clim.mean()))
    return np.array(out)

def shift_week(yw, k):
    if k == 0:
        return yw
    if k > 0:
        return core.next_weeks(yw, k)[-1]
    y, w = divmod(yw, 100)
    from datetime import timedelta
    d = core.week1_start(y) + timedelta(days=7 * (w - 1) + 7 * k)
    return core.date_to_yw(d)

# ---------------------------------------------------------------- 구간(절기) 구성
def build_segments(df, window=12):
    """과거 절기(36주~35주) + 현재 구간(기점 이전 window주). 코로나 기간은 제외, 겹침 없음."""
    df = df[~df["covid"]].reset_index(drop=True)
    cur = df.tail(window)
    cur_start = int(cur["yw"].iloc[0])
    segs = []
    season = np.where(df["week"] >= 36, df["yw"] // 100, df["yw"] // 100 - 1)
    for s in sorted(set(season)):
        d = df[(season == s) & (df["yw"] < cur_start)]
        # 코로나 공백으로 끊긴 부분은 연속 구간만 남김
        if len(d) >= 15:
            segs.append(d.reset_index(drop=True))
    segs.append(cur.reset_index(drop=True))
    return segs

def seg_background(d):
    x = d[(d["pos"] < 3) & (~d["holiday"])]["ILI"]
    return float(x.median()) if len(x) >= 3 else float(d["ILI"].quantile(0.10))

def prep_segments(df, temp, origin, L, window=12, cur_bg=None):
    segs = []
    for d in build_segments(df, window):
        d = d.copy()
        d["Tz"] = temp_series(temp, d["yw"].tolist(), origin, L)
        sch = np.array([core.school_flag(w) for w in d["yw"]], float) if USE_SCHOOL else np.zeros(len(d))
        segs.append(dict(y=d["ILI"].values, hol=d["holiday"].values, Tz=d["Tz"].values, sch=sch,
                         yw=d["yw"].values, b=seg_background(d)))
    segs[-1]["b"] = cur_bg if cur_bg is not None else core.bg_level(df)
    return segs

# ---------------------------------------------------------------- 한 주 전이 + 우도
def week_step(S, I, eta, c_prev, par, Tz, rng, sch=0.0):
    sig = np.exp(par["log_sig"])
    eta = PHI * eta + sig * rng.standard_normal(len(S))
    beta = np.minimum(np.exp(par["theta0"] + par["thetaT"] * Tz + par["thetaS"] * sch + eta), 3.0)
    S, I, c = core.propagate_week(S, I, np.log(beta), GAMMA, IOTA)
    w = expit(par["logit_w"])
    flu = KAPPA * (w * c + (1 - w) * c_prev)
    return S, I, eta, c, flu

OUT_EPS = 0.02   # 이상치 혼합 비율: 단일 SIR로 설명 못 하는 주(다봉 절기 등)가 추정을 망치지 않게

def loglik_obs(y, flu, b, sigb, hol):
    """정규 관측오차 + 2% 폭넓은 이상치 성분의 혼합(강건 우도)."""
    mu = flu + b
    var = (OEV_A + (sigb * flu) ** 2) * (HOL if hol else 1.0)
    l1 = -0.5 * (np.log(2 * np.pi * var) + (y - mu) ** 2 / var)
    v2 = (5.0 + 0.5 * y) ** 2
    l2 = -0.5 * (np.log(2 * np.pi * v2) + (y - mu) ** 2 / v2)
    return np.logaddexp(np.log(1 - OUT_EPS) + l1, np.log(OUT_EPS) + l2)

def systematic(logw, rng):
    m = logw.max(); w = np.exp(logw - m); w /= w.sum()
    u = (rng.random() + np.arange(len(w))) / len(w)
    return np.minimum(np.searchsorted(np.cumsum(w), u), len(w) - 1), m + np.log(np.mean(np.exp(logw - m)))

def init_state(seg, S0, I0):
    n = len(S0)
    return S0.copy(), I0.copy(), np.zeros(n), I0 * 7 / (1 / GAMMA)

# ---------------------------------------------------------------- IF2
def if2(segs, J=1000, M=40, cool=0.93, seed=1, verbose=False):
    rng = np.random.default_rng(seed)
    nseg = len(segs)
    P = dict(theta0=rng.normal(np.log(0.45), 0.3, J), thetaT=rng.normal(0, 0.3, J),
             log_sig=rng.normal(np.log(0.15), 0.3, J), logit_w=rng.normal(lgt(0.6), 0.5, J),
             log_sigb=rng.normal(np.log(0.15), 0.3, J),
             thetaS=rng.normal(0.0, 0.3, J) if USE_SCHOOL else np.zeros(J))
    S0 = rng.uniform(lgt(0.3), lgt(0.95), (J, nseg))
    I0 = np.empty((J, nseg))
    for k, s in enumerate(segs):
        c0 = max(s["y"][:2].mean() - s["b"], 0.2) / KAPPA
        I0[:, k] = np.log(c0 * (1 / GAMMA) / 7) + rng.normal(0, 1.0, J)
    trace = []
    for m in range(M):
        sc = cool ** m
        ll_tot = 0.0
        for k, s in enumerate(segs):
            S0[:, k] += rng.normal(0, IVP_SD["S0"] * sc, J)
            I0[:, k] += rng.normal(0, IVP_SD["I0"] * sc, J)
            S, I, eta, cp = init_state(s, expit(S0[:, k]), np.minimum(np.exp(I0[:, k]), 0.05))
            for t in range(len(s["y"])):
                for p in SHARED:
                    P[p] = P[p] + rng.normal(0, RW_SD[p] * sc, J)
                S, I, eta, c, flu = week_step(S, I, eta, cp, P, s["Tz"][t], rng, s["sch"][t])
                lw = loglik_obs(s["y"][t], flu, s["b"], np.exp(P["log_sigb"]), s["hol"][t])
                idx, ll = systematic(lw, rng); ll_tot += ll
                S, I, eta, cp = S[idx], I[idx], eta[idx], c[idx]
                for p in SHARED:
                    P[p] = P[p][idx]
                S0, I0 = S0[idx], I0[idx]
        est = {p: float(P[p].mean()) for p in SHARED}
        trace.append(dict(iter=m, loglik=ll_tot, **est))
        if verbose and m % 5 == 0:
            print(m, round(ll_tot, 1), {k: round(v, 3) for k, v in est.items()}, flush=True)
    est = {p: float(P[p].mean()) for p in SHARED}
    est["S0"] = expit(S0).mean(0); est["I0"] = np.exp(I0).mean(0)
    est["swarm"] = dict(P=P, S0=expit(S0), I0=np.exp(I0))
    return est, pd.DataFrame(trace)

# ---------------------------------------------------------------- 최대우도점 입자필터
def pfilter(segs, est, J=3000, seed=2, record_last=True):
    """추정값을 고정하고 우도 계산 + 마지막 구간의 필터 상태와 β 궤적 반환.
    초기 S0·I0는 IF2 최종 입자군에서 뽑아 초기값 불확실성을 반영한다."""
    rng = np.random.default_rng(seed)
    sw = est["swarm"]; Jsw = len(sw["S0"])
    par = {p: np.full(J, est[p]) for p in SHARED}
    ll_tot, hist = 0.0, []
    for k, s in enumerate(segs):
        pick = rng.integers(0, Jsw, J)
        S, I, eta, cp = init_state(s, sw["S0"][pick, k], np.minimum(sw["I0"][pick, k], 0.05))
        for t in range(len(s["y"])):
            S, I, eta, c, flu = week_step(S, I, eta, cp, par, s["Tz"][t], rng, s["sch"][t])
            lw = loglik_obs(s["y"][t], flu, s["b"], np.exp(est["log_sigb"]), s["hol"][t])
            idx, ll = systematic(lw, rng); ll_tot += ll
            S, I, eta, cp = S[idx], I[idx], eta[idx], c[idx]
            if k == len(segs) - 1:
                lb = est["theta0"] + est["thetaT"] * s["Tz"][t] + est["thetaS"] * s["sch"][t] + eta
                hist.append(dict(yw=s["yw"][t], y=s["y"][t], tempT=est["thetaT"] * s["Tz"][t], schT=est["thetaS"] * s["sch"][t],
                                 eta_med=np.median(eta), logb_med=np.median(lb),
                                 logb_lo=np.quantile(lb, .1), logb_hi=np.quantile(lb, .9),
                                 Reff_med=np.median(np.exp(lb) * S / GAMMA),
                                 S_med=np.median(S), flu_med=np.median(KAPPA * c)))
    return ll_tot, dict(S=S, I=I, eta=eta, cp=cp), pd.DataFrame(hist)

# ---------------------------------------------------------------- 예측
def forecast(state, est, temp, origin, L, b, horizon=5, seed=3, bg_sd=0.08):
    rng = np.random.default_rng(seed)
    weeks = core.next_weeks(origin, horizon)
    Tz = temp_series(temp, weeks, origin, L)          # L≥h면 실측, 아니면 기후값
    S, I, eta, cp = (state[k].copy() for k in ("S", "I", "eta", "cp"))
    par = {p: np.full(len(S), est[p]) for p in SHARED}
    sigb = np.exp(est["log_sigb"]); lb = np.zeros(len(S)); sims = []
    for h in range(horizon):
        sch = core.school_flag(weeks[h]) if USE_SCHOOL else 0.0
        S, I, eta, c, flu = week_step(S, I, eta, cp, par, Tz[h], rng, sch)
        cp = c
        flu = np.maximum(flu + rng.standard_normal(len(S)) * np.sqrt(OEV_A + (sigb * flu) ** 2), 0)
        lb += rng.normal(0, bg_sd, len(S))
        sims.append(flu + b * np.exp(lb))
    sims = np.array(sims)
    q = np.maximum.accumulate(np.maximum(np.quantile(sims, core.QLEVELS, axis=1).T, 0), axis=1)
    out = pd.DataFrame(q, columns=[f"q{l}" for l in core.QLEVELS])
    out.insert(0, "horizon", range(1, horizon + 1)); out.insert(0, "target_week", weeks)
    out.insert(0, "origin", origin)
    return out, sims, Tz

# ---------------------------------------------------------------- 한 기점 전체
def run_origin(data, temp, origin, lags=(1,), J=1500, M=60, reps=3, window=12, extra=None, verbose=False, horizon=5):
    """reps개의 서로 다른 시작점으로 IF2를 돌리고 입자필터 우도가 가장 높은 것을 채택(다중 시작)."""
    df = core.load_data(data, origin=origin)          # 기점 이후 차단
    if extra is not None:                              # 2차 배포(ILI만) 붙이기
        df = append_extra(df, extra, origin)
    temp = temp[temp["yw"] <= origin]                  # 기온도 기점까지만
    b = core.bg_level(df)
    fits = {}
    for L in lags:
        segs = prep_segments(df, temp, origin, L, window, b)
        for r in range(reps):
            est, trace = if2(segs, J=J, M=M, seed=100 * L + r, verbose=verbose)
            ll, state, hist = pfilter(segs, est, seed=7)
            fits[(L, r)] = dict(est=est, trace=trace, ll=ll, state=state, hist=hist, segs=segs, L=L)
            if verbose:
                print(f"[{origin}] L={L} rep={r} loglik={ll:.1f} thetaT={est['thetaT']:.3f}", flush=True)
    key = max(fits, key=lambda k: fits[k]["ll"])
    f = fits[key]; Lbest = key[0]
    # 다중 시작 앙상블: 각 반복의 추정치로 예측한 표본을 모두 합쳐 분위수 → 모수 불확실성 일부 반영
    sims = []
    for k, fk in fits.items():
        _, s_k, Tz_f = forecast(fk["state"], fk["est"], temp, origin, fk["L"], b, horizon=horizon, seed=3 + k[1])
        sims.append(s_k)
    sims = np.concatenate(sims, axis=1)
    q = np.maximum.accumulate(np.maximum(np.quantile(sims, core.QLEVELS, axis=1).T, 0), axis=1)
    fc = pd.DataFrame(q, columns=[f"q{l}" for l in core.QLEVELS])
    fc.insert(0, "horizon", range(1, len(q) + 1)); fc.insert(0, "target_week", core.next_weeks(origin, len(q)))
    fc.insert(0, "origin", origin)
    return dict(origin=origin, L=Lbest, key=key, best=f, fits=fits, fc=fc, sims=sims, b=b, df=df, Tz_f=Tz_f)

def append_extra(df, extra, origin):
    """extra: DataFrame[년주, ILI]. 검출률이 없으므로 pos=NaN (모형은 ILI만 쓰므로 문제 없음)."""
    e = extra.rename(columns={"년주": "yw"})[["yw", "ILI"]].dropna()
    e["yw"] = e["yw"].astype(int)
    df = df.copy()
    upd = dict(zip(e["yw"], e["ILI"]))                     # 2차 배포에 기존 주(예: 34주 잠정치)의 수정값이 있으면 덮어씀
    m = df["yw"].isin(upd.keys())
    if m.any():
        old = df.loc[m, "ILI"].values.copy()
        df.loc[m, "ILI"] = df.loc[m, "yw"].map(upd).values
        print("  2차 배포로 수정된 기존 주:", {int(w): (float(a), float(b)) for w, a, b in zip(df.loc[m, "yw"], old, df.loc[m, "ILI"]) if a != b})
    e = e[(e["yw"] > df["yw"].max()) & (e["yw"] <= origin)].copy()
    e["pos"] = np.nan; e["plus"] = np.nan; e["bg"] = np.nan; e["ARI"] = np.nan
    e["holiday"] = e["yw"].isin(core.HOLIDAY); e["covid"] = False; e["week"] = e["yw"] % 100
    return pd.concat([df, e[df.columns]], ignore_index=True)
