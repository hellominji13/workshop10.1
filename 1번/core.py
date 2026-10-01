"""
잠재 SIR 상태공간 모형 (단일 인플루엔자, K=1) + EAKF 추정 + 5주 분위수 예측
------------------------------------------------------------------------
상태      x_t = [S, I, logβ]           (인구 비율, 하루 단위 7스텝 적분)
전파      logβ_t = α_t + zᵀθ           α_t: 랜덤워크(설명 안 된 부분 η), z: 공변량(예: 절대습도)
관측(독감) ILI+_t = κ · c_t + ε_t        c_t: 주간 신규감염 비율, ε ~ N(0, OEV_t)
관측(배경) ILI_t  = ILI+_t + b_t         b_t = ILI×(1-검출률): 비인플루엔자 의사환자
γ는 고정(1/D, D=3일). 기점 이후 자료는 load 단계에서 잘라 절대 쓰지 않는다.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta

QLEVELS = np.array([0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975])

# ---------------------------------------------------------------- 주차·연휴
SEOLLAL = ["2016-02-08","2017-01-28","2018-02-16","2019-02-05","2020-01-25","2021-02-12",
           "2022-02-01","2023-01-22","2024-02-10","2025-01-29","2026-02-17","2027-02-07"]
CHUSEOK = ["2016-09-15","2017-10-04","2018-09-24","2019-09-13","2020-10-01","2021-09-21",
           "2022-09-10","2023-09-29","2024-09-17","2025-10-06","2026-09-25","2027-09-15"]

def week1_start(y):
    """질병청 주차 1주의 일요일. 자료(2016-53, 2022-53 존재)와 문제지(2027-01=1.3) 모두에 맞춤."""
    if y == 2027:
        return date(2027, 1, 3)
    j1 = date(y, 1, 1)
    return j1 - timedelta(days=(j1.weekday() + 1) % 7)   # 1월 1일을 포함하는 주의 일요일

def date_to_yw(d):
    y = d.year + 1
    while week1_start(y) > d:
        y -= 1
    return y * 100 + (d - week1_start(y)).days // 7 + 1

def holiday_weeks():
    out = set()
    for s in SEOLLAL + CHUSEOK:
        d = date.fromisoformat(s)
        for k in (-1, 0, 1):
            out.add(date_to_yw(d + timedelta(days=k)))
    return out

HOLIDAY = holiday_weeks()

def week_start(yw):
    y, w = divmod(int(yw), 100)
    return week1_start(y) + timedelta(days=7 * (w - 1))

def in_vacation(d):
    """근사 학사일정(초·중·고, 전국 평균): 여름방학 7/22~8/17, 겨울(학년말)방학 12/24~2/말.
    (2019년 이전 2월 등교 기간은 짧아 방학으로 묶음 — README에 한계로 기록)"""
    md = (d.month, d.day)
    return ((7, 22) <= md <= (8, 17)) or md >= (12, 24) or md <= (2, 29)

def school_flag(yw):
    """1 = 수업 주, 0 = 방학(평일 5일 중 3일 이상 방학) 또는 설·추석 연휴 주. 달력은 미리 알려진 정보라 미래에도 사용 가능."""
    if int(yw) in HOLIDAY:
        return 0
    s = week_start(yw)
    days = [s + timedelta(days=k) for k in range(1, 6)]   # 월~금
    return 0 if sum(in_vacation(d) for d in days) >= 3 else 1

def covid_period(yw):
    return (yw >= 202010) & (yw <= 202220)

# ---------------------------------------------------------------- 자료
def load_data(path, origin=None, ah=None):
    """origin(YYYYWW)까지만 반환 → 기점 이후 정보 차단. ah: [년주, AH] DataFrame(선택)."""
    df = pd.read_excel(path)
    df = df.rename(columns={"년주": "yw", "전체 검출률(%)": "pos"})
    df = df[["yw", "ARI", "ILI", "pos"]].copy()
    df["yw"] = df["yw"].astype(int)
    if origin is not None:
        df = df[df["yw"] <= origin].reset_index(drop=True)
    df["plus"] = df["ILI"] * df["pos"] / 100.0            # ILI+
    df["bg"] = df["ILI"] - df["plus"]                     # 비인플루엔자 배경
    df["holiday"] = df["yw"].isin(HOLIDAY)
    df["covid"] = covid_period(df["yw"])
    df["week"] = df["yw"] % 100
    if ah is not None:
        df = df.merge(ah.rename(columns={"년주": "yw"})[["yw", "AH"]], on="yw", how="left")
    return df

def next_weeks(yw, n):
    """yw 다음 n개 주차 (연말 52/53주 처리)."""
    out, cur = [], yw
    for _ in range(n):
        y, w = divmod(cur, 100)
        nxt_start = week1_start(y) + timedelta(days=7 * w)
        cur = date_to_yw(nxt_start)
        out.append(cur)
    return out

# ---------------------------------------------------------------- 모형 설정
DEFAULT = dict(
    D=3.0,            # 감염 가능 기간(일). γ = 1/D  (Carrat 2008 등 문헌 2~4일)
    kappa=1000.0,     # ILI+ 척도: ILI+ = κ × 주간 신규감염 비율
    iota=2e-6,        # 외부 유입 (소멸 방지)
    beta_max=3.0,     # β 상한 (일 단위). R0 상한 = beta_max·D
    window=10,        # 기점 이전 몇 주부터 필터를 돌릴지 (튜닝)
    n_ens=400,
    sig_beta=0.15,    # logβ 주간 랜덤워크 표준편차 (과정오차 η, 튜닝)
    oev_a=0.3,        # 관측오차 분산 = oev_a + (oev_b·y)^2
    oev_b=0.10,       # (튜닝)
    hol_infl=9.0,     # 연휴 주 관측오차 분산 배수
    infl=1.03,        # 앙상블 공분산 팽창
    S0=(0.35, 0.95),  # 초기 감수성 비율 사전범위
    bg_sd=0.08,       # 배경 ILI 주간 로그 변화 표준편차
    use_plus=False,   # ILI+를 두 번째 관측으로 쓸지 (튜닝 결과 미사용이 약간 우세)
    rho_plus=0.25,    # ILI+ ≈ 0.25 × (ILI - 기저)  (과거 자료 중앙값)
    plus_cv=0.5,      # 그 비율의 변동(변동계수)
    theta_ah=None,    # 절대습도 계수 (None이면 AH 미사용)
    seed=20260930,
)

# ---------------------------------------------------------------- SIR 전이
def propagate_week(S, I, logb, gamma, iota, ah_term=0.0):
    beta = np.exp(logb + ah_term)
    c = np.zeros_like(S)
    g = 1.0 - np.exp(-gamma)
    for _ in range(7):                       # 하루 단위 7스텝
        lam = 1.0 - np.exp(-beta * (I + iota))
        new = lam * S
        S = S - new
        I = I + new - g * I
        c = c + new
    return S, I, c

def eakf_update(X, ypred, yobs, oev, infl):
    """EAKF 한 번: X (n,p) 상태, ypred (n,) 예측관측."""
    m = ypred.mean()
    X = X.mean(0) + infl * (X - X.mean(0))
    ypred = m + infl * (ypred - m)
    v = ypred.var()
    if v < 1e-12:
        return X
    vpost = 1.0 / (1.0 / v + 1.0 / oev)
    mpost = vpost * (m / v + yobs / oev)
    dy = mpost + np.sqrt(vpost / v) * (ypred - m) - ypred
    for k in range(X.shape[1]):
        cov = np.cov(X[:, k], ypred)[0, 1]
        X[:, k] = X[:, k] + cov / v * dy
    return X

def oev_of(y, cfg, holiday):
    v = cfg["oev_a"] + (cfg["oev_b"] * max(y, 0.0)) ** 2
    return v * (cfg["hol_infl"] if holiday else 1.0)

# ---------------------------------------------------------------- 초기값
def init_ensemble(df, t0, cfg, rng, b):
    """S0, I0, β0: 필터 시작 시점의 첫 관측과 초기 증가율로부터."""
    gamma = 1.0 / cfg["D"]
    n = cfg["n_ens"]
    y0 = max(df["ILI"].iloc[t0:t0 + 2].mean() - b, 0.2)   # 첫 관측의 독감 초과분
    c0 = y0 / cfg["kappa"]                              # 첫 주 신규감염 비율
    I0 = c0 * cfg["D"] / 7.0 * np.exp(rng.normal(0, 0.5, n))   # 유병 ≈ 일 발생 × 감염기간
    seg = np.log(np.maximum(df["ILI"].iloc[t0:t0 + 4].values - b, 0) + 0.5)
    r = np.polyfit(np.arange(len(seg)), seg, 1)[0] / 7.0 if len(seg) >= 3 else 0.0
    r = float(np.clip(r, -0.05, 0.15))                  # 하루 성장률
    S0 = rng.uniform(*cfg["S0"], n)
    # 초기 성장률이 알려주는 것은 β·S0 = γ + r (유효재생산수) → S0 표본마다 β를 맞춤
    bS = np.clip(gamma + r + rng.normal(0, 0.04, n), 0.6 * gamma, None)
    logb = np.log(bS / S0)
    return np.column_stack([S0, np.minimum(I0, 0.05), logb])

# ---------------------------------------------------------------- 배경 ILI
def bg_level(df):
    """비인플루엔자 기저 ILI: 최근 52주 중 검출률<3%·비연휴 주의 중앙값.
    (ILI×(1-검출률)은 유행기에 같이 커져서 배경으로 쓸 수 없음 → README 참고)"""
    d = df.tail(52)
    d = d[(d["pos"] < 3) & (~d["holiday"])]
    if len(d) >= 3:
        return float(d["ILI"].tail(12).median())
    return float(df["ILI"].tail(52).quantile(0.10))

# ---------------------------------------------------------------- 필터 + 예측
def ah_clim(df_hist, weeks):
    """기점 이후 AH는 쓰면 안 되므로, 과거 같은 주차 평균(기후값)으로 대체."""
    cl = df_hist.dropna(subset=["AH"]).groupby("week")["AH"].mean()
    return np.array([cl.get(w % 100, cl.mean()) for w in weeks])

def fit_and_forecast(df, cfg=None, horizon=5, return_ens=False):
    """관측 두 개를 순차 동화:
       ILI  = κ·c + b                      (예측 대상, 오차 oev_a + (oev_b·y)^2)
       ILI+ = ρ+·κ·c                        (독감 특이 신호, 비율 ρ+의 변동이 커서 오차 큼)"""
    cfg = {**DEFAULT, **(cfg or {})}
    rng = np.random.default_rng(cfg["seed"])
    gamma = 1.0 / cfg["D"]
    use_ah = cfg["theta_ah"] is not None and "AH" in df.columns
    ahm = df["AH"].mean() if use_ah else 0.0
    T = len(df)
    t0 = max(0, T - cfg["window"])
    b = bg_level(df)
    X = init_ensemble(df, t0, cfg, rng, b)

    for t in range(t0, T):
        row = df.iloc[t]
        X[:, 2] += rng.normal(0, cfg["sig_beta"], len(X))            # η: 랜덤워크
        ah_t = cfg["theta_ah"] * (row["AH"] - ahm) if use_ah else 0.0
        S, I, c = propagate_week(X[:, 0], X[:, 1], X[:, 2], gamma, cfg["iota"], ah_t)
        X = np.column_stack([S, I, X[:, 2], c])
        hol = bool(row["holiday"])
        y = row["ILI"]                                                 # 관측 1: ILI
        X = eakf_update(X, cfg["kappa"] * X[:, 3] + b, y,
                        oev_of(max(y - b, 0), cfg, hol), cfg["infl"])
        if cfg["use_plus"] and pd.notna(row["plus"]):                  # 관측 2: ILI+
            yp = row["plus"]
            X = eakf_update(X, cfg["rho_plus"] * cfg["kappa"] * X[:, 3], yp,
                            (cfg["oev_a"] * 0.1 + (cfg["plus_cv"] * yp) ** 2) * (cfg["hol_infl"] if hol else 1),
                            1.0)
        X = X[:, :3]
        X[:, 0] = np.clip(X[:, 0], 1e-4, 1.0)
        X[:, 1] = np.clip(X[:, 1], 1e-9, 1.0 - X[:, 0])
        X[:, 2] = np.clip(X[:, 2], np.log(0.05), np.log(cfg["beta_max"]))

    # ---------- 예측
    origin = int(df["yw"].iloc[-1])
    weeks = next_weeks(origin, horizon)
    ah_f = cfg["theta_ah"] * (ah_clim(df, weeks) - ahm) if use_ah else np.zeros(horizon)
    S, I, logb = X[:, 0].copy(), X[:, 1].copy(), X[:, 2].copy()
    lb = np.zeros(len(S))
    sims = []
    for h in range(horizon):
        logb = logb + rng.normal(0, cfg["sig_beta"], len(S))
        S, I, c = propagate_week(S, I, logb, gamma, cfg["iota"], ah_f[h])
        flu = cfg["kappa"] * c
        flu = np.maximum(flu + rng.normal(0, 1, len(S)) * np.sqrt(cfg["oev_a"] + (cfg["oev_b"] * flu) ** 2), 0)
        lb = lb + rng.normal(0, cfg["bg_sd"], len(S))
        sims.append(flu + b * np.exp(lb))
    sims = np.array(sims)                                  # (horizon, n_ens)
    q = np.quantile(sims, QLEVELS, axis=1).T               # (horizon, 7)
    q = np.maximum.accumulate(np.maximum(q, 0), axis=1)
    out = pd.DataFrame(q, columns=[f"q{l}" for l in QLEVELS])
    out.insert(0, "horizon", range(1, horizon + 1))
    out.insert(0, "target_week", weeks)
    out.insert(0, "origin", origin)
    return (out, sims, X) if return_ens else out

# ---------------------------------------------------------------- 절대습도 계수 추정 (2단계)
def filtered_logbeta(df, cfg=None, start=None):
    """AH 없이 긴 구간 필터를 돌려 주별 사후평균 logβ를 얻는다 (1단계)."""
    cfg = {**DEFAULT, **(cfg or {}), "theta_ah": None}
    rng = np.random.default_rng(cfg["seed"]); gamma = 1 / cfg["D"]
    t0 = 0 if start is None else int(np.where(df["yw"] == start)[0][0])
    b = bg_level(df.iloc[: t0 + 20])
    X = init_ensemble(df, t0, cfg, rng, b); out = []
    for t in range(t0, len(df)):
        row = df.iloc[t]
        X[:, 2] += rng.normal(0, cfg["sig_beta"], len(X))
        S, I, c = propagate_week(X[:, 0], X[:, 1], X[:, 2], gamma, cfg["iota"])
        X = np.column_stack([S, I, X[:, 2], c])
        X = eakf_update(X, cfg["kappa"] * X[:, 3] + b, row["ILI"],
                        oev_of(max(row["ILI"] - b, 0), cfg, row["holiday"]), cfg["infl"])[:, :3]
        X[:, 0] = np.clip(X[:, 0], 1e-4, 1); X[:, 1] = np.clip(X[:, 1], 1e-9, 1 - X[:, 0])
        X[:, 2] = np.clip(X[:, 2], np.log(0.05), np.log(cfg["beta_max"]))
        out.append((row["yw"], X[:, 2].mean(), X[:, 2].std(), row["ILI"] - b))
    return pd.DataFrame(out, columns=["yw", "logb", "logb_sd", "excess"])

def estimate_theta_ah(df, cfg=None, min_excess=3.0, lag=1):
    """2단계: 절기별로 logβ̂_t를 구한 뒤  logβ̂_t = θ0_절기 + θ_AH·AH_{t-lag} + e 를 가중최소제곱.
    유행이 없는 주(excess<min_excess)는 β가 식별되지 않으므로 제외. 기점까지 자료만 넣을 것."""
    rows = []
    seasons = sorted(set(np.where(df["week"] >= 36, df["yw"] // 100, df["yw"] // 100 - 1)))
    for s in seasons:
        st, en = s * 100 + 36, (s + 1) * 100 + 35
        d = df[(df["yw"] >= st) & (df["yw"] <= en)].reset_index(drop=True)
        if len(d) < 20 or d["covid"].mean() > 0.5:
            continue
        fb = filtered_logbeta(d, cfg)
        fb["AH"] = d["AH"].shift(lag).values; fb["season"] = s
        rows.append(fb)
    f = pd.concat(rows).dropna()
    f = f[f["excess"] >= min_excess]
    Xd = pd.get_dummies(f["season"], prefix="s").astype(float)
    Xd["AH"] = f["AH"].values
    w = 1 / np.maximum(f["logb_sd"].values, 0.05) ** 2
    W = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(Xd.values * W, f["logb"].values * W[:, 0], rcond=None)
    resid = f["logb"].values - Xd.values @ coef
    cov = np.linalg.pinv((Xd.values * W).T @ (Xd.values * W)) * np.sum(w * resid ** 2) / max(len(f) - Xd.shape[1], 1)
    return float(coef[-1]), float(np.sqrt(cov[-1, -1])), f

# ---------------------------------------------------------------- 채점·기준모형
def wis(q, y):
    """부록 B. q: 7개 분위수 (QLEVELS 순서)."""
    m = q[3]
    tot = 0.5 * abs(y - m)
    for a, (lo, hi) in zip([0.05, 0.20, 0.50], [(0, 6), (1, 5), (2, 4)]):
        l, u = q[lo], q[hi]
        IS = (u - l) + 2 / a * (l - y) * (y < l) + 2 / a * (y - u) * (y > u)
        tot += a / 2 * IS
    return tot / 3.5

def baseline(df, horizon=5):
    """부록 B 기준모형: 기점값 + 대칭화한 h주 변화량 경험분위수."""
    y = df.loc[~df["covid"], "ILI"].values
    last = df["ILI"].iloc[-1]
    rows = []
    for h in range(1, horizon + 1):
        d = y[h:] - y[:-h]
        d = np.concatenate([d, -d])
        rows.append(np.maximum(last + np.quantile(d, QLEVELS), 0))
    return np.array(rows)
