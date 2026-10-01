"""
구조가 다른 두 번째 모형: 직접(direct) 분위수 그래디언트 부스팅
  목표   : log(ILI_{t+h}+1) - log(ILI_t+1)       (h = 1..5 각각 별도 모형, 분위수 7개 각각 별도 모형)
  특징   : 최근 로그 증가율 3개, 현재 수준, 검출률(없으면 결측 처리), 절기 주차(sin/cos),
           기점·목표 주의 연휴/연휴 다음 주/학교 수업 여부, 기점 절대습도와 목표 주 기후값
  학습자료: 기점까지 관측된 (t, t+h) 쌍만 사용. 코로나 기간 제외.
SIR과 달리 기전 가정이 없고, 과거 비슷한 상황에서 실제로 몇 % 움직였는지를 배운다.
"""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
import core

POST_HOL = {core.next_weeks(h, 1)[0] for h in core.HOLIDAY}

def features(df, ah, idx, tw):
    """df 행 idx(기점)에서 목표 주 tw를 예측하는 특징 벡터."""
    y = np.log(df["ILI"].values + 1)
    g = lambda k: y[idx - k] - y[idx - k - 1] if idx - k - 1 >= 0 else np.nan
    o = int(df["yw"].iloc[idx]); wk = o % 100; h = core.next_weeks(o, 60).index(tw) + 1
    return [y[idx], g(0), g(1), g(2), df["pos"].iloc[idx],
            np.sin(2 * np.pi * wk / 52), np.cos(2 * np.pi * wk / 52),
            int(o in core.HOLIDAY), int(o in POST_HOL), int(tw in core.HOLIDAY), int(tw in POST_HOL),
            core.school_flag(o), core.school_flag(tw), ah.get(o, np.nan), ah.get(("clim", tw % 100), np.nan)]

def ah_lookup(ahdf, origin):
    if ahdf is None:
        return {}
    h = ahdf[ahdf["yw"] <= origin]
    d = dict(zip(h["yw"], h["Tz"]))
    for w, v in h.groupby(h["yw"] % 100)["Tz"].mean().items():
        d[("clim", w)] = v
    return d

def forecast(df, ahdf=None, horizon=5, seed=0):
    df = df.reset_index(drop=True)
    origin = int(df["yw"].iloc[-1]); ah = ah_lookup(ahdf, origin)
    ok = ~df["covid"].values; yv = np.log(df["ILI"].values + 1)
    out = []
    for h in range(1, horizon + 1):
        X, Y = [], []
        for i in range(3, len(df) - h):
            if ok[i - 3:i + h + 1].all():
                tw = int(df["yw"].iloc[i + h])
                X.append(features(df, ah, i, tw)); Y.append(yv[i + h] - yv[i])
        X, Y = np.array(X, float), np.array(Y)
        tw = core.next_weeks(origin, h)[-1]
        x0 = np.array([features(df, ah, len(df) - 1, tw)], float)
        qs = []
        for q in core.QLEVELS:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150, learning_rate=0.05,
                                              max_depth=3, min_samples_leaf=15, random_state=seed)
            m.fit(X, Y); qs.append(m.predict(x0)[0])
        qs = np.sort(qs)
        out.append(np.maximum(np.exp(yv[-1] + qs) - 1, 0))
    q = np.maximum.accumulate(np.array(out), axis=1)
    fc = pd.DataFrame(q, columns=[f"q{l}" for l in core.QLEVELS])
    fc.insert(0, "horizon", range(1, horizon + 1)); fc.insert(0, "target_week", core.next_weeks(origin, horizon))
    fc.insert(0, "origin", origin)
    return fc
