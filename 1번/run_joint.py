"""
문제 1 전체 실행 (동시추정 SIR, 기온, 관측지연)
  python run_joint.py                                   # 1-가 A·B·C + 1-나
  python run_joint.py --D 202xxx                        # 기점 D 공개 후 추가
  python run_joint.py --extra 2차배포.csv                 # 1-다 (열: 년주, ILI) → 기점 = extra의 마지막 주
  python run_joint.py --data 2차배포_전체.xlsx --c1 202639  # 2차 배포가 전체 파일 형식이면
출력: out_joint/forecast_long.csv, scores.csv, params.csv, 그림 png
"""
import argparse, os, time
import numpy as np, pandas as pd
import core, joint, stat_model

def pos_rule(df, T=8.0, D=1.0):
    """검출률 규칙: 기점의 (마지막으로 알려진) 검출률 < T% 이고 그 2주 전보다 D%p 이하로만 올랐으면 True.
    'ILI는 오르는데 검출률은 오르지 않는' 상황 = 비인플루엔자 원인 상승 → SIR(인플루엔자 기전) 가중을 낮춤.
    근거: 기점 A(2023-37) 오차 분석. 2016~2022 기점 24개(A 제외)에서 성능 저하 없음 확인(0.425 → 0.423).
    2차 배포처럼 최근 검출률이 없으면 마지막으로 관측된 주의 값으로 판단."""
    p = df["pos"].values.astype(float); k = np.where(np.isfinite(p))[0]
    if len(k) < 3:
        return False
    i = k[-1]; j = i - 2
    if j < 0 or not np.isfinite(p[j]):
        return False
    return bool(p[i] < T and p[i] - p[j] <= D)

TEAM, MODEL = "T00", "ens_sirAHschool_gbm_posrule"

def to_s(yw): return f"{yw//100}-{yw%100:02d}"

def read_extra(path):
    e = pd.read_excel(path) if path.endswith("xlsx") else pd.read_csv(path)
    return e.rename(columns={"yw": "년주"})[["년주", "ILI"]]

def main(a):
    os.makedirs(a.out, exist_ok=True)
    temp = joint.load_ah(a.ah) if a.ah else joint.load_temp(a.temp)
    jobs = [("1A", 202337), ("1A", 202541), ("1A", 202450)]
    if a.D: jobs.append(("1A", a.D))
    jobs.append(("1B", 202634))
    extra = read_extra(a.extra) if a.extra else None
    if extra is not None:
        jobs.append(("1C", int(extra["년주"].max())))
    elif a.c1:
        jobs.append(("1C", a.c1))
    full = core.load_data(a.data); truth = dict(zip(full["yw"], full["ILI"]))
    long, scores, params, results = [], [], [], {}
    for task, o in jobs:
        t = time.time()
        r = joint.run_origin(a.data, temp, o, extra=extra if task == "1C" else None)
        r["fc_sir"] = r["fc"].copy()
        if a.sir_weight < 1:                       # 앙상블: SIR와 GBM 분위수의 가중 평균 (가중치는 2016~2022 기점에서 선택)
            g = stat_model.forecast(r["df"], temp)
            qc = [c for c in r["fc"].columns if c.startswith("q")]
            r["fc_gbm"] = g
            w = a.sir_weight
            r["rule"] = bool(a.pos_rule) and pos_rule(r["df"])
            if r["rule"]:
                w = a.w_low                         # 검출률 규칙 작동 → SIR 가중 낮춤
            print(f"  [{task} {o}] 검출률 규칙 {'작동' if r['rule'] else '미작동'} → SIR 가중 {w}", flush=True)
            r["fc"][qc] = np.maximum.accumulate(w * r["fc"][qc].values + (1 - w) * g[qc].values, axis=1)
        results[o] = r
        e = r["best"]["est"]; cur_S0 = float(np.median(r["best"]["est"]["S0"][-1]))
        params.append(dict(task=task, origin=o, L=r["L"], loglik=round(r["best"]["ll"], 1),
                           theta0=e["theta0"], thetaT=e["thetaT"], beta_x_per_unit_drop=np.exp(-e["thetaT"]),
                           thetaS=e["thetaS"], beta_x_school=np.exp(e["thetaS"]), sigma_eta=np.exp(e["log_sig"]), w_sameweek=joint.expit(e["logit_w"]),
                           sigma_b=np.exp(e["log_sigb"]), S0_current=cur_S0, b=r["b"], sec=round(time.time() - t)))
        bl = core.baseline(r["df"])
        for i, row in r["fc"].iterrows():
            tw = int(row["target_week"]); q = row.values[3:].astype(float)
            for lv, v in zip(core.QLEVELS, q):
                long.append(dict(team_id=TEAM, model_id=MODEL, task=task, origin=to_s(o), scenario_id="NA",
                                 target="ili", target_week=to_s(tw), horizon=i + 1, quantile=lv, value=round(float(v), 3)))
            y = truth.get(tw) if task == "1A" else None
            if y is not None:
                scores.append(dict(origin=o, h=i + 1, y=y, median=q[3], wis=core.wis(q, y), wis_baseline=core.wis(bl[i], y),
                                   in50=q[2] <= y <= q[4], in95=q[0] <= y <= q[6]))
        print(task, o, "done", round(time.time() - t), "s", flush=True)
    pd.DataFrame(long).to_csv(f"{a.out}/forecast_long.csv", index=False)
    pd.DataFrame(params).to_csv(f"{a.out}/params.csv", index=False)
    if scores:
        s = pd.DataFrame(scores); s.to_csv(f"{a.out}/scores.csv", index=False)
        g = s.groupby("origin")[["wis", "wis_baseline"]].mean(); g["relWIS"] = g.wis / g.wis_baseline
        print(g.round(2)); print("overall relWIS", round(s.wis.mean() / s.wis_baseline.mean(), 3))
    import figures; figures.make_all(results, truth, jobs, a.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/user-data/uploads/인플루엔자_학습_데이터.xlsx")
    ap.add_argument("--temp", default="/mnt/user-data/uploads/기온2016_20260929.csv")
    ap.add_argument("--ah", default=None, help="절대습도 csv를 주면 기온 대신 사용")
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--c1", type=int, default=None)
    ap.add_argument("--out", default="out_joint")
    ap.add_argument("--sir_weight", type=float, default=0.8, help="앙상블에서 SIR 가중치 (1이면 SIR 단독)")
    ap.add_argument("--pos_rule", type=int, default=1, help="1이면 검출률 규칙 사용 (검출률<8%% & 2주 변화≤1%%p → SIR 가중 w_low)")
    ap.add_argument("--w_low", type=float, default=0.2, help="검출률 규칙 작동 시 SIR 가중치")
    main(ap.parse_args())
