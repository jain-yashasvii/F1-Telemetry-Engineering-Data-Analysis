"""Feature-set sensitivity experiment for unsupervised F1 telemetry."""
from __future__ import annotations
import argparse
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from telemetry_processor import get_telemetry_matrix
OPTIONAL=["Throttle","Brake","Gear","Brake_Rate","TyreLife"]
def run_experiment(telemetry,contamination=.02,output_dir=None):
    if not .001<=float(contamination)<=.25: raise ValueError("contamination must be between 0.001 and 0.25")
    frame=telemetry.copy().reset_index(drop=True)
    if not {"Speed","RPM"}.issubset(frame.columns): raise ValueError("Telemetry must include Speed and RPM.")
    sets={"A - Speed + RPM":["Speed","RPM"],"B - Driver inputs":["Speed","RPM","Throttle","Brake"],"C - All available":list(dict.fromkeys(["Speed","RPM",*[c for c in OPTIONAL if c in frame]]))}
    valid=frame[["Speed","RPM"]].apply(pd.to_numeric,errors="coerce").notna().all(axis=1)
    idx=np.flatnonzero(valid.to_numpy())
    if len(idx)<20: raise ValueError(f"Need at least 20 rows with Speed and RPM; found {len(idx)}.")
    cut=int(len(idx)*.8); train_idx,test_idx=idx[:cut],idx[cut:]
    outputs,summaries,flags=[],[],{}
    for label,wanted in sets.items():
        cols=[c for c in wanted if c in frame]
        raw=frame[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        cols=[c for c in cols if raw.loc[train_idx,c].notna().any()]
        if not {"Speed","RPM"}.issubset(cols): raise ValueError(f"Required channels unavailable in {label}.")
        imp=SimpleImputer(strategy="median"); xtr=imp.fit_transform(raw.loc[train_idx,cols]); xte=imp.transform(raw.loc[test_idx,cols])
        scale=StandardScaler(); xtr_s=scale.fit_transform(xtr); xte_s=scale.transform(xte)
        model=IsolationForest(n_estimators=200,contamination=contamination,random_state=42,n_jobs=-1).fit(xtr_s)
        if_score=-model.score_samples(xte_s); if_flag=model.predict(xte_s)==-1
        center=np.median(xtr,axis=0); mad=1.4826*np.median(np.abs(xtr-center),axis=0); std=np.std(xtr,axis=0)
        spread=np.where(mad>1e-12,mad,np.where(std>1e-12,std,1.))
        ztr=np.max(np.abs((xtr-center)/spread),axis=1); zte=np.max(np.abs((xte-center)/spread),axis=1)
        threshold=float(np.quantile(ztr,1-float(contamination))); z_flag=zte>threshold
        row=pd.DataFrame({"row_index":test_idx,"Time_s":frame.loc[test_idx,"Time_s"].to_numpy() if "Time_s" in frame else test_idx,"Distance":frame.loc[test_idx,"Distance"].to_numpy() if "Distance" in frame else np.nan,"Speed":frame.loc[test_idx,"Speed"].to_numpy(),"RPM":frame.loc[test_idx,"RPM"].to_numpy(),"feature_set":label,"features_used":", ".join(cols),"isolation_score":if_score,"isolation_flag":if_flag,"statistical_score":zte,"statistical_flag":z_flag,"statistical_threshold":threshold})
        # Consecutive flagged holdout samples are grouped into simple windows.
        window_ids = np.zeros(len(if_flag), dtype=int)
        window_number = 0
        for pos, flagged in enumerate(if_flag):
            if flagged and (pos == 0 or not if_flag[pos-1] or test_idx[pos] != test_idx[pos-1] + 1):
                window_number += 1
            if flagged:
                window_ids[pos] = window_number
        row["anomaly_window"] = window_ids
        for c in ("Throttle","Brake"):
            if c in frame: row[c]=frame.loc[test_idx,c].to_numpy()
        outputs.append(row); flags[(label,"IsolationForest")]=set(test_idx[if_flag]); flags[(label,"RobustZ")]=set(test_idx[z_flag])
        overlap,union=int(np.sum(if_flag&z_flag)),int(np.sum(if_flag|z_flag))
        summaries.append({"feature_set":label,"features_used":", ".join(cols),"holdout_samples":len(test_idx),"contamination":float(contamination),"isolationforest_flagged":int(if_flag.sum()),"isolationforest_rate_pct":float(if_flag.mean()*100),"robust_z_flagged":int(z_flag.sum()),"robust_z_rate_pct":float(z_flag.mean()*100),"method_overlap_count":overlap,"method_overlap_jaccard":overlap/union if union else 1.})
    summary=pd.DataFrame(summaries)
    for method in ("IsolationForest","RobustZ"):
        for (an,a),(bn,b) in combinations([(n,flags[(n,method)]) for n in sets],2):
            union=a|b; summary.loc[summary.feature_set==an,f"{method}_Jaccard_vs_{bn}"]=len(a&b)/len(union) if union else 1.
    detail=pd.concat(outputs,ignore_index=True)
    if output_dir is not None:
        out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
        detail.to_csv(out/"anomaly_scores.csv",index=False); summary.to_csv(out/"anomaly_summary.csv",index=False); _plots(detail,summary,out)
    return detail,summary
def _plots(detail,summary,out):
    x="Distance" if detail["Distance"].notna().any() else "Time_s"
    for label,data in detail.groupby("feature_set",sort=False):
        key=label[0].lower(); signals=[c for c in ["Speed","Throttle","Brake"] if c in data]
        px.line(data,x=x,y=signals,title=f"Telemetry signals - {label}").write_html(out/f"telemetry_{key}.html",include_plotlyjs="cdn")
        px.scatter(data,x=x,y="isolation_score",color="isolation_flag",hover_data=["Speed","RPM","statistical_score"],title=f"Isolation Forest score - {label}").write_html(out/f"anomalies_{key}.html",include_plotlyjs="cdn")
    px.bar(summary,x="feature_set",y=["isolationforest_rate_pct","robust_z_rate_pct"],barmode="group",title="Holdout anomaly rate by features and method",labels={"value":"Flagged samples (%)","variable":"Method"}).write_html(out/"feature_set_comparison.html",include_plotlyjs="cdn")
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--year",type=int,required=True); p.add_argument("--track",required=True); p.add_argument("--session",default="Q"); p.add_argument("--driver",required=True); p.add_argument("--contamination",type=float,default=.02); p.add_argument("--output",default="results"); a=p.parse_args()
    try:
        telemetry=get_telemetry_matrix(a.year,a.track,a.session,a.driver); _,summary=run_experiment(telemetry,a.contamination,a.output)
    except Exception as exc: p.error(f"Experiment could not run: {exc}")
    print("Chronological 80/20 holdout. Anomaly flags are not verified driving errors."); print(summary.to_string(index=False)); print(f"Saved CSV and HTML outputs to {Path(a.output).resolve()}")
if __name__=="__main__": main()



