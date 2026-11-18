"""Introductory efficiency figure: replot audited results; no model loading/training."""
from pathlib import Path
import csv,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
root=Path("/home/home_new/qsmx/pycodes/GreyFerm")
src=root/"test_results/varbridge_timebase_analysis/efficiency/results.csv"
dst=root/"test_results/varbridge_timebase_analysis/figures"
before=hashlib.sha256(src.read_bytes()).hexdigest()
order=["DLinear","PatchTST","iTransformer","Crossformer","TimeMixer","TimesNet","TimeLLM","VarBridge"]
r=pd.read_csv(src).set_index("model").loc[order].reset_index()
assert len(r)==8 and r.model.nunique()==8
assert (r.epochs==100).all() and r.physical_gpu.nunique()==1 and (r.test_windows==381).all()
assert (r.samples==160).all() and (r.seed==2021).all()
plt.rcParams.update({"font.family":"DejaVu Serif","font.size":11,"axes.labelsize":11,"axes.titlesize":12,"axes.spines.top":False,"axes.spines.right":False,"mathtext.fontset":"dejavuserif"})
palette=["#5785a6","#33988d","#8e9d45","#947bb5","#c7964c","#79a6af","#aa72a5","#d45147"]
markers=["o","s","^","D","v","P","X","*"]
fig,axs=plt.subplots(1,2,figsize=(12.6,4.45),gridspec_kw={"wspace":.25})
fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.88)
offsets0={"DLinear":(6,8),"PatchTST":(-8,-20),"iTransformer":(4,9),"Crossformer":(-6,11),"TimeMixer":(1,-22),"TimesNet":(-5,12),"TimeLLM":(13,12),"VarBridge":(9,-3)}
offsets1={"DLinear":(6,8),"PatchTST":(-6,-22),"iTransformer":(5,8),"Crossformer":(-8,10),"TimeMixer":(-5,9),"TimesNet":(8,7),"TimeLLM":(-8,-15),"VarBridge":(7,-1)}
for i,row in r.iterrows():
    label="Time-LLM" if row.model=="TimeLLM" else row.model
    color=palette[i]; emph=row.model=="VarBridge"
    axs[0].scatter(row.latency_median_ms,row.mse,s=row.peak_allocated_MiB*.65,color=color,edgecolor=color,linewidth=1.2,alpha=.5 if not emph else .85,zorder=3)
    axs[1].scatter(row.resident_parameters,row.mse,s=80 if not emph else 155,marker=markers[i],color=color,edgecolor=color,linewidth=.7,zorder=4)
    for ax,x,offset in [(axs[0],row.latency_median_ms,offsets0[row.model]),(axs[1],row.resident_parameters,offsets1[row.model])]:
        dx,dy=offset
        ax.annotate(label,(x,row.mse),xytext=(dx,dy),textcoords="offset points",ha="right" if dx<0 else "left",va="center",fontsize=10.6,fontweight="bold" if emph else "normal",color=color,zorder=6)
for ax in axs:
    ax.set_xscale("log");ax.set_ylim(.125,.585);ax.set_ylabel("Full-test MSE")
    ax.set_yticks([.15,.25,.35,.45,.55]);ax.grid(axis="y",color="#e9e9e9",linewidth=.7);ax.tick_params(labelsize=10)
axs[0].set_xlim(.29,115)
axs[1].set_xlim(1200,5.5e8)
axs[0].set_xlabel("Cached prediction latency (ms, log scale)")
axs[1].set_xlabel("Resident predictor parameters (log scale)")
axs[0].set_title("(a) Accuracy and inference cost",loc="left",pad=12)
axs[1].set_title("(b) Accuracy and predictor size",loc="left",pad=12)
axs[0].text(.03,.025,"Bubble area: peak allocated GPU memory",transform=axs[0].transAxes,fontsize=9,color="#555555")
fig.savefig(dst/"efficiency_intro_overview.png",dpi=400,facecolor="white")
plt.close(fig)
assert hashlib.sha256(src.read_bytes()).hexdigest()==before
(dst/"efficiency_intro_provenance.txt").write_text("Source: "+str(src)+"\nSHA256: "+before+"\nEight unchanged audited rows; no training; latency median and MSE unchanged.\n")
print("INTRO_EFFICIENCY_FIGURE_COMPLETE",dst/"efficiency_intro_overview.png")

