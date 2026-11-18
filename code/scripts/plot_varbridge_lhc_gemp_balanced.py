"""Plot the balanced three-trial subset; all successful raw replies remain preserved."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R=Path("/home/home_new/qsmx/pycodes/GreyFerm/test_results/varbridge_lhc_gemp")
rows=list(csv.DictReader((R/"api/fixed_checkpoint_downstream.csv").open()))
rows=[x for x in rows if x["model"]=="Original descriptions" or int(x["trial"])<3]
assert len(rows)==74
models=["DeepSeek-V3.2","DeepSeek-R1-0528","DeepSeek-V3-0324","Qwen3.8-27B","GLM-5.3","Kimi-K2.6"]
labels=["DS-V3.2","DS-R1\n0528","DS-V3\n0324","Qwen3.8\n27B","GLM5.3","KimiK2.6"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.labelsize":12,"axes.titlesize":12,"xtick.labelsize":9,"ytick.labelsize":10,"axes.spines.top":False,"axes.spines.right":False})
for task,title in [("EFP_long_sub2","EFP-sub2"),("CHO","CHO")]:
 q=[x for x in rows if x["dataset"]==task];base=next(x for x in q if x["model"]=="Original descriptions")
 fig,axs=plt.subplots(1,2,figsize=(7.8,3.4),layout="constrained")
 for ax,metric in zip(axs,["mse","mae"]):
  for i,model in enumerate(models):
   z=[x for x in q if x["model"]==model];assert len(z)==6
   vals=np.array([100*(float(x[metric])/float(base[metric])-1) for x in z])
   offsets=np.linspace(-.1,.1,6)
   ax.scatter(i+offsets,vals,color="#8c7ba7",s=23,alpha=.65,zorder=3)
   ax.plot([i,i],[vals.min(),vals.max()],color="#8c7ba7",lw=1.4,zorder=2)
   ax.scatter([i],[vals.mean()],color="#3b7778",marker="D",s=49,zorder=4)
  ax.axhline(0,color="#b85b50",lw=1.2,ls="--")
  ax.set_xticks(range(6),labels);ax.set_ylabel("Change in "+metric.upper()+" (%)");ax.set_title(title+" / "+metric.upper());ax.grid(axis="y",alpha=.16)
 fig.suptitle("Frozen VarBridge weights; two wordings × three API replies",fontsize=12.5,fontweight="bold")
 fig.supxlabel("DS = DeepSeek; dots = replies; diamonds = means; lower is better",fontsize=9)
 fig.savefig(R/"figures"/("llm_source_forecasting_"+task+".png"),dpi=360,bbox_inches="tight",facecolor="white");plt.close(fig)
(R/"api/plot_cohort.json").write_text(json.dumps(dict(successful_raw_calls=81,plotted_balanced_calls=72,trials=[0,1,2],samples_per_model_per_task=6,weights_fixed=True,dots="generated-description variation, not training seeds",zero_line="original description on identical windows",missing_planned_calls=39),indent=2))
print("BALANCED_THREE_TRIAL_PLOTS_COMPLETE")
