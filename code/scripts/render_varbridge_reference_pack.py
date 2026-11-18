"""Materialize the LHC/GEMP correspondence as source-backed PNG tables and figures."""
from pathlib import Path
import os,sys,json,csv,math,statistics,ast,textwrap,collections
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm");sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
R=ROOT/"test_results/varbridge_lhc_gemp";D=R/"deliverables";D.mkdir(exist_ok=True)
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde
torch.set_num_threads(2)
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.labelsize":11,"axes.titlesize":11,"legend.fontsize":9,"xtick.labelsize":9,"ytick.labelsize":9,"axes.spines.top":False,"axes.spines.right":False})
C=["#5689a3","#298e83","#c96b5b","#a08bb5","#c39a53"]
TASKS=["EFP_long_sub1","EFP_long_sub2","EFP_long_sub3","CHO"];DISPLAY={t:t.replace("EFP_long_","EFP-") for t in TASKS}
bind=pd.read_csv(R/"target_bindings.csv")
SYMBOL={t:bind[bind.dataset==t].symbol.tolist() for t in TASKS}
FEATURE={t:bind[bind.dataset==t].variable.tolist() for t in TASKS}
def read(p):return list(csv.DictReader(p.open()))
def save(fig,name):
 target=D/(name+".png")
 if target.is_symlink():target.unlink()
 fig.savefig(target,dpi=360,bbox_inches="tight",facecolor="white");plt.close(fig);print("RENDERED",name,flush=True)
def link(name,p):
 target=D/(name+".png");p=Path(p).resolve();assert p.exists(),p
 if target.is_symlink():target.unlink()
 assert not target.exists(),target
 target.symlink_to(p)
def drawtable(name,columns,rows,title,width=9.5,fontsize=9.5,widths=None,row_height=.27,footer=None,colors=None):
 height=max(2.0,(len(rows)+1)*row_height+1.1)
 fig=plt.figure(figsize=(width,height));ax=fig.add_axes([.015,.055,.97,.885]);ax.axis("off")
 table=ax.table(cellText=rows,colLabels=columns,cellLoc="center",loc="center",bbox=[0,0,1,1],colWidths=widths)
 table.auto_set_font_size(False);table.set_fontsize(fontsize)
 for (i,j),cell in table.get_celld().items():
  cell.set_linewidth(.45);cell.set_edgecolor("#d8e1e1")
  if i==0:cell.set_facecolor("#e7efef");cell.set_text_props(weight="bold");cell.set_height(cell.get_height()*1.4)
  else:cell.set_facecolor("#f6f8f8" if i%2==0 else "white")
 for (i,j),color in (colors or {}).items():table[(i+1,j)].get_text().set_color(color);table[(i+1,j)].get_text().set_weight("bold")
 fig.text(.5,.985,title,ha="center",va="top",fontsize=12,fontweight="bold")
 if footer:fig.text(.5,.018,footer,ha="center",va="bottom",fontsize=8.5)
 save(fig,name)
# Seven distinct table products cover the ten source table entries.
stats=read(R/"descriptive_statistics.csv")
def names_from_ast(path,key):
 tree=ast.parse(path.read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id==key for z in n.targets))
efp_names=names_from_ast(ROOT/"preprocess_window_semantic.py","NAMES")
cho_names=names_from_ast(ROOT/"scripts/cho_data.py","NAMES")
sensor={t:{f:(cho_names[j] if t=="CHO" else efp_names[f]) for j,f in enumerate(FEATURE[t])} for t in TASKS}
rows=[[DISPLAY[x["dataset"]],textwrap.fill(sensor[x["dataset"]][x["variable"]],25),x["symbol"]]+[f"{float(x[k]):.2f}" for k in ["mean","median","sample_sd","q25","q75"]] for x in stats]
drawtable("LHC_Table_I",["Task","Sensor","Symbol","Mean","Median","SD","Q25","Q75"],rows,"Training-set descriptive statistics",fontsize=9,widths=[.11,.25,.07,.115,.115,.11,.115,.115],row_height=.33)
link("GEMP_Table_I",D/"LHC_Table_I.png")
inv=read(D/"baseline_inventory.csv")
rows=[[x["model"],x["modality"],textwrap.fill(x["representation"],36),x["budget_max_epochs"],x["observed_completed_epochs_min"]+"–"+x["observed_completed_epochs_max"]] for x in inv]
drawtable("LHC_Table_II",["Model","Inputs","Numerical/text interface","Epoch cap","Actual epochs"],rows,"Model interfaces and actual training budgets",fontsize=9.5,widths=[.16,.20,.41,.10,.13],row_height=.43,footer="Existing early-stopped baselines keep their true epoch counts; new fits use exact 100/10 budgets.")
link("GEMP_Table_II",D/"LHC_Table_II.png")
rows=[[DISPLAY[t],SYMBOL[t][j],textwrap.fill(sensor[t][f],37),str(j+1)] for t in TASKS for j,f in enumerate(FEATURE[t])]
drawtable("LHC_Table_III",["Task","Target","Measured variable","Own mechanism slot"],rows,"Variable definitions and target-wise binding",width=8.2,fontsize=10,widths=[.15,.12,.53,.20],row_height=.33,footer="Raw prompt slot0 is the observed-window summary; slots1–5 hold the five own-target mechanisms.")
# Functional reference groups are an additional view of Table III, never model partitions.
groups=read(D/"researcher_functional_groups.csv")
drawtable("LHC_Table_III_reference_groups",["Task","Reference group","Functional label"],[[DISPLAY[x["dataset"]],"g"+x["group_index"],x["group_name"]] for x in groups],"Researcher-defined LLM assessment groups",width=8.2,fontsize=10,widths=[.16,.19,.65],row_height=.4,footer="Assessment labels only; not independent expert truth or VarBridge architecture.")
main=read(D/"main_three_seed_statistics.csv")
order=[x["model"] for x in inv if x["model"]!="VarBridge"]+["VarBridge"]
lookup={(x["dataset"],x["model"],str(x["horizon"]),x["metric"]):x for x in main}
rows=[];color={}
for model in order:
 for H in [6,12,24,48]:
  row=[model if H==6 else "",str(H)]
  for t in TASKS:
   for metric in ["mse","mae"]:
    x=lookup[(t,model,str(H),metric)];row.append(f"{float(x['mean']):.3f} ± {float(x['sample_sd']):.4f}")
    ranking=sorted(order,key=lambda m:float(lookup[(t,m,str(H),metric)]["mean"]))
    if model==ranking[0]:color[(len(rows),len(row)-1)]=C[2]
    elif model==ranking[1]:color[(len(rows),len(row)-1)]=C[0]
  rows.append(row)
cols=["Model","H"]+[DISPLAY[t]+"\n"+metric.upper() for t in TASKS for metric in ["mse","mae"]]
drawtable("LHC_Table_IV",cols,rows,"Main forecasting results: three actual training seeds",width=11.6,fontsize=9,widths=[.13,.035]+[.104375]*8,row_height=.265,colors=color,footer="Mean ± sample SD, seeds 2021/2022/2023. Red: lowest mean; blue: second-lowest. VarBridge100, baseline cap10.")
link("GEMP_Table_III",D/"LHC_Table_IV.png")
tests=read(D/"exploratory_seed_pair_tests.csv");testlookup={(x["dataset"],x["baseline"],str(x["horizon"]),x["metric"]):x for x in tests}
rows=[]
for t in TASKS:
 reference="PatchTST" if t=="CHO" else "Crossformer"
 for H in ["6","12","24","48","Avg"]:
  row=[DISPLAY[t],H]
  for model in ["VarBridge",reference]:
   for metric in ["mse","mae"]:
    x=lookup[(t,model,H,metric)];star=""
    if model=="VarBridge" and H!="Avg":
     p=testlookup[(t,reference,H,metric)]
     if p["fusion_lower_and_adjusted_p_below05"]=="True":star="*"
    row.append(f"{float(x['ci95_lower']):.4f}–{float(x['ci95_upper']):.4f}"+star)
  rows.append(row)
drawtable("LHC_Table_V",["Task","H","VarBridge\nMSE95%CI","VarBridge\nMAE95%CI","Reference\nMSE95%CI","Reference\nMAE95%CI"],rows,"Three-seed intervals; Crossformer(EFP) / PatchTST(CHO)",width=9.4,fontsize=9.5,widths=[.13,.065,.20125,.20125,.20125,.20125],row_height=.29,footer="* Exploratory paired t, Holm p<0.05 across384 comparisons; n=3 seeds. No significance test on averaged horizons.")
modal=read(D/"llm_modal_tables.csv");available_replies=min(int(x["replies"]) for x in modal);names=list(dict.fromkeys(x["model"] for x in modal))
for paper,metrics,title in [("LHC",["score_MAE","QWK","NDCG","latency_mean_seconds"],"LLM definition scoring: original / paraphrased prompt"),("GEMP",["latency_mean_seconds","partition_FMI","SSim"],"LLM partition/score consistency: original / paraphrased prompt")]:
 rows=[]
 for t in ["EFP_long_sub2","CHO"]:
  for m in names:
   z=[next(x for x in modal if x["dataset"]==t and x["model"]==m and int(x["prompt_version"])==v) for v in [0,1]]
   rows.append([DISPLAY[t],m]+[" / ".join(f"{float(x[k]):.1f}" if k=="latency_mean_seconds" else f"{float(x[k]):.3f}" for x in z) for k in metrics])
 drawtable(paper+"_Table_"+("VI" if paper=="LHC" else "IV"),["Task","LLM"]+[{"score_MAE":"Score MAE↓","latency_mean_seconds":"Time(s)↓","partition_FMI":"Partition FMI↑","SSim":"Score similarity↑","QWK":"QWK↑","NDCG":"NDCG↑"}[k] for k in metrics],rows,title,width=9.4,fontsize=9,row_height=.31,footer=f"{available_replies} complete replies per wording; modal ratings, mean time. Researcher rubric.")
# Architectures and actual definition-processing flow are existing approved assets.
link("LHC_Figure_01",R/"figures/varbridge_full_architecture.png")
link("GEMP_Figure_01",R/"figures/varbridge_full_architecture.png")
link("LHC_Figure_09",R/"figures/llm_definition_to_forecast_workflow.png")
link("GEMP_Figure_15",D/"LHC_Figure_09.png")
# Distribution gallery: full training range, retained outliers; original units.
from analyze_varbridge_timebase import raw_task
fig,axs=plt.subplots(4,5,figsize=(11,8.4),layout="constrained")
for row,t in enumerate(TASKS):
 raw,features,ids,_,_,_=raw_task(t);v=raw[raw.batch_id.isin(ids["train"])][features].to_numpy(float)
 for j,a in enumerate(axs[row]):
  a.hist(v[:,j],bins="fd",density=True,color=C[j],alpha=.7,edgecolor="white",lw=.25)
  a.set(title=DISPLAY[t]+" / "+SYMBOL[t][j],xlabel="Original measurement");a.grid(axis="y",alpha=.13)
  if j==0:a.set_ylabel("Density")
save(fig,"LHC_Figure_02")
# Four training-only numerical relationship matrices.
rel=read(ROOT/"test_results/varbridge_timebase_analysis/data_relationships.csv")
fig,axs=plt.subplots(2,2,figsize=(7.6,6.4),layout="constrained")
for a,t in zip(axs.flat,TASKS):
 mat=np.array([[float(next(x["value"] for x in rel if x["dataset"]==t and x["matrix"]=="within_batch_level_pearson" and x["row"]==f and x["column"]==g)) for g in FEATURE[t]] for f in FEATURE[t]])
 im=a.imshow(mat,cmap="RdBu_r",vmin=-1,vmax=1)
 a.set(xticks=range(5),xticklabels=SYMBOL[t],yticks=range(5),yticklabels=SYMBOL[t],title=DISPLAY[t])
 for i in range(5):
  for j in range(5):a.text(j,i,f"{mat[i,j]:.2f}",ha="center",va="center",fontsize=9,color="white" if abs(mat[i,j])>.6 else "#27373b")
fig.colorbar(im,ax=axs.ravel().tolist(),label="Within-batch level Pearson correlation",fraction=.028,pad=.02)
save(fig,"LHC_Figure_04");link("GEMP_Figure_04",D/"LHC_Figure_04.png")
lag=read(R/"training_lagged_associations.csv")
fig,axs=plt.subplots(2,2,figsize=(7.8,6.7),layout="constrained")
vmax=max(float(x["log1p_F"]) for x in lag)
for a,t in zip(axs.flat,TASKS):
 m=np.zeros((5,5))
 for x in lag:
  if x["dataset"]==t:m[FEATURE[t].index(x["target"]),FEATURE[t].index(x["source"])]=float(x["log1p_F"])
 im=a.imshow(m,cmap="YlGnBu",vmin=0,vmax=vmax);a.set(xticks=range(5),xticklabels=SYMBOL[t],yticks=range(5),yticklabels=SYMBOL[t],title=DISPLAY[t],xlabel="Lagged source",ylabel="Predicted sensor")
 for i in range(5):
  for j in range(5):a.text(j,i,"—" if i==j else f"{m[i,j]:.1f}",ha="center",va="center",fontsize=9,color="white" if m[i,j]>.6*vmax else "#27373b")
fig.colorbar(im,ax=axs.ravel().tolist(),label="log(1 + F statistic); predictive association",fraction=.028,pad=.02)
save(fig,"GEMP_Figure_03")
# All24 ablations, including variants that improve over the full model.
abl=read(R/"existing_ablation_aggregates.csv");variants=[x["variant"] for x in abl if x["dataset"]=="EFP" and x["variant"]!="full"]
fig,axs=plt.subplots(2,2,figsize=(10.4,12.7),layout="constrained")
for col,ds in enumerate(["EFP","CHO"]):
 base=next(x for x in abl if x["dataset"]==ds and x["variant"]=="full")
 for row,metric in enumerate(["mse","mae"]):
  a=axs[row,col];vals=np.array([100*(float(next(x[metric] for x in abl if x["dataset"]==ds and x["variant"]==v))/float(base[metric])-1) for v in variants])
  yy=np.arange(len(variants));a.barh(yy,vals,color=[C[0] if z>=0 else C[1] for z in vals],height=.7)
  a.axvline(0,color=C[2],ls="--",lw=1.1);a.set(yticks=yy,yticklabels=[v.replace("_"," ") for v in variants],xlabel="Change from full model(%)",title=ds+" / "+metric.upper());a.invert_yaxis();a.tick_params(axis="y",labelsize=8.5);a.grid(axis="x",alpha=.14)
save(fig,"LHC_Figure_05");link("GEMP_Figure_05",D/"LHC_Figure_05.png")
# Actual trained parameter sweeps.
sensitivity=read(ROOT/"test_results/current_bound_supplement/sensitivity_results.csv")
fig,axs=plt.subplots(2,3,figsize=(8.8,5.8),layout="constrained")
for col,(param,title,prefix) in enumerate([("alpha","EMA coefficient","alpha_"),("fusion_heads","Fusion heads","fusion_heads_"),("expert_width","Expert width","expert_width_")]):
 for t,color,marker in [("EFP_long_sub2",C[0],"o"),("CHO",C[1],"s")]:
  base=next(x for x in sensitivity if x["dataset"]==t and x["variant"]=="full")
  q=[x for x in sensitivity if x["dataset"]==t and (x["variant"]=="full" or x["variant"].startswith(prefix))]
  q=sorted(q,key=lambda x:float(x[param]))
  for row,metric in enumerate(["mse","mae"]):
   a=axs[row,col];a.plot([float(x[param]) for x in q],[100*(float(x[metric])/float(base[metric])-1) for x in q],marker=marker,color=color,label=DISPLAY[t]);a.axhline(0,color="#777777",lw=.6);a.set(xlabel=title,ylabel="Change in "+metric.upper()+"(%)");a.grid(alpha=.15)
   if param!="alpha":a.set_xscale("log",base=2);a.set_xticks(sorted(set(float(x[param]) for x in q)),[str(int(v)) for v in sorted(set(float(x[param]) for x in q))])
fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=2,frameon=False);save(fig,"LHC_Figure_07");link("GEMP_Figure_14",D/"LHC_Figure_07.png")
look=read(R/"lookback/results.csv")
fig,axs=plt.subplots(2,2,figsize=(7.8,5.7),layout="constrained")
for col,t in enumerate(["EFP_long_sub2","CHO"]):
 for model,color,marker in [("VarBridge",C[2],"D"),("DLinear","#6c777c","o"),("PatchTST",C[0],"s"),("iTransformer",C[4],"^")]:
  q=sorted([x for x in look if x["dataset"]==t and x["model"]==model],key=lambda x:int(x["lookback"]))
  for row,metric in enumerate(["mse","mae"]):
   a=axs[row,col];a.plot([int(x["lookback"]) for x in q],[float(x[metric]) for x in q],color=color,marker=marker,label=model);a.set(xlabel="History(h)",ylabel=metric.upper(),title=DISPLAY[t],xticks=[24,48,72,96]);a.grid(alpha=.15)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=4,frameon=False);save(fig,"LHC_Figure_08");link("GEMP_Figure_13",D/"LHC_Figure_08.png")
# Read original saved diagnostic cohorts; no reselection by forecasting error.
data={};clean={}
for t in ["EFP_long_sub2","CHO"]:
 base=ROOT/"test_results/current_bound_supplement"/t
 data[t]=torch.load(base/"samples.pt",map_location="cpu");clean[t]=torch.load(base/"clean_analysis.pt",map_location="cpu")
def trajectory(t):
 x=data[t];bid=x["bids"].numpy()[0];ix=np.flatnonzero(x["bids"].numpy()==bid);age=np.array(x["age_end"],float);ix=ix[np.argsort(age[ix])]
 return ix,age[ix],int(bid)
# Dynamic actual channel/general-context attention at first/middle/last observed windows.
fig,axs=plt.subplots(4,3,figsize=(8.2,9),layout="constrained")
for block,t in enumerate(["EFP_long_sub2","CHO"]):
 ix,age,bid=trajectory(t);sel=[ix[0],ix[len(ix)//2],ix[-1]]
 for row,(key,labels,title) in enumerate([("channel_attention_weights",SYMBOL[t],"Sensor attention"),("cross_attention_weights",["Window","Stage","Operation"],"General-text attention")]):
  for col,i in enumerate(sel):
   a=axs[2*block+row,col];m=clean[t]["trace"][key][int(i)].numpy()
   im=a.imshow(m,cmap="BuGn",vmin=0,vmax=1,aspect="auto");a.set(xticks=range(len(labels)),xticklabels=labels,yticks=range(5),yticklabels=SYMBOL[t],title=DISPLAY[t]+f" / age{float(data[t]['age_end'][int(i)]):.0f}h")
   if col==0:a.set_ylabel(title)
   for j in range(5):
    for k in range(len(labels)):a.text(k,j,f"{m[j,k]:.2f}",ha="center",va="center",fontsize=8.5,color="white" if m[j,k]>.6 else "#253d39")
fig.colorbar(im,ax=axs.ravel().tolist(),label="Actual attention weight",fraction=.02,pad=.02)
save(fig,"LHC_Figure_10")
# All five targets along the predetermined first held-out trajectory.
fig,axs=plt.subplots(5,2,figsize=(7.8,8.1),sharey=True,layout="constrained")
for col,t in enumerate(["EFP_long_sub2","CHO"]):
 ix,age,bid=trajectory(t);w=clean[t]["trace"]["tri_router"][ix].mean(1).numpy()
 for j in range(5):
  a=axs[j,col]
  for k,label in enumerate(["Backbone","Numerical","Corrected backbone"]):a.plot(age,w[:,j,k],color=C[k],label=label,lw=1.3)
  a.set(ylim=(-.02,1.02),ylabel="Weight",title=DISPLAY[t]+" / "+SYMBOL[t][j]);a.grid(alpha=.13)
  if j==4:a.set_xlabel("Observed batch age(h)")
fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=3,frameon=False);save(fig,"GEMP_Figure_08")
# More specific alignment: measured signals and first-target fusion use separate axes.
fig,axs=plt.subplots(2,2,figsize=(8.1,5.7),layout="constrained")
for row,t in enumerate(["EFP_long_sub2","CHO"]):
 ix,age,bid=trajectory(t);x=data[t]["x"][ix,-1,:].numpy();w=clean[t]["trace"]["tri_router"][ix,:,0,:].mean(1).numpy()
 for j in range(5):axs[row,0].plot(age,x[:,j],color=C[j],label=SYMBOL[t][j],lw=1.2)
 for k,label in enumerate(["Backbone","Numerical","Corrected backbone"]):axs[row,1].plot(age,w[:,k],color=C[k],label=label,lw=1.4)
 axs[row,0].set(title=DISPLAY[t]+" / observed sensors",ylabel="Train-standardized value",xlabel="Observed batch age(h)")
 axs[row,1].set(title=DISPLAY[t]+" / fusion for "+SYMBOL[t][0],ylabel="Weight",xlabel="Observed batch age(h)",ylim=(-.02,1.02))
 axs[row,0].legend(ncol=5,fontsize=8,loc="lower center",bbox_to_anchor=(.5,1.0),frameon=False);axs[row,0].set_title(DISPLAY[t]+" / observed sensors",pad=25)
 for a in axs[row]:a.grid(alpha=.13)
fig.legend(*axs[0,1].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=3,frameon=False);save(fig,"GEMP_Figure_16")
# Match scatter entries and marginal densities for every one of the five targets.
fig=plt.figure(figsize=(8.4,15.2),layout="constrained");subs=fig.subfigures(5,2)
for col,t in enumerate(["EFP_long_sub2","CHO"]):
 numerical="Crossformer" if t!="CHO" else "PatchTST";base=ROOT/"test_results/current_bound_supplement"/t
 predictions=[clean[t]["prediction"][:,:24],torch.load(base/(numerical+"_benchmark_predictions.pt"),map_location="cpu")["prediction"][:,:24],torch.load(base/"T3Time_benchmark_predictions.pt",map_location="cpu")["prediction"][:,:24]]
 truth=data[t]["y"][:,:24]
 for j in range(5):
  gs=subs[j,col].add_gridspec(2,2,height_ratios=[1,4],width_ratios=[4,1],hspace=.03,wspace=.03);top=subs[j,col].add_subplot(gs[0,0]);a=subs[j,col].add_subplot(gs[1,0]);side=subs[j,col].add_subplot(gs[1,1])
  y=truth[:,:,j].flatten().numpy();ix=np.unique(np.linspace(0,len(y)-1,min(3000,len(y))).astype(int));y=y[ix]
  pp=[p[:,:,j].flatten().numpy()[ix] for p in predictions];lo=min(z.min() for z in [y]+pp);hi=max(z.max() for z in [y]+pp);grid=np.linspace(lo,hi,160)
  top.plot(grid,gaussian_kde(y)(grid),color="#444444",lw=1);side.plot(gaussian_kde(y)(grid),grid,color="#444444",lw=1)
  for k,(p,label) in enumerate(zip(pp,["VarBridge",numerical,"T3Time"])):
   a.scatter(y,p,s=2,alpha=.2,color=[C[2],C[0],C[3]][k],label=label);side.plot(gaussian_kde(p)(grid),grid,color=[C[2],C[0],C[3]][k],ls=["-","--",":"][k],lw=1)
  a.plot([lo,hi],[lo,hi],ls="--",color="#777777",lw=.8);a.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Observed z-score",ylabel="Predicted z-score")
  top.set_title(DISPLAY[t]+" / "+SYMBOL[t][j]);top.axis("off");side.axis("off");a.tick_params(labelsize=8)
  if j==0 and col==0:joint_handles=a.get_legend_handles_labels()[0]
fig.legend(joint_handles,["VarBridge","Crossformer(EFP) / PatchTST(CHO)","T3Time"],loc="upper center",bbox_to_anchor=(.5,-.015),ncol=3,frameon=False);save(fig,"LHC_Figure_06");link("GEMP_Figure_07",D/"LHC_Figure_06.png")
# Aggregate every training snapshot once; final means epoch100, not selected-checkpoint time.
conditions=["own_target","group_1","noisy_text","contradictory"];cache={};probe_rows=[]
for t in ["EFP_long_sub2","CHO"]:
 for v in conditions:
  matrices=[];ent=[]
  for epoch in range(101):
   z=torch.load(R/"traces"/f"{t}_{v}_epoch{epoch:03d}.pt",map_location="cpu")["weights"]["tri_router"]
   assert z.shape[-2:]==(5,3) and torch.isfinite(z).all()
   m=z.mean((0,1)).numpy();matrices.append(m);ent.append(float(-(z*z.clamp_min(1e-12).log()).sum(-1).mean()))
   for j in range(5):
    for k in range(3):probe_rows.append(dict(dataset=t,condition=v,epoch=epoch,variable=FEATURE[t][j],candidate=k,weight=float(m[j,k]),mean_query_entropy=ent[-1]))
  cache[t,v]=(np.array(matrices),np.array(ent))
with (D/"training_probe_routing.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=list(probe_rows[0]));w.writeheader();w.writerows(probe_rows)
fig,axs=plt.subplots(4,4,figsize=(8.5,8.5),layout="constrained")
for row,v in enumerate(conditions):
 for taskcol,t in enumerate(["EFP_long_sub2","CHO"]):
  for stage,epoch in enumerate([0,100]):
   a=axs[row,2*taskcol+stage];m=cache[t,v][0][epoch]
   im=a.imshow(m,cmap="BuGn",vmin=0,vmax=1,aspect="auto");a.set(xticks=range(3),xticklabels=["Base","Num","Corr"],yticks=range(5),yticklabels=SYMBOL[t],title=DISPLAY[t]+f" / epoch{epoch}")
   if taskcol==0 and stage==0:a.set_ylabel(v.replace("_"," "))
fig.colorbar(im,ax=axs.ravel().tolist(),label="Mean final-router weight",fraction=.018,pad=.02)
save(fig,"LHC_Figure_03")
fig,axs=plt.subplots(2,2,figsize=(7.8,5.5),layout="constrained")
for col,t in enumerate(["EFP_long_sub2","CHO"]):
 for v,color,label in zip(conditions,[C[1],C[4],C[3],C[2]],["Own target","Global pooled","Noisy embedding","Contradictory"]):
  m,e=cache[t,v];axs[0,col].plot(range(101),m[:,:,2].mean(1),color=color,label=label);axs[1,col].plot(range(101),e,color=color,label=label)
 axs[0,col].set(title=DISPLAY[t],xlabel="Training epoch",ylabel="Corrected-backbone share",ylim=(0,1));axs[1,col].set(xlabel="Training epoch",ylabel="Mean routing entropy",ylim=(0,math.log(3)))
 for a in axs[:,col]:a.grid(alpha=.14)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=2,frameon=False);save(fig,"GEMP_Figure_09")
for v,name in [("group_1","GEMP_Figure_10"),("contradictory","GEMP_Figure_11")]:
 fig,axs=plt.subplots(5,2,figsize=(7.8,8.1),layout="constrained")
 for col,t in enumerate(["EFP_long_sub2","CHO"]):
  m,_=cache[t,v]
  for j in range(5):
   a=axs[j,col]
   for k,label in enumerate(["Backbone","Numerical","Corrected backbone"]):a.plot(range(101),m[:,j,k],color=C[k],label=label,lw=1.2)
   a.set(title=DISPLAY[t]+" / "+SYMBOL[t][j],ylim=(-.02,1.02),ylabel="Weight")
   if j==4:a.set_xlabel("Training epoch")
   a.grid(alpha=.13)
 fig.legend(*axs[0,0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=3,frameon=False);save(fig,name)
fig,axs=plt.subplots(6,2,figsize=(7.8,9.0),layout="constrained")
for block,t in enumerate(["EFP_long_sub2","CHO"]):
 for k,label in enumerate(["Backbone","Numerical","Corrected backbone"]):
  for col,v in enumerate(["own_target","contradictory"]):
   a=axs[3*block+k,col];m=cache[t,v][0][:,:,k].T;im=a.imshow(m,origin="lower",aspect="auto",cmap="BuGn",vmin=0,vmax=1,extent=[-.5,100.5,-.5,4.5]);a.set(yticks=range(5),yticklabels=SYMBOL[t],xlabel="Training epoch",title=DISPLAY[t]+" / "+("Correct" if col==0 else "Contradictory"))
   if col==0:a.set_ylabel(label)
fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.018,pad=.02,label="Mean final-router weight");save(fig,"GEMP_Figure_12")
# Group-count experiment is an existing complete100-epoch input-control comparison.
controls=read(R/"context_control_results.csv")
fig,axs=plt.subplots(2,2,figsize=(7.8,5.5),layout="constrained")
for col,t in enumerate(["EFP_long_sub2","CHO"]):
 for row,metric in enumerate(["mse","mae"]):
  a=axs[row,col];q=[next(x for x in controls if x["dataset"]==t and x["variant"]==v) for v in ["group_1","group_2","group_3","group_4","own_target"]]
  vals=[float(x[metric]) for x in q];a.plot(range(1,6),vals,color=C[col],marker="o");a.scatter([5],[vals[-1]],marker="*",s=110,color=C[2],zorder=4);a.axhline(vals[-1],color=C[2],ls="--",lw=.8);a.set(xticks=range(1,6),xlabel="Conditioning groups",ylabel=metric.upper(),title=DISPLAY[t]);a.grid(alpha=.14)
save(fig,"LHC_Figure_12")

# Recorded epoch walltime is separate from controlled inference and keeps actual budgets.
eff_sources=read(R/"budget_efficiency/sources.csv");cost=[]
for x in eff_sources:
 text=Path(x["log"]).read_text();pairs=[(int(a),float(b)) for a,b in __import__("re").findall(r"Epoch:\s*(\d+).*?seconds:\s*([\d.eE+-]+)",text)]
 assert [e for e,_ in pairs]==list(range(1,int(x["epochs"])+1)),x["log"]
 backbone_seconds=0.
 if x["model"]=="VarBridge":
  bp=ROOT/"test_results/cho/Backbone_h24_seed2021_full.log";bt=bp.read_text()
  bb=[(int(a),float(b)) for a,b in __import__("re").findall(r"Epoch:\s*(\d+).*?seconds:\s*([\d.eE+-]+)",bt)]
  assert [e for e,_ in bb]==list(range(1,101));backbone_seconds=sum(v for _,v in bb)
 cost.append(dict(model=x["model"],adaptation_or_baseline_epochs=int(x["epochs"]),recorded_main_epoch_seconds=sum(v for _,v in pairs),recorded_backbone_epoch_seconds=backbone_seconds,recorded_total_epoch_seconds=sum(v for _,v in pairs)+backbone_seconds,physical_ranks=6,interpretation="Observed log epoch walltime under original shared-server runs, actual unequal budgets; excludes text/API/cache preparation"))
with (D/"recorded_training_cost.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=list(cost[0]));w.writeheader();w.writerows(cost)
fig,ax=plt.subplots(figsize=(6.6,3.7),layout="constrained")
xx=np.arange(len(cost));a=np.array([z["recorded_backbone_epoch_seconds"]/3600 for z in cost]);b=np.array([z["recorded_main_epoch_seconds"]/3600 for z in cost])
ax.bar(xx,a,color=C[4],label="Frozen-backbone preparation (100 epochs)");ax.bar(xx,b,bottom=a,color=C[0],label="Baseline 10 / VarBridge adaptation 100")
ax.set(xticks=xx,xticklabels=[x["model"] for x in cost],ylabel="Recorded epoch walltime(h)",title="Training cost under the recorded 100/10 budgets");ax.grid(axis="y",alpha=.15)
fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.5,-.015),ncol=1,frameon=False,fontsize=9)
save(fig,"LHC_Figure_11")
wins=[]
for model in order:
 row=dict(model=model)
 for t in TASKS:
  for metric in ["mse","mae"]:
   row[t+"_"+metric+"_wins"]=sum(float(lookup[(t,model,str(h),metric)]["mean"])==min(float(lookup[(t,m,str(h),metric)]["mean"]) for m in order) for h in [6,12,24,48])
 wins.append(row)
with (D/"main_wins.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=list(wins[0]));w.writeheader();w.writerows(wins)
drawtable("LHC_Table_IV_wins",["Model"]+[DISPLAY[t]+"\nMSE/MAE wins" for t in TASKS],[[x["model"]]+[str(x[t+"_mse_wins"])+"/"+str(x[t+"_mae_wins"]) for t in TASKS] for x in wins],"First-place counts across four horizons per task",width=8,fontsize=10,row_height=.3)


# Dataset composition plus empirical distributions replaces unverified mechanistic arrows.
fig=plt.figure(figsize=(10.5,10.2),layout="constrained");grid=fig.add_gridspec(2,1,height_ratios=[1.3,5])
a=fig.add_subplot(grid[0]);keys=[]
for t in TASKS:
 for f,symbol in zip(FEATURE[t],SYMBOL[t]):
  key=("CHO" if t=="CHO" else "EFP",f,symbol)
  if key not in keys:keys.append(key)
m=np.array([[int(("CHO" if t=="CHO" else "EFP")==p and f in FEATURE[t]) for p,f,_ in keys] for t in TASKS])
a.imshow(m,cmap="Greys",vmin=0,vmax=1,aspect="auto")
a.set(yticks=range(4),yticklabels=[DISPLAY[t] for t in TASKS],xticks=range(len(keys)),xticklabels=[p+"\n"+s for p,_,s in keys],title="Measured sensor composition; EFP subsets are variable views")
for i in range(4):
 for j in range(len(keys)):
  if m[i,j]:a.text(j,i,"●",ha="center",va="center",color="white",fontsize=10)
a.tick_params(axis="x",labelsize=8.5)
b=fig.add_subplot(grid[1]);b.imshow(plt.imread(D/"LHC_Figure_02.png"));b.axis("off")
save(fig,"GEMP_Figure_02")

print("REFERENCE_RENDER_CORE_COMPLETE",flush=True)
