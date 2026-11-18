"""Publish concrete one-to-one LHC/GEMP artifact inventory, with truthful incomplete states."""
from pathlib import Path
import csv,json,hashlib,time
ROOT=Path("/home/home_new/qsmx/pycodes/GreyFerm");R=ROOT/"test_results/varbridge_lhc_gemp";D=R/"deliverables"
def rows(p):return list(csv.DictReader(p.open()))
descriptions={
("LHC","Table I"):("Training sensor statistics","descriptive_statistics.csv","All retained training records; EFP views remain variable combinations."),
("LHC","Table II"):("Baseline numerical/text interfaces and budgets","deliverables/baseline_inventory.csv","Existing baseline early stopping remains visible; no relabelling as exact10."),
("LHC","Table III"):("Variable definitions and own-target mechanism slots","target_bindings.csv;deliverables/researcher_functional_groups.csv","Own-target binding is model structure; functional groups are researcher assessment labels."),
("LHC","Table IV"):("Full horizon-wise three-seed main table","deliverables/main_three_seed_raw.csv;deliverables/main_three_seed_statistics.csv","624 current formal results;416 rounded mean/SD cells match the canonical manuscript."),
("LHC","Table V"):("Three-seed95% intervals and exploratory paired tests","deliverables/main_three_seed_statistics.csv;deliverables/exploratory_seed_pair_tests.csv","Actual seeds2021/2022/2023; n3; Holm correction over384 comparisons; budgets remain unequal."),
("LHC","Table VI"):("LLM modal scoring and API latency, two wordings","deliverables/llm_modal_tables.csv;api/receipts.jsonl","Researcher rubric; three complete replies per wording instead of the planned five."),
("GEMP","Table I"):("Training sensor statistics","descriptive_statistics.csv","Same evidence as LHC Table I."),
("GEMP","Table II"):("Baseline numerical/text interfaces and budgets","deliverables/baseline_inventory.csv","Same evidence as LHC Table II."),
("GEMP","Table III"):("Full horizon-wise three-seed main table","deliverables/main_three_seed_raw.csv;deliverables/main_wins.csv","Same formal results as LHC Table IV; raw current main scores unchanged."),
("GEMP","Table IV"):("LLM partition/score consistency under two wordings","deliverables/llm_modal_tables.csv;api/receipts.jsonl","Modal ratings and modal assignments across three replies; researcher reference, not expert truth."),
("LHC","Figure 1"):("Original VarBridge architecture","figures/varbridge_full_architecture.png","Approved original architecture; no imported hypergraph or candidate model."),
("LHC","Figure 2"):("Four-task training-value distribution gallery","descriptive_statistics.csv","Complete original value ranges; retained outliers; hourly observations."),
("LHC","Figure 3"):("Initial-to-final routing under four context controls","deliverables/training_probe_routing.csv;traces/","Final-router probabilities at epoch0/100, not hypergraph weights; fixed training probes."),
("LHC","Figure 4"):("Training-only within-batch sensor correlations","../varbridge_timebase_analysis/data_relationships.csv","Numerical Pearson association; EFP views are not physiological stages."),
("LHC","Figure 5"):("All24 ablation effects on MSE and MAE","existing_ablation_aggregates.csv","Single seed2021; EFP12 equal cells, CHO four horizons; improved variants remain visible."),
("LHC","Figure 6"):("All-five-target scatter and marginal KDE","../current_bound_supplement/EFP_long_sub2/clean_analysis.pt;../current_bound_supplement/CHO/clean_analysis.pt","Matched prespecified diagnostic windows; at most3000 evenly spaced forecast entries per target."),
("LHC","Figure 7"):("EMA, attention-head and expert-width sweeps","../current_bound_supplement/sensitivity_results.csv","Completed100-epoch fits; default-relative errors; no single-seed SD."),
("LHC","Figure 8"):("Lookback sensitivity on common forecast origins","lookback/results.csv","L24/48/72/96h; H24; VarBridge backbone100+adaptation100, numerical baselines10."),
("LHC","Figure 9"):("Actual LLM response, assessment and variable binding","api/receipts.jsonl;api/researcher_reference.json","First chronological valid response; only descriptions enter VarBridge, scores are assessment-only."),
("LHC","Figure 10"):("Actual dynamic sensor and general-text attention","../current_bound_supplement/EFP_long_sub2/clean_analysis.pt;../current_bound_supplement/CHO/clean_analysis.pt","First/middle/last windows of first held-out trajectory; not single-key mechanism retrieval."),
("LHC","Figure 11"):("Recorded training cost under actual100/10 budgets","deliverables/recorded_training_cost.csv","Original six-rank epoch walltimes; backbone preparation included; observational shared-server cost."),
("LHC","Figure 12"):("Target-wise versus pooled conditioning groups","context_control_results.csv","Groups1–4 from training numeric relations; G5 is own-target; no new hypergraph or KNN architecture."),
("GEMP","Figure 1"):("Original VarBridge architecture","figures/varbridge_full_architecture.png","Same approved original-model diagram as LHC Figure1."),
("GEMP","Figure 2"):("Sensor composition and empirical distributions","target_bindings.csv;descriptive_statistics.csv","Sensor inclusion replaces unverified mechanistic arrows; EFP views share original batches."),
("GEMP","Figure 3"):("Training-only lagged predictive associations","training_lagged_associations.csv","Within-trajectory lag2 restricted/full regression; log(1+F); predictive relation, not causal truth."),
("GEMP","Figure 4"):("Training-only within-batch sensor correlations","../varbridge_timebase_analysis/data_relationships.csv","Same empirical correlation evidence as LHC Figure4."),
("GEMP","Figure 5"):("All24 original-model ablations","existing_ablation_aggregates.csv","All variants retained, including lower-error alternatives."),
("GEMP","Figure 6"):("100/10-checkpoint cached inference, parameters and GPU memory","budget_efficiency/results.csv","Existing checkpoints; same quiet GPU, FP32/batch1; native prompt-input embedding cache retains full GPT-2."),
("GEMP","Figure 7"):("All-five-target scatter and marginal KDE","../current_bound_supplement/EFP_long_sub2/clean_analysis.pt;../current_bound_supplement/CHO/clean_analysis.pt","Same matched diagnostic evidence as LHC Figure6."),
("GEMP","Figure 8"):("Five-target fusion trajectories along observed batch age","routing_diagnostics.csv;../current_bound_supplement/EFP_long_sub2/samples.pt;../current_bound_supplement/CHO/samples.pt","Actual first held-out trajectory; H24-averaged three-candidate probabilities for each sensor."),
("GEMP","Figure 9"):("Context-intervention routing adaptation and entropy","deliverables/training_probe_routing.csv","VarBridge has no trust lambda: show its actual corrected-candidate share and mean query entropy."),
("GEMP","Figure 10"):("Training trajectories under globally pooled context","deliverables/training_probe_routing.csv","Global text pooling, not an invented CI/CD expert pair."),
("GEMP","Figure 11"):("Training trajectories under contradictory descriptions","deliverables/training_probe_routing.csv","Actual contradictory-text control; original architecture and numerical inputs retained."),
("GEMP","Figure 12"):("Correct-versus-contradictory routing-by-epoch heatmaps","deliverables/training_probe_routing.csv","All five targets and three final candidates; epoch0–100; last-epoch states, not selected-checkpoint claims."),
("GEMP","Figure 13"):("Lookback sensitivity on common forecast origins","lookback/results.csv","Same fully completed40 stages as LHC Figure8."),
("GEMP","Figure 14"):("Actual VarBridge parameter sensitivity","../current_bound_supplement/sensitivity_results.csv","EMA, head count and expert width replace nonexistent prior-strength Smax."),
("GEMP","Figure 15"):("Actual definition-processing and binding workflow","api/receipts.jsonl;api/researcher_reference.json","No expert-annotation claim; generated descriptions and researcher ordinal scoring have different roles."),
("GEMP","Figure 16"):("Sensor history and internal routing on a common time axis","../current_bound_supplement/EFP_long_sub2/samples.pt;../current_bound_supplement/CHO/samples.pt","All observed sensors plus first-target fusion on separate scales; no annotated process-stage truth.")
}
modal=rows(D/"llm_modal_tables.csv");N=min(int(x["replies"]) for x in modal)
api_rows=[json.loads(s) for s in (R/"api/receipts.jsonl").read_text().splitlines()];received=sum(x["status"]=="received" for x in api_rows)
base=rows(R/"figure_table_correspondence.csv");assert len(base)==38
result=[]
for x in base:
 p=x["paper"];item=x["item"];kind,num=item.split()
 name=p+"_"+kind+"_"+(num if kind=="Table" else "%02d"%int(num))+".png"
 asset=D/name;exists=asset.exists()
 title,evidence,note=descriptions[(p,item)]
 status="ready_adapted"
 if (p,item) in [("LHC","Table VI"),("GEMP","Table IV")] and N<5:status="partial_api_three_of_five"
 elif not exists:status="waiting_quiet_gpu" if (p,item)==("GEMP","Figure 6") else "missing"
 if (p,item)==("GEMP","Figure 6") and exists and not (R/"budget_efficiency/ALL_BUDGET_INFERENCE_COMPLETE").exists():status="waiting_quiet_gpu"
 result.append(dict(paper=p,source_item=item,counterpart=title,status=status,asset=str(asset.relative_to(R)),asset_exists=exists,asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest() if exists else "",evidence=evidence,scope_and_adaptation=note))
with (D/"correspondence.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
with (R/"figure_table_status.csv").open("w") as f:w=csv.DictWriter(f,fieldnames=list(result[0]));w.writeheader();w.writerows(result)
counts={s:sum(x["status"]==s for x in result) for s in sorted({x["status"] for x in result})}
summary=dict(source_items=38,source_tables=10,source_figures=28,counts=counts,primary_assets_present=sum(x["asset_exists"] for x in result),new_training_completed=56,main_source_rows=624,main_seeds=[2021,2022,2023],supplement_training_seed=2021,canonical_main_display_cells=416,canonical_main_mismatches=len(json.loads((D/"canonical_table_consistency.json").read_text())["mismatches"]),api_received=received,api_planned=120,balanced_api_calls=24*N,no_recharge=True,canonical_manuscript_modified=False,all_planned_experiments_complete=all(x["status"]=="ready_adapted" for x in result))
(D/"completion_audit.json").write_text(json.dumps(summary,indent=2))
word={"ready_adapted":"完成（按VarBridge对应）","partial_api_three_of_five":f"{N}轮有效，五轮未满","waiting_quiet_gpu":"等待空闲GPU","missing":"缺项"}
lines=["# LHC-Net / GEMP 图表逐项对应","",
"两篇共38个条目：LHC-Net为6表12图，GEMP为4表16图。下列文件均在服务器，不改当前论文。主实验624条日志结果覆盖13模型、4任务、4个horizon和3个真实种子；416个显示单元格与当前论文一致。",
"","既有主模型100轮，既有基线保留最多10轮及其真实早停记录。新历史窗口实验中，每个数值骨干100轮、每个VarBridge适配100轮、数值基线10轮。新训练总计56组，均已核验。",
"",f"LLM目前{received}/120次有效回复，表格使用完整均衡的{24*N}次（六模型、两任务、两措辞、每种{N}次）；未充值。评分参照由研究者制定。",
"","原文的超图、信任系数、CI/CD专家对等不是VarBridge模块。本清单按研究问题对应到现有条件输入、变量注意力与三路预测融合，具体差异见correspondence.csv；不套用原模型名称与结论。",
"","|来源|VarBridge对应内容|状态|PNG|","|---|---|---|---|"]
for x in result:
 lines.append("|"+x["paper"]+" "+x["source_item"]+"|"+x["counterpart"]+"|"+word[x["status"]]+"|["+Path(x["asset"]).name+"]("+Path(x["asset"]).name+")|")
lines+=["","补充文件：LHC_Table_III_reference_groups.png、LHC_Table_IV_wins.png；全部数值源见相应CSV。main_three_seed_statistics.csv含全部模型/任务的均值、样本SD和95%种子区间；单seed消融不生成跨seedSD。","",
"统计推断采用三个不同种子，不沿用原文固定同一seed的三次重复。配对检验属于探索性分析，Holm家族为384个任务/horizon/指标/基线比较；它不消除100/10训练预算的差异。",
"","LHC Figure11为原始训练日志记录的epoch墙钟成本；新的GEMP Figure6另用同一空闲GPU做统一FP32/batch1缓存推理。历史八模型100/100的效率结果不混入该新对照。"]
(D/"index.md").write_text("\n".join(lines)+"\n")
print(json.dumps(summary),flush=True)
