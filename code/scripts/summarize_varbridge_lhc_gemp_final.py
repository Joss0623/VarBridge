"""Summarize only fully audited supplement outputs, without manuscript edits."""
from pathlib import Path
import json,csv,statistics,math,collections
R=Path("/home/home_new/qsmx/pycodes/GreyFerm/test_results/varbridge_lhc_gemp");A=R/"api"
assert (R/"FINAL_AUDIT_PASS.json").exists()
receipt=[json.loads(x) for x in (A/"receipts.jsonl").read_text().splitlines()]
success=[x for x in receipt if x["status"]=="received"]
keys=[(x["model"],x["dataset"],x["prompt_version"],x["trial"]) for x in success]
assert len(success)==120 and len(set(keys))==120
assert all(sum(x["dataset"]==d and x["name"]==n and x["prompt_version"]==v for x in success)==5 for d in ["EFP_long_sub2","CHO"] for n in {x["name"] for x in success} for v in [0,1])
spent=sum(float(x.get("usage_cost_cny",x.get("bound_cny",0))) for x in receipt)
held=sum(float(x["bound_cny"]) for x in receipt if x["status"]=="ambiguous_failure" and "usage_cost_cny" not in x)
assert spent<=5
down=list(csv.DictReader((A/"fixed_checkpoint_downstream.csv").open()))
assert len(down)==122
models=list(dict.fromkeys(x["name"] for x in success))
metrics=list(csv.DictReader((A/"ordinal_metrics_per_call.csv").open()))
def mean(seq):return statistics.mean(float(x) for x in seq)
def save(name,rows):
 with (R/name).open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
agg=[]
for task in ["EFP_long_sub2","CHO"]:
 baseline=next(x for x in down if x["dataset"]==task and x["model"]=="Original descriptions")
 for model in models:
  q=[x for x in metrics if x["dataset"]==task and x["model"]==model]
  d=[x for x in down if x["dataset"]==task and x["model"]==model]
  assert len(q)==len(d)==10
  versions={v:mean(x["mse"] for x in d if int(x["prompt_version"])==v) for v in [0,1]}
  row=dict(dataset=task,model=model,real_calls=10,prompt_versions=2,repetitions_per_prompt=5,valid_calls=sum(x["format_valid"]=="True" for x in q),score_MAE_mean=mean(x["score_MAE"] for x in q),QWK_mean=mean(x["QWK"] for x in q),NDCG_mean=mean(x["NDCG"] for x in q),latency_mean_seconds=mean(x["latency_seconds"] for x in q),downstream_mse_mean=mean(x["mse"] for x in d),downstream_mse_sample_sd=statistics.stdev(float(x["mse"]) for x in d),downstream_mae_mean=mean(x["mae"] for x in d),original_description_mse=float(baseline["mse"]),original_description_mae=float(baseline["mae"]),forecast_windows=int(baseline["windows"]),prompt0_mse=versions[0],prompt1_mse=versions[1],training_seeds=1,training_sample_sd=None,variability_source="API text variations with one fixed trained checkpoint; not training-seed SD")
  row["mse_change_percent"]=100*(row["downstream_mse_mean"]/row["original_description_mse"]-1)
  agg.append(row)
save("llm_final_summary.csv",agg)
ctx=list(csv.DictReader((R/"context_control_results.csv").open()))
look=list(csv.DictReader((R/"lookback/results.csv").open()))
lines=["# EFP与CHO补充实验汇总","",
"本次新增训练56组全部完成：16组文本/分组对照，加上40组历史窗口阶段训练。VarBridge与其新骨干阶段各100轮，新增数值基线10轮；seed均为2021。六种LLM各20次回复，总计120次，覆盖两个任务、两种措辞、每种5次。主模型、论文和已有正式结果未修改。","",
"## 1. 正确文本没有在新训练对照中形成稳定优势","",
"|条件|EFP-sub2 MSE|EFP-sub2 MAE|CHO MSE|CHO MAE|","|---|---:|---:|---:|---:|"]
labels={"own_target":"正确的逐变量描述","contradictory":"矛盾描述","noisy_text":"扰动文本嵌入","correlation_pool":"相关性加权上下文","group_1":"单组全局平均","group_2":"2组上下文","group_3":"3组上下文","group_4":"4组上下文"}
for v in labels:
 e=next(x for x in ctx if x["dataset"]=="EFP_long_sub2" and x["variant"]==v);c=next(x for x in ctx if x["dataset"]=="CHO" and x["variant"]==v)
 lines.append("|"+labels[v]+"|"+ "|".join(f"{float(x[k]):.6f}" for x,k in [(e,"mse"),(e,"mae"),(c,"mse"),(c,"mae")])+"|")
lines+=["","正确描述、矛盾描述和扰动嵌入的差距较小，而且最优条件随任务和指标变化。因此，本组结果没有体现正确机制语义的稳定独立收益。这不等价于语言条件分支无用：保留分支、替换文本与删除整个分支是不同干预。不能把结构删除造成的性能下降全部归为预训练语义知识的贡献。",
"","## 2. 分组数量没有单调规律","",
"这里的1至4组由训练集数值关系组织，改变各变量取得的平均上下文；它们与LLM评分中研究者定义的三类功能分组不是同一概念。单组池化在EFP上略差于逐变量描述，在CHO上未显示同样劣势。2至4组也没有连续改善趋势。这些实验改变的是静态机制上下文的平均/混合方式，没有新增分组网络或改变VarBridge的变量交互结构。它们可用于检验输入组织方式，不能直接证明某种分组模块普遍优越。",
"","## 3. 历史越长并不越好","",
"|历史长度/小时|EFP-sub2 MSE|EFP-sub2 MAE|CHO MSE|CHO MAE|","|---|---:|---:|---:|---:|"]
for L in [24,48,72,96]:
 e=next(x for x in look if x["dataset"]=="EFP_long_sub2" and x["model"]=="VarBridge" and int(x["lookback"])==L);c=next(x for x in look if x["dataset"]=="CHO" and x["model"]=="VarBridge" and int(x["lookback"])==L)
 lines.append("|"+str(L)+"|"+"|".join(f"{float(x[k]):.6f}" for x,k in [(e,"mse"),(e,"mae"),(c,"mse"),(c,"mae")])+"|")
lines+=["","不同历史长度使用同一批预测起点：EFP682个测试窗口，CHO237个，预测均为24小时。这是受最长历史要求约束的单独样本队列，不能把这些数值替换主实验4431/381窗口的结果。VarBridge优于所测10轮基线，但VarBridge含新骨干100轮和融合100轮两个阶段，不能据此宣称等训练预算的结构优势。这里最清楚的发现是：增加历史长度未持续改善预测。",
"","## 4. 六种LLM的定义评分与实际预测","",
"下表QWK衡量与研究者预先制定规则的一致性，不是与独立专家真值的一致性。预测数值是在同一固定100轮检查点上替换机制描述，EFP256个、CHO96个固定窗口；没有为每个LLM重新训练。MSE为10份生成描述对应结果的平均值，波动来自API文字而不是多seed训练。",
"","|LLM|EFP QWK|CHO QWK|EFP MSE|CHO MSE|","|---|---:|---:|---:|---:|"]
for model in models:
 e=next(x for x in agg if x["dataset"]=="EFP_long_sub2" and x["model"]==model);c=next(x for x in agg if x["dataset"]=="CHO" and x["model"]==model)
 lines.append(f"|{model}|{e['QWK_mean']:.3f}|{c['QWK_mean']:.3f}|{e['downstream_mse_mean']:.6f}|{c['downstream_mse_mean']:.6f}|")
for task in ["EFP_long_sub2","CHO"]:
 b=next(x for x in down if x["dataset"]==task and x["model"]=="Original descriptions")
 lines.append(f"\n原始描述在{task}这批固定窗口上的MSE/MAE为{float(b['mse']):.6f}/{float(b['mae']):.6f}。")
lines+=["","这些API任务给出了变量定义，请求LLM评分并改写描述。因此，它们检验的是对既有定义的加工与措辞适配，不是从空白条件发现新的发酵机理。不同LLM的评分一致性不应直接解释为预测贡献；应同时看固定检查点误差、两种措辞差异和五次回复变化。正式结果不按测试分数挑选LLM或提示后回填。",
"","## 5. 对现有证据的判断","",
"继续保留原版VarBridge。当前证据可以支持对变量条件、跨传感器数值交互和预测融合进行分析；对“正确机理语义提供稳定增益”“更强LLM必然更好”则没有充分支持。旧消融中删除条件分支造成的下降，与本次文本替换实验不是同一个问题，应分别解释。数据相关图、注意力图和路由图用于展示模型行为，不能单独证明生物过程因果关系。",
"","## 6. 费用与核验","",
f"本轮累计已知费用加未返回超时请求的保守预留上限为{spent:.6f}元，其中超时预留{held:.6f}元，总额低于5元；没有充值。失败请求不计为120个有效比较条件，但计入可能费用。原始回复、失败记录、唯一请求组合、训练曲线、检查点哈希与正式文件保护均保存在服务器。",
"","主要文件：context_control_results.csv、lookback/results.csv、llm_final_summary.csv、api/ordinal_metrics_per_call.csv、api/fixed_checkpoint_downstream.csv、TRAINING_AUDIT_PASS.json、FINAL_AUDIT_PASS.json。"]
(R/"supplement_analysis_zh.md").write_text("\n".join(lines)+"\n")
audit=dict(successful_calls=120,unique_slots=120,downstream_rows=122,known_and_reserved_cost_upper_cny=spent,unresolved_timeout_reserve_cny=held,cap_cny=5,no_recharge=True,training_fits=56,main_model_unchanged=True,manuscript_unchanged=True)
(R/"SUMMARY_AUDIT_PASS.json").write_text(json.dumps(audit,indent=2))
print(json.dumps(audit))
