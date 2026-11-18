"""CPU-only preflight. No optimizer, training loop, GPU, or result metrics."""
import ast, gc, hashlib, inspect, sys, textwrap
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from common_bound_supplement import make_config, seed_all
from models.KIBR_PatchTST_BoundMechanism import Model as Original
from models.VarBridge_TimeLLMInput import Model as Variant
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()
protocol=(ROOT/"test_results/varbridge_timellm_input/protocol.md").read_text()
protected={}
for line in protocol.splitlines():
    parts=line.split()
    if len(parts)==2 and len(parts[1])==64:
        protected[ROOT.parent/parts[0]]=parts[1]
assert len(protected)==4
for p,h in protected.items(): assert sha(p)==h, str(p)
official_src=(ROOT.parent/"Time-LLM/models/TimeLLM.py").read_text()
node=next(n for n in ast.parse(official_src).body if isinstance(n,ast.ClassDef) and n.name=="ReprogrammingLayer")
expected="\n".join(official_src.splitlines()[node.lineno-1:node.end_lineno])
variant_source=(ROOT/"models/VarBridge_TimeLLMInput.py").read_text()
assert expected in variant_source
# Require that the only forecast differences are the documented input changes.
a=textwrap.dedent(inspect.getsource(Original.forecast))
b=textwrap.dedent(inspect.getsource(Variant.forecast))
a=a.replace("stdev = torch.sqrt(torch.var(normed, dim=1, keepdim=True, unbiased=False) + 1e-5)",
            "stdev = torch.sqrt(torch.var(normed, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()")
a=a.replace("    residual_frozen = frozen","""    # Keep the frozen backbone forecast, replace its fusion-facing input features.
    prototypes = self.numeric_input.prototypes()
    frozen = self.numeric_input(normed, prototypes, 'raw')
    if frozen.shape != (bs * n_vars, n_patches, self.hidden_dim):
        raise ValueError('Reprogrammed patches must match the fusion interface')
    residual_frozen = frozen""")
a=a.replace("self.encoder(make_patches(seasonal))","self.numeric_input(seasonal, prototypes, 'seasonal')")
a=a.replace("self.trend_encoder(make_patches(trend))","self.numeric_input(trend, prototypes, 'trend')")
assert a.strip()==b.strip(), "Undocumented downstream forecast change"
print("PASS documented-input-only source comparison",flush=True)
for dataset in ["EFP_long_sub1","EFP_long_sub2","EFP_long_sub3","CHO"]:
    for horizon in [6,12,24]:
        cfg=make_config(dataset);cfg.pred_len=cfg.token_len=horizon
        seed_all()
        old=Original(cfg)
        state={k:v.clone() for k,v in old.state_dict().items()
               if not k.startswith(("encoder.","trend_encoder."))}
        backbone_hash=sha(cfg.numeric_ckpt)
        del old;gc.collect()
        seed_all()
        net=Variant(cfg)
        current=net.state_dict()
        assert all(k in current and torch.equal(v,current[k]) for k,v in state.items())
        del state;gc.collect()
        assert not any("llm_model" in k for k in current)
        assert net.numeric_input.mapping_layer.out_features==1000
        assert not net.numeric_input.word_embeddings.requires_grad
        x=torch.randn(1,48,5)
        text=torch.randn(1,5,8,768)
        # eval-mode gradients ensure no dropout ambiguity; no optimizer steps.
        net.eval()
        y=net(x,text,None,None)
        assert y.shape==(1,horizon,5) and torch.isfinite(y).all()
        loss=torch.nn.functional.huber_loss(y,torch.randn_like(y))
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in net.parameters())
        assert all(p.grad is None for p in net.patch_backbone.parameters())
        for name in ["mapping_layer.weight",
                     "patch_embeddings.raw.tokenConv.weight",
                     "patch_embeddings.seasonal.tokenConv.weight",
                     "patch_embeddings.trend.tokenConv.weight",
                     "bridges.raw.weight","bridges.seasonal.weight","bridges.trend.weight"]:
            p=dict(net.numeric_input.named_parameters())[name]
            assert p.grad is not None and p.grad.abs().sum()>0,name
        # Constant windows exercise normalization stability.
        with torch.no_grad():
            const=net(torch.ones_like(x)*3,text,None,None)
            assert torch.isfinite(const).all()
        assert sha(cfg.numeric_ckpt)==backbone_hash
        print("PASS",dataset,"H",horizon,"shape",tuple(y.shape),
              "trainable",sum(p.numel() for p in net.parameters() if p.requires_grad),
              "frozen_backbone_unchanged",flush=True)
        del net,current,y,loss,const;gc.collect()
for p,h in protected.items(): assert sha(p)==h,str(p)
print("VARBRIDGE_TIMELLM_INPUT_PREFLIGHT_PASS no_training CPU_only",flush=True)
