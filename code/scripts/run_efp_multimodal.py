#!/usr/bin/env python3
"""Aligned EFP evaluation for TimeCMA and T3Time."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/home_new/qsmx/pycodes")
PROJ = ROOT / "GreyFerm"

class EFPWindows(Dataset):
    def __init__(self, name, split, horizon):
        self.horizon = horizon
        raw = pd.read_csv(PROJ / "dataset" / name / f"{name}.csv")
        features = [c for c in raw if c not in ("date", "batch_id", "hh")]
        bids = raw.batch_id.to_numpy()
        bounds = np.r_[0, np.flatnonzero(bids[1:] != bids[:-1]) + 1, len(raw)]
        ordered = [bids[i] for i in bounds[:-1]]
        if len(ordered) != len(set(ordered)):
            raise ValueError("Noncontiguous EFP batch IDs")
        spans = {b: (int(s), int(e)) for b, s, e in zip(ordered, bounds[:-1], bounds[1:])}
        eligible = [b for b in ordered if spans[b][1] - spans[b][0] >= 96]
        nt, nv = int(.7 * len(eligible)), int(.1 * len(eligible))
        groups = {"train": eligible[:nt], "val": eligible[nt:nt+nv], "test": eligible[nt+nv:]}
        trainrows = np.concatenate([np.arange(*spans[b]) for b in groups["train"]])
        values = raw[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite EFP features")
        mean = values[trainrows].mean(0)
        std = values[trainrows].std(0)
        std[std == 0] = 1.
        self.values = torch.from_numpy(((values - mean) / std).astype(np.float32))
        suffix = f"_h{horizon}" if horizon in (6, 12) else ""
        cache = torch.load(PROJ / "dataset" / name / f"{name}__gpt2_window_semantic{suffix}.pt",
                           map_location="cpu", weights_only=False)
        if cache["features"] != features:
            raise ValueError("Prompt features do not match EFP features")
        self.emb = cache["window_embeddings"]
        self.lookup = {int(s): i for i, s in enumerate(cache["starts"])}
        self.starts = [i for b in groups[split] for s, e in [spans[b]]
                       for i in range(s, e-48-horizon+1)]
        if any(i not in self.lookup for i in self.starts):
            raise ValueError("A forecast window lacks a prompt embedding")
        self.mark = torch.zeros(48, 4)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        x = self.values[s:s+48]
        y = self.values[s+48:s+48+self.horizon]
        emb = self.emb[self.lookup[s]].float().T.unsqueeze(-1)
        return x, self.mark, emb, y

def evaluate(model, loader, device, train=False, optimizer=None):
    model.train(train)
    sum_sq = sum_abs = count = 0.
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, mark, emb, y in loader:
            x, mark, emb, y = (z.to(device, non_blocking=True) for z in (x, mark, emb, y))
            pred = model(x, mark, emb)
            if pred.shape != y.shape:
                raise ValueError(f"Prediction shape {pred.shape} != target {y.shape}")
            diff = pred - y
            loss = diff.square().mean()
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
            sum_sq += diff.square().sum().item()
            sum_abs += diff.abs().sum().item()
            count += diff.numel()
    return sum_sq / count, sum_abs / count

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["TimeCMA", "T3Time"], required=True)
    p.add_argument("--dataset", choices=[f"EFP_long_sub{i}" for i in (1,2,3)], required=True)
    p.add_argument("--horizon", type=int, choices=[6,12,24,48], required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=2021)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(ROOT / args.model))
    if args.model == "TimeCMA":
        from models.TimeCMA import Dual as Net
    else:
        from models.T3Time import TriModal as Net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets = {s: EFPWindows(args.dataset, s, args.horizon) for s in ("train","val","test")}
    loaders = {s: DataLoader(datasets[s], batch_size=args.batch_size, shuffle=(s=="train"),
                             num_workers=0, drop_last=False, pin_memory=True) for s in datasets}
    model = Net(device=str(device), channel=32, num_nodes=5, seq_len=48, pred_len=args.horizon,
                dropout_n=.2, d_llm=768, e_layer=1, d_layer=1, head=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    out = PROJ / "checkpoints" / "efp_multimodal" / f"{args.model}_{args.dataset}_h{args.horizon}_seed{args.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    best, stale = float("inf"), 0
    for epoch in range(1, args.epochs+1):
        tr, _ = evaluate(model, loaders["train"], device, True, optimizer)
        val, _ = evaluate(model, loaders["val"], device)
        print(f"epoch:{epoch} train_mse:{tr:.9f} val_mse:{val:.9f}", flush=True)
        if val < best:
            best, stale = val, 0
            torch.save(model.state_dict(), out)
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(out, map_location=device, weights_only=True))
    mse, mae = evaluate(model, loaders["test"], device)
    print(f"mse:{mse:.9f}, mae:{mae:.9f}", flush=True)
    print(json.dumps({"model":args.model,"dataset":args.dataset,"horizon":args.horizon,
                      "seq_len":48,"epochs":args.epochs,"patience":args.patience,
                      "train_windows":len(datasets["train"]),"val_windows":len(datasets["val"]),
                      "test_windows":len(datasets["test"]),"mse":mse,"mae":mae}), flush=True)

if __name__ == "__main__":
    main()
