"""Choose a single PatchTST text encoder from validation MSE only."""
import json
import re
from pathlib import Path
root = Path("test_results")
candidates = {}
for text in ("gpt2", "distilgpt2", "bert", "roberta"):
    cells = []
    for ds in ("EFP_long_sub1", "EFP_long_sub2", "EFP_long_sub3"):
        for h in (6, 12, 24):
            name = (f"encoderplug100_patchtst_{text}_{ds}_seed2021.log" if h == 24
                    else f"encoderplug100_h{h}_patchtst_{text}_{ds}_seed2021.log")
            source = root / name
            body = source.read_text()
            expected = 2 if h == 24 else 1
            if len(re.findall(r"^mse:", body, re.M)) != expected or "Traceback" in body:
                raise RuntimeError(f"Incomplete encoder run: {source}")
            values = [float(x) for x in re.findall(r"Vali Loss: ([0-9.eE+-]+)", body)]
            if not values:
                raise RuntimeError(f"No validation MSE: {source}")
            cells.append({"dataset": ds, "training_horizon": h,
                          "checkpoint_validation_mse": min(values), "log": str(source)})
    candidates[text] = {"mean_validation_mse": sum(c["checkpoint_validation_mse"] for c in cells)/len(cells),
                        "cells": cells}
selected = min(candidates, key=lambda x: candidates[x]["mean_validation_mse"])
record = {"numeric_encoder": "patchtst", "selected_text_encoder": selected,
          "selection_metric": "equal-weight mean validation MSE over 3 subsets and training horizons 6,12,24",
          "horizon48": "Recursive evaluation of horizon24 checkpoint; no separate logged validation48 score. It is not double-weighted in selection.",
          "validation_metric_source": "exp_long_term_forecasting._select_criterion is MSELoss; min Vali Loss selects saved checkpoint.",
          "test_metrics_used_for_selection": False, "candidates": candidates}
out = root / "main_encoder_selection.json"
temp = out.with_suffix(".tmp")
temp.write_text(json.dumps(record, indent=2))
temp.replace(out)
print(selected)
