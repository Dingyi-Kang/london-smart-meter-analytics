"""Equal-weight average of saved C1 predictions, logged to the C1 registry.
Usage: python -m src.lcl.combine NAME [--holdout]"""
import datetime as dt, json, sys
from pathlib import Path
from .c1 import REG, summarize, git
from .compare import load

COMBOS = {
    "E1_avg_chronos_T1": dict(members=["C1_chronos2_zero", "T1_lgbm_topdown"],
                              hypothesis="Equal average of zero-shot Chronos-2 and T1 beats Chronos alone (different families; T1 is +3-4% biased in spring, so a gain is uncertain)."),
}


def main():
    name = sys.argv[1]; h = "--holdout" in sys.argv; spec = COMBOS[name]
    if h:
        import yaml
        cfg = Path("configs/final_C1.yaml")
        if not cfg.exists() or name not in yaml.safe_load(cfg.read_text())["holdout_set"]:
            sys.exit("holdout locked / not allowed")
    P = [load(m, h) for m in spec["members"]]
    out = P[0].copy()
    for p in P[1:]:
        assert (p[["sid", "t"]].values == out[["sid", "t"]].values).all()
    out["pred"] = sum(p["pred"].values for p in P) / len(P)
    rid = f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}{'_HOLDOUT' if h else ''}"
    d = Path("experiments/runs") / rid; d.mkdir(parents=True, exist_ok=True); out.to_parquet(d / "predictions.parquet", index=False)
    m = summarize(out)
    with REG.open("a") as f:
        f.write(json.dumps({"run_id": rid, "experiment": name, "phase": "holdout" if h else "validation", "hypothesis": spec["hypothesis"],
                            "config": {"members": spec["members"]}, "status": "ok", "git": git(), "metrics": m}, default=float) + "\n")
    print(name, {k: round(v, 3) for k, v in m.items()})


if __name__ == "__main__":
    main()
