import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import yaml


def merge_cfg(common: dict, exp: dict) -> dict:
    out = dict(common)
    out.update(exp)
    return out


def build_train_cmd(cfg: dict, run_dir: str):
    cmd = [
        sys.executable,
        "-u",
        "src/train.py",
        "--data_dir",
        str(cfg["data_dir"]),
        "--arch",
        str(cfg["arch"]),
        "--epochs",
        str(cfg["epochs"]),
        "--batch_size",
        str(cfg["batch_size"]),
        "--lr",
        str(cfg["lr"]),
        "--backbone",
        str(cfg.get("backbone", "convnext_tiny")),
        "--seed",
        str(cfg.get("seed", 42)),
        "--run_dir",
        run_dir,
        "--metrics_csv",
        os.path.join(run_dir, "metrics.csv"),
        "--class_stat_mode",
        str(cfg.get("class_stat_mode", "random")),
        "--class_stat_samples",
        str(cfg.get("class_stat_samples", 500)),
        "--max_fac_tries",
        str(cfg.get("max_fac_tries", 3)),
        "--dsdf_mode",
        str(cfg.get("dsdf_mode", "none")),
        "--dsdf_levels",
        str(cfg.get("dsdf_levels", "p2p3")),
        "--edge_weight",
        str(cfg.get("edge_weight", 0.0)),
        "--cons_weight",
        str(cfg.get("cons_weight", 0.0)),
        "--cons_warmup_epochs",
        str(cfg.get("cons_warmup_epochs", 10)),
        "--dice_weight",
        str(cfg.get("dice_weight", 1.0)),
        "--focal_weight",
        str(cfg.get("focal_weight", 1.0)),
        "--boundary_weight",
        str(cfg.get("boundary_weight", 0.5)),
        "--vis_every",
        str(cfg.get("vis_every", 20)),
        "--vis_samples",
        str(cfg.get("vis_samples", 4)),
    ]
    if not cfg.get("fac", True):
        cmd.append("--no_fac")
    if cfg.get("no_aug", False):
        cmd.append("--no_aug")
    if cfg.get("no_pretrained", False):
        cmd.append("--no_pretrained")
    if cfg.get("estimate_fps", False):
        cmd.append("--estimate_fps")
    return cmd


def run_command(cmd, cwd, stdout_path, stderr_path):
    with open(stdout_path, "w", encoding="utf-8") as fo, open(stderr_path, "w", encoding="utf-8") as fe:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=fo, stderr=fe)
        return proc.wait()


def read_best_metrics(metrics_csv: str):
    if not os.path.exists(metrics_csv):
        return None
    rows = []
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        return None
    rows = sorted(rows, key=lambda x: float(x["val_miou"]), reverse=True)
    return rows[0]


def read_model_meta(path: str):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_summary(path: str, rows):
    cols = [
        "name",
        "status",
        "best_epoch",
        "val_miou",
        "iou_weed",
        "weed_recall",
        "boundary_f1",
        "params_million",
        "fps",
        "run_dir",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    parser = argparse.ArgumentParser("Run full comparison/ablation experiments")
    parser.add_argument("--config", type=str, default="scripts/experiments.yaml")
    parser.add_argument("--only", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg.get("common", {})
    experiments = cfg.get("experiments", [])
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        experiments = [e for e in experiments if e.get("name") in keep]

    save_dir = common.get("save_dir", "runs")
    os.makedirs(save_dir, exist_ok=True)
    summary_rows = []

    for exp in experiments:
        name = exp["name"]
        merged = merge_cfg(common, exp)
        run_dir = os.path.join(save_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False)

        print(f"\n=== Running: {name} ===")
        if "command" in merged and merged["command"]:
            cmd = merged["command"]
            if isinstance(cmd, str):
                cmd = cmd.split()
        else:
            cmd = build_train_cmd(merged, run_dir=run_dir)

        if args.dry_run:
            print("[DRY RUN]", " ".join(cmd))
            code = 0
        else:
            code = run_command(
                cmd=cmd,
                cwd=".",
                stdout_path=os.path.join(run_dir, "train.log"),
                stderr_path=os.path.join(run_dir, "train.err"),
            )

        metrics = read_best_metrics(os.path.join(run_dir, "metrics.csv"))
        meta = read_model_meta(os.path.join(run_dir, "model_meta.yaml"))

        if args.dry_run:
            row = {
                "name": name,
                "status": "dry_run",
                "best_epoch": "",
                "val_miou": "",
                "iou_weed": "",
                "weed_recall": "",
                "boundary_f1": "",
                "params_million": "",
                "fps": "",
                "run_dir": run_dir,
            }
            print(f"[DRY RUN OK] {name}")
        elif code == 0 and metrics is not None:
            row = {
                "name": name,
                "status": "ok",
                "best_epoch": metrics.get("epoch", ""),
                "val_miou": metrics.get("val_miou", ""),
                "iou_weed": metrics.get("iou_weed", ""),
                "weed_recall": metrics.get("weed_recall", ""),
                "boundary_f1": metrics.get("boundary_f1", ""),
                "params_million": meta.get("params_million", ""),
                "fps": meta.get("fps", ""),
                "run_dir": run_dir,
            }
            print(f"[OK] {name} best mIoU={row['val_miou']}")
        else:
            row = {
                "name": name,
                "status": f"failed({code})",
                "best_epoch": "",
                "val_miou": "",
                "iou_weed": "",
                "weed_recall": "",
                "boundary_f1": "",
                "params_million": meta.get("params_million", ""),
                "fps": meta.get("fps", ""),
                "run_dir": run_dir,
            }
            print(f"[FAIL] {name} exit_code={code}")
        summary_rows.append(row)
        write_summary(os.path.join(save_dir, "summary_table.csv"), summary_rows)

    print(f"\nDone. Summary: {os.path.join(save_dir, 'summary_table.csv')}")


if __name__ == "__main__":
    main()
