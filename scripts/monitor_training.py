import argparse
import csv
import os
import subprocess
import time


def read_last_row(csv_path: str):
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return rows[-1]


def query_gpu():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if out:
            # single GPU: "RTX..., 45, 3120, 8188, 66"
            p = [x.strip() for x in out.split("\n")[0].split(",")]
            if len(p) >= 5:
                return {
                    "name": p[0],
                    "util": p[1],
                    "mem_used": p[2],
                    "mem_total": p[3],
                    "temp": p[4],
                }
    except Exception:
        return None
    return None


def main():
    parser = argparse.ArgumentParser("Monitor training metrics in real time")
    parser.add_argument("--metrics_csv", type=str, required=True)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--show_gpu", action="store_true")
    args = parser.parse_args()

    print(f"Monitoring: {args.metrics_csv}")
    print("Press Ctrl+C to stop monitor.\n")

    last_epoch = None
    while True:
        row = read_last_row(args.metrics_csv)
        if row is not None:
            epoch = row.get("epoch", "")
            if epoch != last_epoch:
                last_epoch = epoch
                msg = (
                    f"[epoch {epoch}] "
                    f"mIoU={row.get('val_miou', '')} "
                    f"weedIoU={row.get('iou_weed', '')} "
                    f"weedRecall={row.get('weed_recall', '')} "
                    f"boundaryF1={row.get('boundary_f1', '')} "
                    f"lr={row.get('lr', '')} "
                    f"cons={row.get('cons_weight', '')}"
                )
                print(msg)
                if row.get("weed_tp") is not None:
                    print(
                        f"  weed TP/FN/FP = "
                        f"{row.get('weed_tp', '')}/{row.get('weed_fn', '')}/{row.get('weed_fp', '')}"
                    )
                print(
                    f"  FAC avg_fg={row.get('fac_avg_fg_ratio', '')} "
                    f"avg_try={row.get('fac_avg_attempts', '')} "
                    f"samples={row.get('fac_samples', '')}"
                )

        if args.show_gpu:
            g = query_gpu()
            if g is not None:
                print(
                    f"  GPU {g['name']}: util={g['util']}% "
                    f"mem={g['mem_used']}/{g['mem_total']}MB temp={g['temp']}C"
                )

        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
