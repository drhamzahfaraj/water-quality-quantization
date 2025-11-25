import csv
from pathlib import Path

def init_epoch_logger(run_id):
    path = Path("experiments/logs") / f"{run_id}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch","train_loss","val_loss","val_rmse","val_mae","val_r2","lr"])
    return path

def append_epoch_log(path, epoch, train_loss, val_loss, val_rmse, val_mae, val_r2, lr):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_loss, val_loss, val_rmse, val_mae, val_r2, lr])

def append_run_overview(row_list):
    path = Path("experiments/runs_overview.csv")
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "run_id","config_file","seed","split","region","model",
                "rmse","mae","r2","power_mw","latency_ms","energy_mj",
                "model_size_mb","notes"
            ])
        writer.writerow(row_list)
