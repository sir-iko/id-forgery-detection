import csv
import json
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    """Lightweight experiment tracking: a timestamped folder per run holding a
    config snapshot, a metrics.csv, and TensorBoard scalars. No external
    account or internet needed, which matters on compute nodes.
    """

    def __init__(self, output_dir, run_name, config=None):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.dir = Path(output_dir) / f"{run_name}_{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir=str(self.dir / "tb"))
        self.metrics_path = self.dir / "metrics.csv"
        if config is not None:
            with open(self.dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

    def log(self, step, **metrics):
        for k, v in metrics.items():
            self.tb.add_scalar(k, v, step)
        row = {"step": step, **metrics}
        write_header = not self.metrics_path.exists()
        with open(self.metrics_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)

    def close(self):
        self.tb.close()
