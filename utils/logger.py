import csv
from pathlib import Path

class CSVLogger:
    def __init__(self, log_dir, filename="training_log.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self.file = open(self.filepath, mode="w", newline="")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["episode", "reward", "epsilon", "steps", "loss"]
        )
        self.writer.writeheader()

    def log(self, episode, reward, epsilon, steps, loss):
        self.writer.writerow({
            "episode": episode,
            "reward": reward,
            "epsilon": epsilon,
            "steps": steps,
            "loss": loss
        })
        self.file.flush()

    def close(self):
        self.file.close()
