import os
import json
import math
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLogger:
    def __init__(self, log_dir="logs", run_name=None):
        self.log_dir = log_dir
        self.run_folder = run_name if run_name is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join(self.log_dir, self.run_folder)
        os.makedirs(self.save_path, exist_ok=True)
        self.plots_path = os.path.join(self.save_path, "plots")
        os.makedirs(self.plots_path, exist_ok=True)
        self.vis_path = os.path.join(self.save_path, "vis")
        os.makedirs(self.vis_path, exist_ok=True)
        self.saved_model_path = os.path.join(self.save_path, "saved_models")
        os.makedirs(self.saved_model_path, exist_ok=True)
        
        self.jsonl_path = os.path.join(self.save_path, "log.jsonl")
        self.training_summary_path = os.path.join(self.save_path, "training_summary.json")
        self.trainning_summary_path = os.path.join(self.save_path, "trainning_summary.json")
        self.current_step = 0
        self.history = defaultdict(list)  # 用於暫存數據以便繪圖
        self.best_metrics = {}
        self.save_training_summary()

    @staticmethod
    def _to_serializable_float(value):
        if isinstance(value, bool):
            return None

        if hasattr(value, "item"):
            try:
                value = value.item()
            except (TypeError, ValueError):
                return None

        if not isinstance(value, (int, float)):
            return None

        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    def _update_best_metrics(self, metrics: dict, step: int, timestamp: str):
        updated = False

        for metric_name, raw_value in metrics.items():
            metric_value = self._to_serializable_float(raw_value)
            if metric_value is None:
                continue

            previous_best = self.best_metrics.get(metric_name)
            if previous_best is None or metric_value > previous_best["value"]:
                self.best_metrics[metric_name] = {
                    "value": metric_value,
                    "step": int(step),
                    "timestamp": timestamp,
                }
                updated = True

        return updated

    def save_training_summary(self):
        summary_data = {
            "run_name": self.run_folder,
            "log_dir": self.save_path,
            "updated_at": datetime.now().isoformat(),
            "best_metrics": self.best_metrics,
        }

        for summary_path in (self.training_summary_path, self.trainning_summary_path):
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=4, ensure_ascii=False)

    def log(self, metrics: dict, step: int = None, verbose: bool = True):
        """記錄數據，若未提供 step 則自動遞增"""
        if step is not None:
            self.current_step = step
        
        current_timestamp = datetime.now().isoformat()
        current_step = self.current_step

        # 整合數據
        log_entry = {"step": current_step, **metrics, "timestamp": current_timestamp}
        
        # 1. 輸出至終端機
        if verbose:
            nice_str = f"[Step {self.current_step}] " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            print(nice_str)

        # 2. 寫入 JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        # 3. 更新記憶體內的歷史紀錄 (供 plot 使用)
        for k, v in metrics.items():
            self.history[k].append((self.current_step, v))

        if self._update_best_metrics(metrics, step=current_step, timestamp=current_timestamp):
            self.save_training_summary()
            
        self.current_step += 1

    def save_config(self, config: dict):
        with open(os.path.join(self.save_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

    def plot(self):
        """為每個 metric 繪製圖表並存檔，回傳 {metric_name: file_path}"""
        plot_results = {}
        plot_dir = os.path.join(self.save_path, "plots")
        os.makedirs(plot_dir, exist_ok=True)

        for metric, data in self.history.items():
            steps, values = zip(*data)
            
            plt.figure(figsize=(8, 5))
            plt.plot(steps, values, label=metric)
            plt.title(f"Metric: {metric}")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            
            file_name = f"{metric}.png"
            full_path = os.path.join(plot_dir, file_name)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            plt.savefig(full_path)
            plt.close()
            
            plot_results[metric] = full_path
            
        return plot_results
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
if __name__ == "__main__":
    logger = SimpleLogger(run_name="experiment_v1")

    # 自動計步
    for i in range(5):
        logger.log({"loss": 0.5 - i*0.1, "accuracy": 0.7 + i*0.05})

    # 手動指定計步 (例如評估階段)
    logger.log({"val_loss": 0.2}, step=100)

    # 生成圖表
    charts = logger.plot()
    print(f"圖表已生成: {charts}")
