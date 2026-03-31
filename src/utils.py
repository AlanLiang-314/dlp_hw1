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
        self.epoch_log_path = os.path.join(self.save_path, "epoch_log.jsonl")
        
        self.jsonl_path = os.path.join(self.save_path, "log.jsonl")
        self.training_summary_path = os.path.join(self.save_path, "training_summary.json")
        self.trainning_summary_path = os.path.join(self.save_path, "trainning_summary.json")
        self.current_step = 0
        self.history = defaultdict(list)
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
        if step is not None:
            self.current_step = step
        
        current_timestamp = datetime.now().isoformat()
        current_step = self.current_step

        log_entry = {"step": current_step, **metrics, "timestamp": current_timestamp}
        
        if verbose:
            nice_str = f"[Step {self.current_step}] " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            print(nice_str)

        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        for k, v in metrics.items():
            self.history[k].append((self.current_step, v))

        # if self._update_best_metrics(metrics, step=current_step, timestamp=current_timestamp):
        #     self.save_training_summary()
            
        self.current_step += 1
        
    def log_epoch(self, epoch: int, metrics: dict):
        
        with open(self.epoch_log_path, "a", encoding="utf-8") as f:
            log_entry = {"epoch": epoch, **metrics, "timestamp": datetime.now().isoformat()}
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        nice_str = f"[Epoch {epoch}] " + " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
        print(nice_str)

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
    

class LossFactory:
    @staticmethod
    def get_loss(loss_type="bce", **kwargs):
        loss_type = loss_type.lower()
        if loss_type == "bce":
            return BCELossWrapper(**kwargs)
        elif loss_type == "focal":
            return FocalLoss(**kwargs)
        elif loss_type == "bce_focal":
            return BCEFocalLoss(**kwargs)
        elif loss_type == "focal_dice":
            return FocalDiceLoss(**kwargs)
        elif loss_type == "mixed":
            return MixedLoss(**kwargs)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

class BCELossWrapper(nn.Module):
    def __init__(self):
        super(BCELossWrapper, self).__init__()

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets.float())

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        dice_score = (2. * intersection + 1e-8) / (probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-8)
        return 1 - dice_score.mean()

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = probs.size(0)
        h_tv = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :]).sum()
        w_tv = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1]).sum()
        return (h_tv + w_tv) / batch_size

class MixedLoss(nn.Module):
    def __init__(self, alpha=1e-4):
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.bce_loss = BCELossWrapper()
        self.dice_loss = DiceLoss()
        self.tv_loss = TVLoss()

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        tv = self.tv_loss(logits, targets)
        return  bce + dice + self.alpha * tv
    
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha, gamma, reduction)
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return focal + dice
        
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.bce_loss = BCELossWrapper()
        self.focal_loss = FocalLoss(alpha, gamma, reduction)

    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return bce + focal

def visualize_predictions(model, dataloader, device, vis_path, step: int):
    with torch.no_grad():
        for batch_idx, (images, trimaps) in enumerate(dataloader):
            images = images.to(device)
            trimaps = trimaps.to(device).float()

            outputs = model(images)
            break  # just predict one batch for visualization

    plt.figure(figsize=(12, 6))
    for i in range(min(4, images.size(0))):
        plt.subplot(2, 4, i + 1)
        img = images[i].cpu().permute(1, 2, 0) * 0.5 + 0.5
        plt.imshow(img)
        plt.title("Input Image")
        plt.axis('off')

        plt.subplot(2, 4, i + 5)
        trimap = trimaps[i].cpu().squeeze(0)
        pred_mask = (torch.sigmoid(outputs[i]) > 0.5).float().cpu().squeeze(0)
        
        # Overlay trimap and prediction
        overlay = np.zeros((trimap.shape[0], trimap.shape[1], 3))
        overlay[trimap == 1] = [0, 255, 0]  # Green for foreground
        overlay[trimap == 0] = [255, 0, 0]   # Red for background
        overlay[pred_mask == 1] = [0, 0, 255] # Blue for predicted foreground
        
        plt.imshow(overlay.astype(np.uint8))
        plt.title("Trimap & Prediction")
        plt.axis('off')
    plt.tight_layout()
    save_path = os.path.join(vis_path, f"predictions_step_{step}.png")
    plt.savefig(save_path)
    

import torch
import numpy as np
import time
from collections import defaultdict

class MetricFactory:
    @staticmethod
    def get_metrics(metric_names, threshold=0.5):
        metrics = {}
        for name in metric_names:
            name = name.lower()
            if name == "iou":
                metrics["iou"] = IoUMetric(threshold)
            elif name == "dice":
                metrics["dice"] = DiceMetric(threshold)
            elif name == "accuracy":
                metrics["accuracy"] = AccuracyMetric(threshold)
        return metrics

class BaseMetric:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, logits, targets):
        raise NotImplementedError

class IoUMetric(BaseMetric):
    def __call__(self, logits, targets):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        union = (preds + targets).clamp(0, 1).sum(dim=(1, 2, 3))
        iou = (intersection + 1e-7) / (union + 1e-7)
        return iou.mean().item()

class DiceMetric(BaseMetric):
    def __call__(self, logits, targets):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        intersection = (preds * targets).sum(dim=(1, 2, 3))
        dice = (2. * intersection + 1e-7) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + 1e-7)
        return dice.mean().item()

class AccuracyMetric(BaseMetric):
    def __call__(self, logits, targets):
        preds = (torch.sigmoid(logits) > self.threshold).float()
        correct = (preds == targets).float().mean()
        return correct.item()

class MetricManager:
    def __init__(self, loss_type="bce", metric_names=["iou", "dice"], logger=None, threshold=0.3, total_steps=None, eta_every_log=True):
        self.criterion = LossFactory.get_loss(loss_type)
        self.metrics_dict = MetricFactory.get_metrics(metric_names, threshold)
        self.logger = logger
        self.total_steps = total_steps
        self.eta_every_log = eta_every_log
        self.start_time = time.time()
        self.logged_steps = 0

        self.reset_epoch_stats()

    @staticmethod
    def _format_seconds(seconds):
        seconds = int(max(0, seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def reset_epoch_stats(self):
        self.epoch_data = defaultdict(list)

    def _estimate_eta(self):
        self.logged_steps += 1
        elapsed = time.time() - self.start_time
        avg_step_time = elapsed / max(1, self.logged_steps)

        eta_seconds = None
        if self.total_steps is not None:
            remaining_steps = max(0, self.total_steps - self.logged_steps)
            eta_seconds = remaining_steps * avg_step_time

        return elapsed, avg_step_time, eta_seconds

    def update(self, logits, targets, step=None, prefix="train/"):
        results = {}

        loss = self.criterion(logits, targets)
        results[f"{prefix}loss"] = loss.item()

        for name, metric_fn in self.metrics_dict.items():
            val = metric_fn(logits, targets)
            results[f"{prefix}{name}"] = val

        elapsed, avg_step_time, eta_seconds = self._estimate_eta()

        if self.logger:
            if step is not None and step % 50 == 0:
                self.logger.log(results, step=step, verbose=True)
                if self.total_steps is not None:
                    print(
                        f"[ETA] {prefix.rstrip('/')} | progress: {self.logged_steps}/{self.total_steps} "
                        f"| elapsed: {self._format_seconds(elapsed)} "
                        f"| remaining: {self._format_seconds(eta_seconds)} "
                        f"| avg_step: {avg_step_time:.2f}s"
                    )

            else:
                self.logger.log(results, step=step, verbose=False)

        for k, v in results.items():
            self.epoch_data[k].append(v)

        return loss

    def get_epoch_averages(self):
        return {k: np.mean(v) for k, v in self.epoch_data.items()}
