import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.models import ResNet34_UNet, UNet

DATASET_DIR = "dataset"

class OxfordPetsDataset(Dataset):
    def __init__(self, split: str = "test_resnet_unet", with_mask: bool = False):
        split_to_file = {
            "train": "train.txt",
            "val": "val.txt",
            "test_unet": "test_unet.txt",
            "test_resnet_unet": "test_resnet_unet.txt",
        }
        if split not in split_to_file:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.with_mask = with_mask
        split_file = split_to_file[split]
        candidates = [
            os.path.join(DATASET_DIR, split_file),
            os.path.join(DATASET_DIR, "annotations", split_file),
        ]
        self.datalist_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        with open(self.datalist_path, "r", encoding="utf-8") as f:
            self.images = [line.strip() for line in f if line.strip()]

        self.image_path = os.path.join(DATASET_DIR, "images")
        self.trimap_path = os.path.join(DATASET_DIR, "annotations", "trimaps")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_id = self.images[idx]
        img_path = os.path.join(self.image_path, image_id + ".jpg")
        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if not self.with_mask:
            return image, image_id

        trimap_path = os.path.join(self.trimap_path, image_id + ".png")
        trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
        if trimap is None:
            raise FileNotFoundError(f"Failed to load mask: {trimap_path}")

        mask = (trimap == 1).astype(np.uint8)
        return image, mask, image_id


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return image


def make_starts(length: int, tile_size: int, stride: int) -> List[int]:
    if length <= tile_size:
        return [0]

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def overlap_tile_predict_resnet(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    tile_size: int = 256,
    overlap: int = 64,
    tile_batch_size: int = 8,
) -> torch.Tensor:
    b, c, h, w = image_tensor.shape
    if b != 1:
        raise ValueError(f"Only batch size 1 is supported, got {b}")

    if h < tile_size or w < tile_size:
        raise ValueError(f"Input too small for tile_size={tile_size}: got ({h}, {w})")

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap must be smaller than tile_size: got tile={tile_size}, overlap={overlap}")

    ys = make_starts(h, tile_size, stride)
    xs = make_starts(w, tile_size, stride)

    coords: List[Tuple[int, int]] = []
    tiles: List[torch.Tensor] = []

    for y in ys:
        for x in xs:
            tile = image_tensor[:, :, y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
            coords.append((y, x))

    tiles_tensor = torch.cat(tiles, dim=0)

    logits_accum = torch.zeros((1, 1, h, w), device=image_tensor.device, dtype=image_tensor.dtype)
    weight_accum = torch.zeros_like(logits_accum)

    for start in range(0, tiles_tensor.shape[0], tile_batch_size):
        end = min(start + tile_batch_size, tiles_tensor.shape[0])
        logits_batch = model(tiles_tensor[start:end])

        for i, (y, x) in enumerate(coords[start:end]):
            logits_accum[:, :, y : y + tile_size, x : x + tile_size] += logits_batch[i : i + 1]
            weight_accum[:, :, y : y + tile_size, x : x + tile_size] += 1.0

    logits = logits_accum / weight_accum.clamp_min(1.0)
    return logits


def predict_binary_mask_resnet(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    tile_size: int,
    overlap: int,
    tile_batch_size: int,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    orig_h, orig_w = image_np.shape[:2]

    if orig_h < tile_size or orig_w < tile_size:
        resized = cv2.resize(image_np, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        image_input = normalize_image(resized)
        image_tensor = torch.from_numpy(image_input).permute(2, 0, 1).unsqueeze(0).to(device)
        logits_resized = model(image_tensor)
        prob_resized = torch.sigmoid(logits_resized)[0, 0].cpu().numpy()
        prob_map = cv2.resize(prob_resized, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_input = normalize_image(image_np)
        image_tensor = torch.from_numpy(image_input).permute(2, 0, 1).unsqueeze(0).to(device)
        logits = overlap_tile_predict_resnet(
            model=model,
            image_tensor=image_tensor,
            tile_size=tile_size,
            overlap=overlap,
            tile_batch_size=tile_batch_size,
        )
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_mask = (prob_map > threshold).astype(np.uint8)
    return pred_mask, prob_map


def prepare_padded_image(
    image_np: np.ndarray,
    tile_size: int = 572,
    output_size: int = 388,
    stride: int = 388,
) -> Tuple[np.ndarray, Dict[str, int]]:
    h, w = image_np.shape[:2]
    margin = (tile_size - output_size) // 2
    if stride <= 0 or stride > output_size:
        raise ValueError(f"stride must be in [1, {output_size}], got {stride}")

    tiles_h = max(1, int(np.ceil((h - output_size) / stride)) + 1)
    tiles_w = max(1, int(np.ceil((w - output_size) / stride)) + 1)

    padded_h = (tiles_h - 1) * stride + tile_size
    padded_w = (tiles_w - 1) * stride + tile_size

    pad_top = margin
    pad_left = margin
    pad_bottom = padded_h - h - pad_top
    pad_right = padded_w - w - pad_left

    padded = cv2.copyMakeBorder(
        image_np,
        top=pad_top,
        bottom=pad_bottom,
        left=pad_left,
        right=pad_right,
        borderType=cv2.BORDER_REFLECT,
    )
    padded = normalize_image(padded)

    meta = {
        "orig_h": h,
        "orig_w": w,
    }
    return padded, meta


def overlap_tile_predict_unet(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    tile_size: int = 572,
    output_size: int = 388,
    stride: int = 388,
    tile_batch_size: int = 8,
    blend_window: str = "uniform",
) -> torch.Tensor:
    b, c, h, w = image_tensor.shape

    if h < tile_size or w < tile_size:
        raise ValueError(f"Input too small for tile_size={tile_size}: got ({h}, {w})")

    tiles = image_tensor.unfold(2, tile_size, stride).unfold(3, tile_size, stride)
    n_h = tiles.shape[2]
    n_w = tiles.shape[3]
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, c, tile_size, tile_size)

    preds = []
    for start in range(0, tiles.shape[0], tile_batch_size):
        end = start + tile_batch_size
        preds.append(model(tiles[start:end]))
    preds = torch.cat(preds, dim=0)

    preds = preds.view(b, n_h, n_w, 1, output_size, output_size)
    out_h = (n_h - 1) * stride + output_size
    out_w = (n_w - 1) * stride + output_size
    logits_sum = torch.zeros((b, 1, out_h, out_w), dtype=preds.dtype, device=preds.device)
    weight_sum = torch.zeros_like(logits_sum)

    if blend_window == "hann":
        win_1d = torch.hann_window(output_size, periodic=False, device=preds.device)
        win_2d = torch.outer(win_1d, win_1d).clamp_min(1e-6)
        weight = win_2d.view(1, 1, output_size, output_size)
    elif blend_window == "uniform":
        weight = torch.ones((1, 1, output_size, output_size), device=preds.device, dtype=preds.dtype)
    else:
        raise ValueError(f"Unsupported blend_window: {blend_window}")

    for i in range(n_h):
        y = i * stride
        for j in range(n_w):
            x = j * stride
            patch = preds[:, i, j]
            logits_sum[:, :, y : y + output_size, x : x + output_size] += patch * weight
            weight_sum[:, :, y : y + output_size, x : x + output_size] += weight

    return logits_sum / weight_sum.clamp_min(1e-6)


def predict_binary_mask_unet(
    model: torch.nn.Module,
    image_np: np.ndarray,
    device: torch.device,
    tile_batch_size: int,
    tile_stride: int,
    blend_window: str,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    padded_img, meta = prepare_padded_image(
        image_np,
        tile_size=572,
        output_size=388,
        stride=tile_stride,
    )

    image_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).to(device)
    logits_full = overlap_tile_predict_unet(
        model=model,
        image_tensor=image_tensor,
        tile_size=572,
        output_size=388,
        stride=tile_stride,
        tile_batch_size=tile_batch_size,
        blend_window=blend_window,
    )
    logits = logits_full[:, :, : meta["orig_h"], : meta["orig_w"]]
    prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred_mask = (prob_map > threshold).astype(np.uint8)
    return pred_mask, prob_map


def rle_encode(mask: np.ndarray) -> str:
    """Run-Length Encoding in column-major (Fortran) order."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.shape}")

    pixels = mask.astype(np.uint8).flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def calc_iou_and_dice(pred_mask: np.ndarray, target_mask: np.ndarray) -> Tuple[float, float]:
    pred = pred_mask.astype(np.uint8)
    target = target_mask.astype(np.uint8)

    intersection = np.logical_and(pred == 1, target == 1).sum()
    union = np.logical_or(pred == 1, target == 1).sum()
    pred_area = (pred == 1).sum()
    target_area = (target == 1).sum()

    iou = (intersection + 1e-7) / (union + 1e-7)
    dice = (2.0 * intersection + 1e-7) / (pred_area + target_area + 1e-7)
    return float(iou), float(dice)


def load_mask_for_image(image_id: str) -> Optional[np.ndarray]:
    trimap_path = os.path.join(DATASET_DIR, "annotations", "trimaps", image_id + ".png")
    if not os.path.exists(trimap_path):
        return None

    trimap = cv2.imread(trimap_path, cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        return None
    return (trimap == 1).astype(np.uint8)


def _strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    has_prefix = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    if not has_prefix:
        return state_dict

    return {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}


def load_model(model_path: str, device: torch.device, model_type: str) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
        if not isinstance(state_dict, dict):
            raise TypeError("Checkpoint dict does not contain a valid state_dict")

        state_dict = _strip_compile_prefix(state_dict)
        if model_type == "resnet":
            model = ResNet34_UNet(num_classes=1)
        elif model_type == "unet":
            model = UNet(in_channels=3, out_channels=1)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    model.to(device)
    model.eval()
    return model


def run_submission(args, model: torch.nn.Module, device: torch.device) -> None:
    dataset = OxfordPetsDataset(split=args.split, with_mask=False)
    total = len(dataset) if args.max_images < 0 else min(args.max_images, len(dataset))
    iou_list: List[float] = []
    dice_list: List[float] = []

    output_dir = os.path.dirname(args.output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "encoded_mask"])

        with torch.no_grad():
            for idx in range(total):
                image_np, image_id = dataset[idx]
                if args.model_type == "resnet":
                    pred_mask, _ = predict_binary_mask_resnet(
                        model=model,
                        image_np=image_np,
                        device=device,
                        tile_size=args.resnet_tile_size,
                        overlap=args.resnet_overlap,
                        tile_batch_size=args.resnet_tile_batch_size,
                        threshold=args.threshold,
                    )
                else:
                    pred_mask, _ = predict_binary_mask_unet(
                        model=model,
                        image_np=image_np,
                        device=device,
                        tile_batch_size=args.unet_tile_batch_size,
                        tile_stride=args.unet_tile_stride,
                        blend_window=args.unet_blend_window,
                        threshold=args.threshold,
                    )
                writer.writerow([image_id, rle_encode(pred_mask)])

                gt_mask = load_mask_for_image(image_id)
                if gt_mask is not None:
                    iou, dice = calc_iou_and_dice(pred_mask, gt_mask)
                    iou_list.append(iou)
                    dice_list.append(dice)

                if (idx + 1) % 20 == 0 or (idx + 1) == total:
                    print(f"{idx + 1}/{total}")

    print(f"Saved submission CSV: {os.path.abspath(args.output_csv)}")
    if iou_list:
        print(f"Mean IoU  (n={len(iou_list)}): {np.mean(iou_list):.4f}")
        print(f"Mean Dice (n={len(dice_list)}): {np.mean(dice_list):.4f}")
    else:
        print("Mean IoU/Dice skipped: no ground-truth trimaps found for this split.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=["resnet", "unet"], default="resnet")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="submission.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-images", type=int, default=-1)

    parser.add_argument("--resnet-tile-size", type=int, default=256)
    parser.add_argument("--resnet-overlap", type=int, default=64)
    parser.add_argument("--resnet-tile-batch-size", type=int, default=8)

    parser.add_argument("--unet-tile-batch-size", type=int, default=8)
    parser.add_argument("--unet-tile-stride", type=int, default=388)
    parser.add_argument(
        "--unet-blend-window",
        type=str,
        choices=["uniform", "hann"],
        default="uniform",
    )

    args = parser.parse_args()

    if args.split is None:
        args.split = "test_unet" if args.model_type == "unet" else "test_resnet_unet"

    if args.threshold is None:
        args.threshold = 0.5 if args.model_type == "unet" else 0.4

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Split: {args.split}")
    print(f"Threshold: {args.threshold}")
    print(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, device=device, model_type=args.model_type)
    run_submission(args, model, device)


if __name__ == "__main__":
    main()
