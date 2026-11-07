import torch
from torch.nn import functional as F

from .runtime import all_reduce_sum


class Metrics:
    def __init__(self, device, *, pixel_mode=None, image=False, threshold=0.5):
        if pixel_mode not in {None, "per_image", "entire_pixel"}:
            raise ValueError("pixel_mode must be per_image, entire_pixel or None.")
        self.pixel_mode = pixel_mode
        self.image = image
        self.threshold = threshold
        self.pixel_sums = torch.zeros(8, dtype=torch.float64, device=device)
        self.image_counts = torch.zeros(4, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, output, batch):
        if self.pixel_mode:
            truth = batch["mask"] > 0.5
            prediction = output["pred_mask"].float()
            if prediction.shape[-2:] != truth.shape[-2:]:
                prediction = F.interpolate(
                    prediction, size=truth.shape[-2:], mode="bilinear", align_corners=False
                )
            prediction = prediction > self.threshold
            if prediction.shape != truth.shape:
                raise ValueError(
                    f"Prediction/mask shapes differ: {prediction.shape} vs {truth.shape}"
                )
            dims = tuple(range(1, truth.ndim))
            tp = (prediction & truth).sum(dims).double()
            fp = (prediction & ~truth).sum(dims).double()
            fn = (~prediction & truth).sum(dims).double()
            tn = (~prediction & ~truth).sum(dims).double()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            pixel_recall = tp / (tp + fn + 1e-12)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            self.pixel_sums += torch.stack(
                (
                    f1.sum(),
                    pixel_recall.sum(),
                    accuracy.sum(),
                    tp.sum(),
                    fp.sum(),
                    fn.sum(),
                    tn.sum(),
                    tp.new_tensor(len(tp)),
                )
            )
        if self.image:
            truth = batch["label"].view(-1).bool()
            scores = output["pred_score"] if "pred_score" in output else output["pred_label"]
            prediction = scores.view(-1) > self.threshold
            if truth.shape != prediction.shape:
                raise ValueError("Image scores and labels must have matching batch dimensions.")
            self.image_counts += torch.stack(
                (
                    (prediction & truth).sum(),
                    (prediction & ~truth).sum(),
                    (~prediction & truth).sum(),
                    (~prediction & ~truth).sum(),
                )
            ).double()

    def include_pixels(self, other):
        """Pool pixel statistics from another disjoint subset before reduction."""
        self.pixel_sums += other.pixel_sums

    def compute(self):
        result = {}
        if self.pixel_mode:
            values = all_reduce_sum(self.pixel_sums.clone())
            count = int(values[7].item())
            if count == 0:
                raise ValueError("No pixel evaluation samples were processed.")
            if self.pixel_mode == "per_image":
                for index, name in enumerate(
                    ("pixel_f1", "pixel_recall", "pixel_accuracy")
                ):
                    result[name] = (values[index] / count).item()
            else:
                tp, fp, fn = values[3:6]
                result["entire_pixel_f1"] = (2 * tp / (2 * tp + fp + fn).clamp_min(1e-12)).item()
                result["entire_pixel_recall"] = (tp / (tp + fn).clamp_min(1e-12)).item()
            result["num_pixel_images"] = count
        if self.image:
            tp, fp, fn, tn = all_reduce_sum(self.image_counts.clone())
            count = int((tp + fp + fn + tn).item())
            if count == 0:
                raise ValueError("No image evaluation samples were processed.")
            precision, recall = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
            result["image_f1"] = (2 * precision * recall / (precision + recall + 1e-9)).item()
            result["image_accuracy"] = ((tp + tn) / count).item()
            result["num_images"] = count
        return result


def average_domains(results, names):
    selected = [results[name] for name in names if name in results]
    if not selected:
        return {}
    output = {}
    for section in selected[0]:
        entries = [result[section] for result in selected if section in result]
        if not entries:
            continue
        output[section] = {
            key: sum(entry[key] for entry in entries) / len(entries)
            for key in entries[0]
            if not key.startswith("num_")
        }
    return output
