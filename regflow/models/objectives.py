import math

import torch
from torch.nn import functional as F


class ArtifactMILObjective:
    def _resolve_k(self, N: int, ratio_or_k: float) -> int:
        if ratio_or_k <= 0:
            raise ValueError(f"ratio_or_k must be > 0, got {ratio_or_k}")
        if ratio_or_k < 1.0:
            return min(max(1, int(math.ceil(N * ratio_or_k))), N)
        return min(max(1, int(ratio_or_k)), N)

    def _adaptive_pool(self, score_map_small: torch.Tensor):
        flat = score_map_small.flatten(1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.pred_calib_eps)
        normed = (flat - mean) / std
        weight = torch.softmax(self.adaptive_pool_alpha * normed, dim=1)
        bag_score = (weight * flat).sum(dim=1)
        eff_k = 1.0 / weight.pow(2).sum(dim=1).clamp_min(self.pred_calib_eps)
        return (bag_score, eff_k)

    def _mil_ranking_loss(self, logit_map_small: torch.Tensor, label: torch.Tensor):
        y = label.view(-1).long()
        fake_mask = y == 1
        real_mask = y == 0
        zero = logit_map_small.new_tensor(0.0)
        if not fake_mask.any() or not real_mask.any():
            stats = {
                "fake_bag_mean": zero,
                "real_bag_mean": zero,
                "eff_k_fake": zero,
                "eff_k_real": zero,
            }
            return (zero, stats)
        fake_bag, fake_effk = self._adaptive_pool(logit_map_small[fake_mask])
        real_bag, real_effk = self._adaptive_pool(logit_map_small[real_mask])
        rank_loss = F.softplus(self.mil_margin - fake_bag[:, None] + real_bag[None, :]).mean()
        stats = {
            "fake_bag_mean": fake_bag.detach().mean(),
            "real_bag_mean": real_bag.detach().mean(),
            "eff_k_fake": fake_effk.detach().mean(),
            "eff_k_real": real_effk.detach().mean(),
        }
        return (rank_loss, stats)

    def _top_ratio_mean(self, score_map_small: torch.Tensor, ratio: float, largest: bool = True):
        flat = score_map_small.flatten(1)
        N = flat.shape[1]
        k = self._resolve_k(N, ratio)
        vals, _ = torch.topk(flat, k=k, dim=1, largest=largest, sorted=False)
        return vals.mean(dim=1)

    def _real_hard_negative_loss(self, logit_map_small: torch.Tensor, label: torch.Tensor):
        y = label.view(-1).long()
        real_mask = y == 0
        if not real_mask.any():
            zero = logit_map_small.new_tensor(0.0)
            return (zero, zero)
        real_top = self._top_ratio_mean(
            logit_map_small[real_mask], ratio=self.real_hardneg_ratio, largest=True
        )
        loss = F.softplus(real_top).mean()
        return (loss, real_top.detach().mean())

    def _edge_aware_smoothness_loss(
        self, logit_map_small: torch.Tensor, image: torch.Tensor, label: torch.Tensor
    ):
        y = label.view(-1).long()
        fake_mask = y == 1
        if not fake_mask.any():
            return logit_map_small.new_tensor(0.0)
        fake_logits = logit_map_small[fake_mask].unsqueeze(1)
        fake_probs = torch.sigmoid(fake_logits)
        H, W = fake_probs.shape[-2:]
        img_small = F.interpolate(
            image[fake_mask], size=(H, W), mode="bilinear", align_corners=False
        )
        gray = img_small.mean(dim=1, keepdim=True)
        pred_dx = fake_probs[:, :, :, 1:] - fake_probs[:, :, :, :-1]
        pred_dy = fake_probs[:, :, 1:, :] - fake_probs[:, :, :-1, :]
        img_dx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        img_dy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        weight_x = torch.exp(-self.edge_aware_gamma * img_dx)
        weight_y = torch.exp(-self.edge_aware_gamma * img_dy)
        smooth_x = (weight_x * pred_dx.pow(2)).mean()
        smooth_y = (weight_y * pred_dy.pow(2)).mean()
        return smooth_x + smooth_y

    def _fake_sparsity_loss(self, logit_map_small: torch.Tensor, label: torch.Tensor):
        y = label.view(-1).long()
        fake_mask = y == 1
        if not fake_mask.any():
            return logit_map_small.new_tensor(0.0)
        fake_probs = torch.sigmoid(logit_map_small[fake_mask])
        sparse_loss = fake_probs.mean()
        return sparse_loss

    def _fake_peak_loss(self, logit_map_small: torch.Tensor, label: torch.Tensor):
        y = label.view(-1).long()
        fake_mask = y == 1
        if not fake_mask.any():
            zero = logit_map_small.new_tensor(0.0)
            return (zero, zero, zero)
        fake_logits = logit_map_small[fake_mask]
        flat = fake_logits.flatten(1)
        Bf, N = flat.shape
        sorted_vals = torch.sort(flat, dim=1, descending=True).values
        k_top = self._resolve_k(N, self.peak_top_ratio)
        top_mean = sorted_vals[:, :k_top].mean(dim=1)
        rest_start = max(k_top, int(math.floor(N * self.peak_rest_min_ratio)))
        rest_end = int(math.ceil(N * self.peak_rest_max_ratio))
        rest_end = max(rest_start + 1, min(rest_end, N))
        rest_mean = sorted_vals[:, rest_start:rest_end].mean(dim=1)
        gap = top_mean - rest_mean
        loss = F.relu(self.peak_margin - gap).mean()
        return (loss, top_mean.detach().mean(), rest_mean.detach().mean())

    def _gather_top_patch_embeddings(
        self,
        emb_map_small: torch.Tensor,
        logit_map_small: torch.Tensor,
        sample_mask: torch.Tensor,
        top_ratio: float,
    ):
        idx = torch.nonzero(sample_mask, as_tuple=False).squeeze(1)
        D = emb_map_small.shape[1]
        if idx.numel() == 0:
            return emb_map_small.new_empty((0, D))
        emb_sel = emb_map_small[idx].flatten(2).transpose(1, 2)
        logit_sel = logit_map_small[idx].flatten(1)
        N = logit_sel.shape[1]
        k = self._resolve_k(N, top_ratio)
        top_idx = torch.topk(logit_sel.detach(), k=k, dim=1, largest=True, sorted=False).indices
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
        selected_emb = torch.gather(emb_sel, dim=1, index=gather_idx)
        selected_emb = selected_emb.reshape(-1, D)
        selected_emb = F.normalize(selected_emb, dim=1)
        return selected_emb

    def _supervised_contrastive_loss(self, features: torch.Tensor, labels: torch.Tensor):
        M = features.shape[0]
        if M < 2:
            return features.new_tensor(0.0)
        logits = torch.matmul(features, features.t()) / self.contrast_temp
        device = logits.device
        self_mask = torch.eye(M, dtype=torch.bool, device=device)
        logits = logits.masked_fill(self_mask, -10000.0)
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()) & ~self_mask
        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return features.new_tensor(0.0)
        log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        mean_log_prob_pos = (log_prob * pos_mask.float()).sum(dim=1) / pos_count.clamp_min(
            1
        ).float()
        loss = -mean_log_prob_pos[valid].mean()
        return loss

    def _patch_contrastive_loss(
        self,
        emb_map_small: torch.Tensor,
        logit_map_small: torch.Tensor,
        label: torch.Tensor,
    ):
        y = label.view(-1).long()
        fake_mask = y == 1
        real_mask = y == 0
        fake_high = self._gather_top_patch_embeddings(
            emb_map_small, logit_map_small, fake_mask, self.contrast_top_ratio
        )
        real_hard = self._gather_top_patch_embeddings(
            emb_map_small, logit_map_small, real_mask, self.contrast_top_ratio
        )
        if fake_high.shape[0] < 2 or real_hard.shape[0] < 2:
            zero = emb_map_small.new_tensor(0.0)
            return (zero, zero, zero)
        feats = torch.cat([fake_high, real_hard], dim=0)
        labels = torch.cat(
            [
                torch.ones(fake_high.shape[0], device=feats.device, dtype=torch.long),
                torch.zeros(real_hard.shape[0], device=feats.device, dtype=torch.long),
            ],
            dim=0,
        )
        ctr_loss = self._supervised_contrastive_loss(feats, labels)
        return (
            ctr_loss,
            fake_high.detach().norm(dim=1).mean(),
            real_hard.detach().norm(dim=1).mean(),
        )
