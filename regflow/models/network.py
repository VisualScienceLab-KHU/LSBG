import math

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import FeatureHook, FrozenCLIP, ReGFLoWAdapter
from .heads import (
    CLSAttentionAggregator,
    ImageMLPClassifier,
    LocalizationGuidedImageFusion,
    PixelMILHead,
    ReconMapEncoder,
    ReconResidualFusion,
    ZeroInitReconLogitBias,
)
from .objectives import ArtifactMILObjective


MODEL_SETTINGS = {
    "resolution": 512,
    "selected_layers": [2, 4, 8, 12, 16, 20, 23],
    "fusion_map": {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6},
}


class ReGFLoW(ArtifactMILObjective, nn.Module):
    def __init__(
        self,
        model_setting_name="ViTL",
        mil_pool: int = 16,
        score_hidden_dim: int = 256,
        embed_dim: int = 256,
        adaptive_pool_alpha: float = 2.0,
        ce_w: float = 1.0,
        rank_w: float = 1.0,
        smooth_w: float = 0.03,
        sparse_w: float = 0.003,
        real_hardneg_w: float = 0.05,
        contrast_w: float = 0.02,
        peak_w: float = 0.0,
        mil_margin: float = 0.5,
        edge_aware_gamma: float = 10.0,
        contrast_top_ratio: float = 0.1,
        contrast_temp: float = 0.1,
        real_hardneg_ratio: float = 0.1,
        peak_top_ratio: float = 0.1,
        peak_rest_min_ratio: float = 0.3,
        peak_rest_max_ratio: float = 0.7,
        peak_margin: float = 0.25,
        use_imagewise_pred_calibration: int = 1,
        pred_calib_k: float = 0.9,
        pred_calib_scale: float = 2.0,
        pred_calib_eps: float = 1e-06,
        pred_calib_ratio_center: float = 0.08,
        pred_calib_ratio_gain: float = 0.5,
        pred_calib_k_min: float = 0.4,
        pred_calib_k_max: float = 1.2,
        new_head_lr_mult: float = 3.0,
        image_hidden_dim: int = 256,
        image_dropout: float = 0.1,
        use_loc_to_img: int = 1,
        loc_to_img_top_ratio: float = 0.1,
        loc_to_img_attn_temp: float = 6.0,
        loc_to_img_grad_scale: float = 0.1,
        loc_to_img_detach_score: int = 1,
        loc_to_img_detach_feature: int = 0,
        loc_to_img_hidden_dim: int = 512,
        loc_to_img_dropout: float = 0.1,
        loc_to_img_scale_init: float = 0.05,
        loc_to_img_use_score_stats: int = 1,
        use_recon_fusion: int = 1,
        use_prior_logit_bias: int = 1,
        recon_dim: int = 32,
        recon_dropout: float = 0.05,
        recon_proj_dim: int = 128,
        rgb_proj_dim: int = 512,
        recon_clamp: float = 8.0,
        recon_fusion_scale_init: float = 0.05,
        recon_bias_hidden_dim: int = 128,
        recon_bias_recon_hidden_dim: int = 64,
        recon_bias_scale_init: float = 0.05,
        mae_checkpoint="weights/mae/mae_pretrain_vit_base.pth",
        clip_cache_dir="weights/clip",
        initialize_pretrained=True,
    ):
        super().__init__()
        if model_setting_name != "ViTL":
            raise ValueError("ReGFLoW supports model_setting_name=ViTL only.")
        settings = MODEL_SETTINGS
        self.model_setting_name = model_setting_name
        self.selected_layers = settings["selected_layers"]
        self.resolution = settings["resolution"]
        if mil_pool not in (8, 16):
            raise ValueError("mil_pool must be 8 or 16.")
        self.mil_pool = int(mil_pool)
        self.score_hidden_dim = int(score_hidden_dim)
        self.embed_dim = int(embed_dim)
        self.adaptive_pool_alpha = float(adaptive_pool_alpha)
        self.ce_w = float(ce_w)
        self.rank_w = float(rank_w)
        self.smooth_w = float(smooth_w)
        self.sparse_w = float(sparse_w)
        self.real_hardneg_w = float(real_hardneg_w)
        self.contrast_w = float(contrast_w)
        self.peak_w = float(peak_w)
        self.mil_margin = float(mil_margin)
        self.edge_aware_gamma = float(edge_aware_gamma)
        self.contrast_top_ratio = float(contrast_top_ratio)
        self.contrast_temp = float(contrast_temp)
        self.real_hardneg_ratio = float(real_hardneg_ratio)
        self.peak_top_ratio = float(peak_top_ratio)
        self.peak_rest_min_ratio = float(peak_rest_min_ratio)
        self.peak_rest_max_ratio = float(peak_rest_max_ratio)
        self.peak_margin = float(peak_margin)
        self.use_imagewise_pred_calibration = bool(int(use_imagewise_pred_calibration))
        self.pred_calib_k = float(pred_calib_k)
        self.pred_calib_scale = float(pred_calib_scale)
        self.pred_calib_eps = float(pred_calib_eps)
        self.pred_calib_ratio_center = float(pred_calib_ratio_center)
        self.pred_calib_ratio_gain = float(pred_calib_ratio_gain)
        self.pred_calib_k_min = float(pred_calib_k_min)
        self.pred_calib_k_max = float(pred_calib_k_max)
        self.new_head_lr_mult = float(new_head_lr_mult)
        self.use_loc_to_img = bool(int(use_loc_to_img))
        self.loc_to_img_top_ratio = float(loc_to_img_top_ratio)
        self.loc_to_img_attn_temp = float(loc_to_img_attn_temp)
        self.loc_to_img_grad_scale = float(loc_to_img_grad_scale)
        self.loc_to_img_detach_score = bool(int(loc_to_img_detach_score))
        self.loc_to_img_detach_feature = bool(int(loc_to_img_detach_feature))
        self.loc_to_img_hidden_dim = int(loc_to_img_hidden_dim)
        self.loc_to_img_dropout = float(loc_to_img_dropout)
        self.loc_to_img_scale_init = float(loc_to_img_scale_init)
        self.loc_to_img_use_score_stats = bool(int(loc_to_img_use_score_stats))
        self.use_recon_fusion = bool(int(use_recon_fusion))
        self.use_recon_logit_bias = bool(int(use_prior_logit_bias))
        self.recon_dim = int(recon_dim)
        self.recon_dropout = float(recon_dropout)
        self.recon_proj_dim = int(recon_proj_dim)
        self.rgb_proj_dim = int(rgb_proj_dim)
        self.recon_clamp = float(recon_clamp)
        self.recon_fusion_scale_init = float(recon_fusion_scale_init)
        self.clip = FrozenCLIP(clip_cache_dir, initialize_pretrained)
        self.hooks = [
            FeatureHook(self.clip.visual.transformer.resblocks[i].ln_2)
            for i in self.selected_layers
        ]
        visemb_dim = self.clip.visual.conv1.out_channels
        align_dim = 768
        self.align_dim = align_dim
        self.aggregator = ReGFLoWAdapter(
            img_size=self.resolution,
            clip_channels=visemb_dim,
            fusion_map=settings["fusion_map"],
            mae_checkpoint=mae_checkpoint,
            initialize_pretrained=initialize_pretrained,
        )
        self.cls_aggregator = CLSAttentionAggregator(
            feature_dim=visemb_dim,
            out_dim=align_dim,
            num_heads=8,
            num_layers=1,
            dropout_rate=0.1,
        )
        self.image_classifier = ImageMLPClassifier(
            in_dim=align_dim,
            hidden_dim=int(image_hidden_dim),
            dropout=float(image_dropout),
        )
        if self.use_loc_to_img:
            self.loc_image_fusion = LocalizationGuidedImageFusion(
                feature_dim=align_dim,
                hidden_dim=self.loc_to_img_hidden_dim,
                top_ratio=self.loc_to_img_top_ratio,
                attn_temp=self.loc_to_img_attn_temp,
                dropout=self.loc_to_img_dropout,
                grad_scale=self.loc_to_img_grad_scale,
                detach_score=self.loc_to_img_detach_score,
                detach_feature=self.loc_to_img_detach_feature,
                use_score_stats=self.loc_to_img_use_score_stats,
                scale_init=self.loc_to_img_scale_init,
                eps=self.pred_calib_eps,
            )
        else:
            self.loc_image_fusion = nn.Identity()
        self.mil_head = PixelMILHead(
            in_dim=align_dim, hidden_dim=self.score_hidden_dim, embed_dim=self.embed_dim
        )
        if self.use_recon_fusion:
            self.recon_encoder = ReconMapEncoder(
                in_ch=1, base_dim=16, out_dim=self.recon_dim, dropout=self.recon_dropout
            )
            self.recon_fusion = ReconResidualFusion(
                align_dim=align_dim,
                recon_dim=self.recon_dim,
                rgb_proj_dim=self.rgb_proj_dim,
                recon_proj_dim=self.recon_proj_dim,
                dropout=self.recon_dropout,
                scale_init=self.recon_fusion_scale_init,
            )
        else:
            self.recon_encoder = nn.Identity()
            self.recon_fusion = nn.Identity()
        if self.use_recon_logit_bias:
            self.recon_logit_bias = ZeroInitReconLogitBias(
                align_dim=align_dim,
                recon_dim=self.recon_dim,
                hidden_dim=int(recon_bias_hidden_dim),
                recon_hidden_dim=int(recon_bias_recon_hidden_dim),
                dropout=self.recon_dropout,
                scale_init=float(recon_bias_scale_init),
            )
        else:
            self.recon_logit_bias = nn.Identity()
        self.ce_criterion = nn.CrossEntropyLoss()

    def get_param_groups(
        self, base_lr: float, weight_decay: float, *, separate_image_groups=False
    ):
        base_prefixes = (
            "aggregator.vit_model.",
            "aggregator.fusion_layers.",
            "cls_aggregator.",
            "image_classifier.",
        )
        head_prefixes = (
            "mil_head.",
            "recon_encoder.",
            "recon_fusion.",
            "recon_logit_bias.",
            "loc_image_fusion.",
        )
        image_base_prefixes = ("cls_aggregator.", "image_classifier.")
        buckets = {name: [] for name in ("base", "heads", "image_base", "image_fusion")}
        trainable_names = []
        for name, param in self.named_parameters():
            param.requires_grad = False
            if name.startswith(base_prefixes):
                group = (
                    "image_base"
                    if separate_image_groups and name.startswith(image_base_prefixes)
                    else "base"
                )
            elif name.startswith(head_prefixes):
                group = (
                    "image_fusion"
                    if separate_image_groups and name.startswith("loc_image_fusion.")
                    else "heads"
                )
            else:
                continue
            param.requires_grad = True
            buckets[group].append(param)
            trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError("No trainable parameters were selected in ReGFLoW.get_param_groups().")
        groups = []
        for name, params in buckets.items():
            if not params:
                continue
            scale = self.new_head_lr_mult if name in {"heads", "image_fusion"} else 1.0
            groups.append(
                {
                    "params": params,
                    "name": name,
                    "is_image": name in {"image_base", "image_fusion"},
                    "lr": base_lr * scale,
                    "lr_scale": scale,
                    "weight_decay": weight_decay,
                }
            )
        return groups, trainable_names

    def _pool_feature_map(self, feat_32: torch.Tensor) -> torch.Tensor:
        if self.mil_pool == 16:
            return F.avg_pool2d(feat_32, kernel_size=2, stride=2)
        return F.avg_pool2d(feat_32, kernel_size=4, stride=4)

    def _compute_raw_pos_ratio(self, pre_calib_pred_mask: torch.Tensor):
        return pre_calib_pred_mask.flatten(1).mean(dim=1)

    def _compute_ratio_adaptive_calib_k(self, raw_pos_ratio: torch.Tensor):
        center = max(self.pred_calib_ratio_center, self.pred_calib_eps)
        ratio_delta = (raw_pos_ratio - center) / center
        ratio_delta = ratio_delta.clamp(-1.0, 1.0)
        adaptive_k = self.pred_calib_k - self.pred_calib_ratio_gain * ratio_delta
        adaptive_k = adaptive_k.clamp(self.pred_calib_k_min, self.pred_calib_k_max)
        return adaptive_k

    def _imagewise_calibrate_and_sigmoid(self, raw_map: torch.Tensor, raw_pos_ratio: torch.Tensor):
        B = raw_map.shape[0]
        flat = raw_map.flatten(1)
        map_mean = flat.mean(dim=1, keepdim=True)
        map_std = flat.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.pred_calib_eps)
        adaptive_k = self._compute_ratio_adaptive_calib_k(raw_pos_ratio).view(B, 1)
        tau = map_mean + adaptive_k * map_std
        map_std_4d = map_std.view(B, 1, 1, 1)
        tau_4d = tau.view(B, 1, 1, 1)
        calibrated_logit = self.pred_calib_scale * ((raw_map - tau_4d) / map_std_4d)
        pred_mask = torch.sigmoid(calibrated_logit)
        stats = {
            "map_mean": map_mean.detach().mean(),
            "map_std": map_std.detach().mean(),
            "map_tau": tau.detach().mean(),
            "adaptive_k": adaptive_k.detach().mean(),
            "raw_pos_ratio": raw_pos_ratio.detach().mean(),
        }
        return (pred_mask, calibrated_logit, stats)

    def _prepare_recon_map(self, recon_map: torch.Tensor, image: torch.Tensor):
        B = image.shape[0]
        target_size = (128, 128)
        if recon_map is None:
            raise ValueError(
                "A precomputed recon_map is required when use_recon_fusion is enabled."
            )
        recon = recon_map.to(device=image.device, dtype=image.dtype)
        if recon.dim() == 3:
            recon = recon.unsqueeze(1)
        if recon.dim() != 4:
            raise ValueError(f"recon_map must be [B,H,W] or [B,1,H,W], got {recon.shape}")
        if recon.shape[0] != B:
            raise ValueError(f"Batch mismatch: image B={B}, recon_map B={recon.shape[0]}")
        if recon.shape[1] != 1:
            recon = recon.mean(dim=1, keepdim=True)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
        if recon.shape[-2:] != target_size:
            recon = F.interpolate(recon, size=target_size, mode="bilinear", align_corners=False)
        if self.recon_clamp > 0:
            recon = recon.clamp(-self.recon_clamp, self.recon_clamp)
        return recon

    def forward(self, image, mask=None, label=None, recon_map=None, **kwargs):
        if image.ndim != 4 or image.shape[1:] != (3, self.resolution, self.resolution):
            raise ValueError(f"image must have shape [B, 3, {self.resolution}, {self.resolution}].")
        batch_size = image.shape[0]
        clip_image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        with torch.no_grad():
            self.clip.encode_image(clip_image)
            selected_features = torch.stack([hook.output for hook in self.hooks], dim=2)
        token_count, _, layer_count, channels = selected_features.shape
        patch_width = math.isqrt(token_count - 1)
        cls_features = selected_features[0]
        patch_tokens = selected_features[1:]
        patch_features = [
            patch_tokens[:, :, index, :]
            .permute(1, 2, 0)
            .reshape(batch_size, channels, patch_width, patch_width)
            for index in range(layer_count)
        ]
        cls_embed = self.cls_aggregator(cls_features)
        feat_32 = self.aggregator.forward_features(image, patch_features)
        feat_small_base = self._pool_feature_map(feat_32)
        feat_small = feat_small_base
        if self.use_recon_fusion:
            recon_128 = self._prepare_recon_map(recon_map, image)
            recon_feat_32 = self.recon_encoder(recon_128)
            if recon_feat_32.shape[-2:] != feat_32.shape[-2:]:
                recon_feat_32 = F.interpolate(
                    recon_feat_32,
                    size=feat_32.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            recon_feat_small = self._pool_feature_map(recon_feat_32)
            feat_small, _ = self.recon_fusion(feat_small_base, recon_feat_small)
        emb_small, mil_logit_small = self.mil_head(feat_small)
        if self.use_recon_logit_bias and self.use_recon_fusion:
            recon_bias, recon_bias_aux = self.recon_logit_bias(feat_small, recon_feat_small)
            mil_logit_small = mil_logit_small + recon_bias_aux["recon_bias_scale"] * recon_bias
        if self.use_loc_to_img:
            image_embed, _ = self.loc_image_fusion(
                cls_embed=cls_embed,
                loc_feat_map=feat_small,
                loc_logit_map=mil_logit_small,
            )
        else:
            image_embed = cls_embed
        image_logits = self.image_classifier(image_embed)
        mil_logit_up = F.interpolate(
            mil_logit_small.unsqueeze(1),
            size=image.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        pre_calib_pred_mask = torch.sigmoid(mil_logit_up)
        raw_pos_ratio = self._compute_raw_pos_ratio(pre_calib_pred_mask)
        if self.use_imagewise_pred_calibration:
            pred_mask, _, _ = self._imagewise_calibrate_and_sigmoid(mil_logit_up, raw_pos_ratio)
        else:
            pred_mask = pre_calib_pred_mask
        result = {
            "pred_mask": pred_mask,
            "pred_label": torch.argmax(image_logits, dim=1),
            "pred_score": torch.softmax(image_logits, dim=1)[:, 1],
            "image_logits": image_logits,
        }
        if label is None:
            return result
        label = label.view(-1).long()
        ce_loss = self.ce_criterion(image_logits, label)
        rank_loss, _ = self._mil_ranking_loss(mil_logit_small, label)
        real_hardneg_loss, _ = self._real_hard_negative_loss(mil_logit_small, label)
        smooth_loss = self._edge_aware_smoothness_loss(mil_logit_small, image, label)
        sparse_loss = self._fake_sparsity_loss(mil_logit_small, label)
        peak_loss, _, _ = self._fake_peak_loss(mil_logit_small, label)
        contrast_loss, _, _ = self._patch_contrastive_loss(emb_small, mil_logit_small, label)
        loss = (
            self.ce_w * ce_loss
            + self.rank_w * rank_loss
            + self.smooth_w * smooth_loss
            + self.sparse_w * sparse_loss
            + self.real_hardneg_w * real_hardneg_loss
            + self.peak_w * peak_loss
            + self.contrast_w * contrast_loss
        )
        result.update(
            {
                "backward_loss": loss,
                "visual_loss": {
                    "combined_loss": loss,
                    "loss_ce": ce_loss,
                    "loss_rank": rank_loss,
                    "loss_real_hardneg": real_hardneg_loss,
                    "loss_peak": peak_loss,
                    "loss_contrast": contrast_loss,
                    "loss_smooth": smooth_loss,
                    "loss_sparse": sparse_loss,
                },
            }
        )
        return result


def load_model_weights(model, state_dict):
    ignored_prefixes = (
        "clip.transformer.",
        "clip.token_embedding.",
        "clip.ln_final.",
        "aggregator.mask_decoder.",
        "aggregator.text_projection.",
    )
    ignored_names = {
        "clip.positional_embedding",
        "clip.text_projection",
        "clip.logit_scale",
    }
    cleaned = {}
    removed = []
    for original_key, value in state_dict.items():
        key = original_key.removeprefix("module.")
        if key in ignored_names or key.startswith(ignored_prefixes):
            removed.append(original_key)
        else:
            cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    return removed
