import math

import torch
from torch import nn
from torch.nn import functional as F


def _pick_group_count(num_channels: int, candidates=(32, 16, 8, 4, 2, 1)) -> int:
    for g in candidates:
        if num_channels % g == 0:
            return g
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.GroupNorm(_pick_group_count(out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class CLSAttentionAggregator(nn.Module):
    def __init__(self, feature_dim=1024, out_dim=768, num_heads=8, num_layers=1, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "multihead_attn": nn.MultiheadAttention(
                            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
                        ),
                        "layer_norm": nn.LayerNorm(feature_dim),
                        "dropout": nn.Dropout(dropout_rate),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.projector = nn.Linear(feature_dim, out_dim, bias=False)

    def forward(self, x):
        for layer in self.layers:
            attn_output, _ = layer["multihead_attn"](x, x, x)
            x = layer["layer_norm"](attn_output + x)
            x = layer["dropout"](x)
        x = torch.mean(x, dim=1)
        x = self.projector(x)
        return x


class ImageMLPClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class LocalizationGuidedImageFusion(nn.Module):
    def __init__(
        self,
        feature_dim=768,
        hidden_dim=512,
        top_ratio=0.1,
        attn_temp=6.0,
        dropout=0.1,
        grad_scale=0.1,
        detach_score=True,
        detach_feature=False,
        use_score_stats=True,
        scale_init=0.05,
        eps=1e-06,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.top_ratio = float(top_ratio)
        self.attn_temp = float(attn_temp)
        self.grad_scale = float(grad_scale)
        self.detach_score = bool(detach_score)
        self.detach_feature = bool(detach_feature)
        self.use_score_stats = bool(use_score_stats)
        self.eps = float(eps)
        self.loc_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.use_score_stats:
            self.score_proj = nn.Sequential(
                nn.LayerNorm(3),
                nn.Linear(3, feature_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_in_dim = feature_dim * 3
        else:
            self.score_proj = None
            fusion_in_dim = feature_dim * 2
        self.delta_mlp = nn.Sequential(
            nn.LayerNorm(fusion_in_dim),
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.delta_mlp[-1].weight)
        nn.init.zeros_(self.delta_mlp[-1].bias)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))

    def _resolve_k(self, N: int, ratio_or_k: float) -> int:
        if ratio_or_k <= 0:
            raise ValueError(f"ratio_or_k must be > 0, got {ratio_or_k}")
        if ratio_or_k < 1.0:
            return min(max(1, int(math.ceil(N * ratio_or_k))), N)
        return min(max(1, int(ratio_or_k)), N)

    def _grad_scale_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.detach_feature:
            return x.detach()
        if self.grad_scale >= 1.0:
            return x
        if self.grad_scale <= 0.0:
            return x.detach()
        return x.detach() + self.grad_scale * (x - x.detach())

    def forward(
        self,
        cls_embed: torch.Tensor,
        loc_feat_map: torch.Tensor,
        loc_logit_map: torch.Tensor,
    ):
        B, C, h, w = loc_feat_map.shape
        assert cls_embed.shape == (B, C), (
            f"cls_embed shape mismatch: expected {(B, C)}, got {cls_embed.shape}"
        )
        assert loc_logit_map.shape == (B, h, w), (
            f"loc_logit_map shape mismatch: expected {(B, h, w)}, got {loc_logit_map.shape}"
        )
        flat_logit = loc_logit_map.flatten(1)
        N = flat_logit.shape[1]
        k = self._resolve_k(N, self.top_ratio)
        if self.detach_score:
            score_for_attn = flat_logit.detach()
        else:
            score_for_attn = flat_logit
        score_mean = score_for_attn.mean(dim=1, keepdim=True)
        score_std = score_for_attn.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
        normed_score = (score_for_attn - score_mean) / score_std
        top_vals, top_idx = torch.topk(normed_score, k=k, dim=1, largest=True, sorted=False)
        masked_score = normed_score.new_full(normed_score.shape, -10000.0)
        masked_score.scatter_(1, top_idx, top_vals)
        attn = torch.softmax(self.attn_temp * masked_score, dim=1)
        loc_feat_for_image = self._grad_scale_tensor(loc_feat_map)
        loc_tokens = loc_feat_for_image.flatten(2).transpose(1, 2)
        loc_context = torch.sum(attn.unsqueeze(-1) * loc_tokens, dim=1)
        loc_embed = self.loc_proj(loc_context)
        bag_logit = torch.sum(attn * score_for_attn, dim=1)
        top_raw_vals = torch.gather(score_for_attn, dim=1, index=top_idx)
        top_logit_mean = top_raw_vals.mean(dim=1)
        prob_mean = torch.sigmoid(score_for_attn).mean(dim=1)
        if self.use_score_stats:
            score_stats = torch.stack([bag_logit, top_logit_mean, prob_mean], dim=1)
            score_embed = self.score_proj(score_stats)
            fusion_input = torch.cat([cls_embed, loc_embed, score_embed], dim=1)
        else:
            fusion_input = torch.cat([cls_embed, loc_embed], dim=1)
        delta = self.delta_mlp(fusion_input)
        scale = torch.tanh(self.scale)
        fused_embed = cls_embed + scale * delta
        eff_k = 1.0 / attn.pow(2).sum(dim=1).clamp_min(self.eps)
        aux = {
            "loc_img_scale": scale,
            "loc_img_context_norm": loc_context.detach().norm(dim=1).mean(),
            "loc_img_delta_norm": delta.detach().norm(dim=1).mean(),
            "loc_img_eff_k": eff_k.detach().mean(),
            "loc_img_bag_logit": bag_logit.detach().mean(),
            "loc_img_top_logit": top_logit_mean.detach().mean(),
            "loc_img_prob_mean": prob_mean.detach().mean(),
            "loc_img_attn": attn.detach().view(B, 1, h, w),
        }
        return (fused_embed, aux)


class PixelMILHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, embed_dim=256):
        super().__init__()
        self.stem = ConvGNAct(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0)
        self.emb_proj = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1, bias=True)
        self.logit_head = nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        feat = self.stem(x)
        emb = self.emb_proj(feat)
        emb = F.normalize(emb, dim=1)
        logit = self.logit_head(feat).squeeze(1)
        return (emb, logit)


class ReconMapEncoder(nn.Module):
    def __init__(self, in_ch=1, base_dim=16, out_dim=32, dropout=0.05):
        super().__init__()
        self.net = nn.Sequential(
            ConvGNAct(in_ch, base_dim, kernel_size=3, stride=2, padding=1),
            ConvGNAct(base_dim, out_dim, kernel_size=3, stride=2, padding=1),
            ConvGNAct(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ReconResidualFusion(nn.Module):
    def __init__(
        self,
        align_dim=768,
        recon_dim=32,
        rgb_proj_dim=512,
        recon_proj_dim=128,
        dropout=0.05,
        scale_init=0.05,
    ):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            ConvGNAct(align_dim, rgb_proj_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(rgb_proj_dim, rgb_proj_dim, kernel_size=3, stride=1, padding=1),
        )
        self.recon_pix_proj = nn.Sequential(
            ConvGNAct(recon_dim, recon_proj_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(recon_proj_dim, recon_proj_dim, kernel_size=3, stride=1, padding=1),
        )
        self.loc_fusion = nn.Sequential(
            ConvGNAct(
                rgb_proj_dim + recon_proj_dim,
                align_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Dropout2d(dropout),
            ConvGNAct(align_dim, align_dim, kernel_size=3, stride=1, padding=1),
        )
        self.gamma = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, feat_rgb, feat_recon):
        rgb_proj_feat = self.rgb_proj(feat_rgb)
        recon_proj_feat = self.recon_pix_proj(feat_recon)
        feat_cat = torch.cat([rgb_proj_feat, recon_proj_feat], dim=1)
        delta = self.loc_fusion(feat_cat)
        scale = torch.tanh(self.gamma)
        feat_fused = feat_rgb + scale * delta
        aux = {
            "rgb_proj_feat": rgb_proj_feat,
            "recon_proj_feat": recon_proj_feat,
            "delta": delta,
            "scale": scale,
        }
        return (feat_fused, aux)


class ZeroInitReconLogitBias(nn.Module):
    def __init__(
        self,
        align_dim=768,
        recon_dim=32,
        hidden_dim=128,
        recon_hidden_dim=64,
        dropout=0.05,
        scale_init=0.05,
    ):
        super().__init__()
        self.rgb_proj = nn.Sequential(
            ConvGNAct(align_dim, hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.recon_proj = nn.Sequential(
            ConvGNAct(recon_dim, recon_hidden_dim, kernel_size=1, stride=1, padding=0),
            ConvGNAct(recon_hidden_dim, recon_hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(
                hidden_dim + recon_hidden_dim,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Dropout2d(dropout),
            ConvGNAct(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        )
        self.out_conv = nn.Conv2d(hidden_dim, 1, kernel_size=1, stride=1, padding=0)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        self.beta = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(self, feat_small, recon_feat_small):
        rgb_feat = self.rgb_proj(feat_small)
        recon_feat = self.recon_proj(recon_feat_small)
        feat_cat = torch.cat([rgb_feat, recon_feat], dim=1)
        fused = self.fuse(feat_cat)
        recon_bias = self.out_conv(fused).squeeze(1)
        scale = torch.tanh(self.beta)
        aux = {
            "recon_bias_raw": recon_bias,
            "recon_bias_scale": scale,
            "recon_bias_rgb_feat": rgb_feat,
            "recon_bias_recon_feat": recon_feat,
        }
        return (recon_bias, aux)
