from functools import partial
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .vit import LocalizationViT, get_abs_pos


class ChannelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-06):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class CrossFeatureFusion(nn.Module):
    def __init__(self, clip_dim=1024, x_dim=768, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = x_dim // num_heads
        assert self.head_dim * num_heads == x_dim, "x_dim must be divisible by num_heads"
        self.channel_adjust = nn.Sequential(
            nn.Conv2d(clip_dim, x_dim, kernel_size=1), ChannelLayerNorm(x_dim)
        )
        self.query_proj = nn.Linear(x_dim, x_dim)
        self.key_proj = nn.Linear(x_dim, x_dim)
        self.value_proj = nn.Linear(x_dim, x_dim)
        self.output_proj = nn.Sequential(nn.Linear(x_dim, x_dim), nn.LayerNorm(x_dim))
        self.scale = self.head_dim ** (-0.5)

    def forward(self, x, clip_features):
        batch_size = x.shape[0]
        target_h, target_w = x.shape[1:3]
        clip_resized = F.interpolate(
            clip_features,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        clip_adjusted = self.channel_adjust(clip_resized)
        clip_adjusted = clip_adjusted.permute(0, 2, 3, 1)
        seq_len = target_h * target_w
        query = clip_adjusted.reshape(batch_size, seq_len, -1)
        key = x.reshape(batch_size, seq_len, -1)
        value = x.reshape(batch_size, seq_len, -1)
        query = self.query_proj(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.key_proj(key).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        value = self.value_proj(value).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        attended_features = torch.matmul(attention_weights, value)
        attended_features = attended_features.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        fused = self.output_proj(attended_features)
        fused = fused.reshape(batch_size, target_h, target_w, -1)
        return fused


class FeatureHook:
    def __init__(self, module):
        self.output = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.output = output

    def close(self):
        self.handle.remove()


class FrozenCLIP(nn.Module):
    def __init__(self, cache_dir, initialize_pretrained):
        super().__init__()
        if initialize_pretrained:
            import clip

            checkpoint_path = Path(cache_dir) / "ViT-L-14.pt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"CLIP checkpoint not found: {checkpoint_path}. "
                    "Download ViT-L/14 once as described in README.md before starting training."
                )
            pretrained, _ = clip.load(str(checkpoint_path), device="cpu", jit=False)
            self.visual = pretrained.visual.float()
        else:
            from clip.model import VisionTransformer

            self.visual = VisionTransformer(
                input_resolution=224,
                patch_size=14,
                width=1024,
                layers=24,
                heads=16,
                output_dim=768,
            )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def encode_image(self, image):
        return self.visual(image.to(dtype=self.visual.conv1.weight.dtype))


class ReGFLoWAdapter(nn.Module):
    def __init__(self, img_size, clip_channels, fusion_map, mae_checkpoint, initialize_pretrained):
        super().__init__()
        self.vit_model = LocalizationViT(
            img_size=img_size,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            window_block_indexes=[0, 1, 3, 4, 6, 7, 9, 10],
            use_rel_pos=True,
        )
        if initialize_pretrained:
            checkpoint_path = Path(mae_checkpoint)
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"MAE checkpoint not found: {checkpoint_path}. "
                    "Download the official ViT-Base checkpoint as described in README.md."
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = checkpoint.get("model", checkpoint)
            incompatible = self.vit_model.load_state_dict(state, strict=False)
            missing = [
                key
                for key in incompatible.missing_keys
                if not key.endswith((".attn.rel_pos_h", ".attn.rel_pos_w"))
            ]
            unexpected = [
                key
                for key in incompatible.unexpected_keys
                if key not in ("cls_token", "mask_token", "norm.weight", "norm.bias")
                and not key.startswith("decoder_")
            ]
            if missing or unexpected:
                raise RuntimeError(
                    "The checkpoint is not compatible with the official MAE ViT-Base encoder: "
                    f"missing={missing}, unexpected={unexpected}"
                )
        self.fusion_layers = nn.ModuleDict(
            {f"layer_{target}": CrossFeatureFusion(clip_channels, 768) for target in fusion_map}
        )
        self.fusion_map = dict(fusion_map)

    def forward_features(self, image, clip_features):
        x = self.vit_model.patch_embed(image)
        x = x + get_abs_pos(
            self.vit_model.pos_embed,
            self.vit_model.pretrain_use_cls_token,
            x.shape[1:3],
        )
        for index, block in enumerate(self.vit_model.blocks):
            x = block(x)
            if index in self.fusion_map:
                x = x + self.fusion_layers[f"layer_{index}"](
                    x, clip_features[self.fusion_map[index]]
                )
        return x.permute(0, 3, 1, 2)
