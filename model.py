# DiT with cross attention and REPA loss


import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from dinov2 import vit_base, vit_giant2, vit_large, vit_small


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    return embedding


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_dtype = x.dtype
        x = x.float()
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * norm * self.weight).to(dtype=x_dtype)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        is_self_attn=True,
        cross_attn_input_size=None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.is_self_attn = is_self_attn

        if is_self_attn:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.context_kv = nn.Linear(cross_attn_input_size, dim * 2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context=None):
        if self.is_self_attn:
            qkv = self.qkv(x)
            qkv = rearrange(qkv, "b l (k h d) -> k b h l d", k=3, h=self.num_heads)
            q, k, v = qkv.unbind(0)
        else:
            q = rearrange(self.q(x), "b l (h d) -> b h l d", h=self.num_heads)
            kv = rearrange(
                self.context_kv(context),
                "b l (k h d) -> k b h l d",
                k=2,
                h=self.num_heads,
            )
            k, v = kv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b h l d -> b l (h d)")
        x = self.proj(x)
        return x


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        cross_attn_input_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = RMSNorm(hidden_size)
        self.self_attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, is_self_attn=True
        )

        self.norm2 = RMSNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            is_self_attn=False,
            cross_attn_input_size=cross_attn_input_size,
        )

        self.norm3 = RMSNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

        self.adaLN_modulation[-1].weight.data.zero_()
        self.adaLN_modulation[-1].bias.data.zero_()

    def forward(self, x, context, c):
        (
            shift_sa,
            scale_sa,
            gate_sa,
            shift_ca,
            scale_ca,
            gate_ca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(9, dim=1)

        norm_x = self.norm1(x)
        norm_x = norm_x * (1 + scale_sa[:, None, :]) + shift_sa[:, None, :]
        x = x + gate_sa[:, None, :] * self.self_attn(norm_x)

        norm_x = self.norm2(x)
        norm_x = norm_x * (1 + scale_ca[:, None, :]) + shift_ca[:, None, :]
        x = x + gate_ca[:, None, :] * self.cross_attn(norm_x, context)

        norm_x = self.norm3(x)
        norm_x = norm_x * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
        x = x + gate_mlp[:, None, :] * self.mlp(norm_x)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.patch_size = patch_size

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_proj(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class DiT(nn.Module):
    def __init__(
        self,
        in_channels=4,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=128,
        repa_target_layers=[6, 8],
        repa_target_dim=128,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.repa_target_layers = repa_target_layers
        self.patch_embed = PatchEmbed(patch_size, in_channels, hidden_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, 1024, hidden_size))

        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.SiLU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    cross_attn_input_size=cross_attn_input_size,
                )
                for _ in range(depth)
            ]
        )

        self.repa_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, repa_target_dim),
                )
                for _ in self.repa_target_layers
            ]
        )

        self.final_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

        self.final_norm = RMSNorm(hidden_size)
        self.final_proj = nn.Linear(
            hidden_size, patch_size * patch_size * self.out_channels
        )
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(self, x, context, timesteps):
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.positional_embedding.repeat(b, 1, 1)[:, : x.shape[1], :]

        t_emb = timestep_embedding(timesteps, self.hidden_size).to(
            x.device, dtype=x.dtype
        )
        t_emb = self.time_embed(t_emb)

        repa_feats = []
        repa_idx = 0
        for idx, block in enumerate(self.blocks):
            x = block(x, context, t_emb)

            if idx in self.repa_target_layers:
                repa_feat = self.repa_transforms[repa_idx](x)
                repa_feats.append(repa_feat)
                repa_idx += 1

        final_shift, final_scale = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x)
        x = x * (1 + final_scale[:, None, :]) + final_shift[:, None, :]
        x = self.final_proj(x)
        x = rearrange(
            x,
            "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
            h=h // self.patch_size,
            w=w // self.patch_size,
            p1=self.patch_size,
            p2=self.patch_size,
        )
        return x, repa_feats

    def get_mup_setup(self, learning_rate, weight_decay, constant_param_classes):

        no_decay_name_list = ["bias", "norm"]

        optimizer_grouped_parameters = []
        final_optimizer_settings = {}

        param_groups = defaultdict(
            lambda: {"params": [], "weight_decay": None, "lr": None}
        )

        for n, p in self.named_parameters():
            if p.requires_grad:
                # Define learning rate for specific types of params
                if any(ndnl in n for ndnl in no_decay_name_list):
                    lr_value = learning_rate * 0.1
                    per_layer_weight_decay_value = 0.0
                else:
                    hidden_dim = p.shape[-1]
                    lr_value = learning_rate * (32 / hidden_dim)
                    per_layer_weight_decay_value = (
                        weight_decay * hidden_dim / 1024
                    )  # weight decay 0.1 (SP: 1024)

                # in the case of embedding layer, we use higher lr.
                if any(
                    constant_param_class in n
                    for constant_param_class in constant_param_classes
                ):
                    lr_value = learning_rate * 0.1
                    per_layer_weight_decay_value = 0.0

                group_key = (lr_value, per_layer_weight_decay_value)
                param_groups[group_key]["params"].append(p)
                param_groups[group_key]["weight_decay"] = per_layer_weight_decay_value
                param_groups[group_key]["lr"] = lr_value

                final_optimizer_settings[n] = {
                    "lr": lr_value,
                    "wd": per_layer_weight_decay_value,
                    "shape": str(list(p.shape)),
                }

        optimizer_grouped_parameters = [v for v in param_groups.values()]

        return optimizer_grouped_parameters, final_optimizer_settings


class REPALoss(nn.Module):
    def __init__(self, dino_ver="dinov2_vitg14", device="cuda", dtype=torch.bfloat16):
        super().__init__()
        model_ver = {
            "dinov2_vits14": vit_small,
            "dinov2_vitb14": vit_base,
            "dinov2_vitl14": vit_large,
            "dinov2_vitg14": vit_giant2,
        }
        self.dino = model_ver[dino_ver]()
        self.dino.load_state_dict(torch.load(f"{dino_ver}.pth", map_location=device))
        self.dino.requires_grad_(False)
        self.dino.to(device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def forward(self, x_feat, y_image):

        y_rep = self.dino.forward_features(y_image.to(self.device, dtype=self.dtype))[
            "x_norm_patchtokens"
        ]
        x_feat = x_feat.to(self.device, dtype=self.dtype)
        if x_feat.ndim == 4:
            x_feat = rearrange(x_feat, "b n h w -> b (h w) n")
        # print(x_feat.shape, y_rep.shape)
        return -F.cosine_similarity(x_feat, y_rep, dim=-1).mean(dim=[1])


def test_model():

    device = "cuda:1"
    dtype = torch.bfloat16

    model = DiT(
        in_channels=4,
        patch_size=2,
        hidden_size=512,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        cross_attn_input_size=129,
        repa_target_layers=[6, 8],
        repa_target_dim=128,
    )

    model.to(device, dtype=dtype)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count / 1e6 :.4f} M")

    image_data = torch.randn(5, 4, 6, 8 * 17).to(device, dtype=dtype)
    context = torch.randn(5, 77, 129).to(device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (5,)).to(device)

    output, feats = model(image_data, context, timesteps)
    print(f"Output shape: {output.shape}")
    print(f"Number of repa features: {len(feats)}")
    print(f"Repa feature 0 shape: {feats[0].shape}")
    print(f"Repa feature 1 shape: {feats[1].shape}")
    assert (
        image_data.shape == output.shape
    ), f"Input shape: {image_data.shape}, Output shape: {output.shape}"


def test_repa_loss():
    loss = REPALoss(device="cuda:1", dtype=torch.bfloat16)
    x = torch.randn(5, 1536, 16, 16)
    y = torch.randn(5, 3, 224, 224)
    print(loss(x, y).shape)


activation_scales = {}

from functools import partial


def register_forward_hook(model):
    forward_hooks = []

    for name, module in model.named_modules():

        def hook(module, input, output, name=name):
            if isinstance(output, tuple):
                return

            print(f"{name} input shape: {input[0].shape}")
            print(f"{name} output shape: {output.shape}")
            activation_scales[name + "_input"] = {
                "mean": input[0].mean().detach().cpu().item(),
                "std": input[0].std().detach().cpu().item(),
            }
            activation_scales[name + "_output"] = {
                "mean": output.mean().detach().cpu().item(),
                "std": output.std().detach().cpu().item(),
            }
            print(f"{name} input scale: {activation_scales[name + '_input']}")
            print(f"{name} output scale: {activation_scales[name + '_output']}")

        forward_hooks.append(module.register_forward_hook(partial(hook, name=name)))

    return forward_hooks


torch.no_grad()


def test_activation_scales():

    device = "cuda:1"
    dtype = torch.bfloat16

    global activation_scales

    vae_input = torch.load("test_data/vae_latent_0.pt", map_location="cpu")
    caption_encoded = torch.load("test_data/caption_encoded_0.pt", map_location="cpu")
    timesteps = torch.load("test_data/timesteps_0.pt", map_location="cpu")

    print(vae_input.std(), caption_encoded.std(), timesteps.std())

    for width in [16, 64, 512]:

        dit_model = DiT(
            in_channels=16,
            patch_size=2,
            depth=9,
            num_heads=4,
            mlp_ratio=4.0,
            cross_attn_input_size=4096,
            hidden_size=width,
            repa_target_layers=[6],
            repa_target_dim=1536,
        ).to(device, dtype=dtype)

        forward_hooks = register_forward_hook(dit_model)

        input_image = vae_input.to(device, dtype=dtype)
        input_context = caption_encoded.to(device, dtype=dtype)
        input_timesteps = timesteps.to(device, dtype=dtype)

        dit_model(input_image, input_context, input_timesteps)

        for hook in forward_hooks:
            hook.remove()

        #
        # Create plotly figure for activation scales
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Sort layer names to plot in order
        layer_names = sorted(activation_scales.keys())
        x = list(range(len(layer_names)))

        # Plot mean activations
        means = [activation_scales[name]["mean"] for name in layer_names]
        fig.add_trace(
            go.Scatter(
                x=x, y=means, name=f"Mean (width={width})", mode="lines+markers"
            ),
            secondary_y=False,
        )

        # Plot std activations on secondary axis
        stds = [activation_scales[name]["std"] for name in layer_names]
        fig.add_trace(
            go.Scatter(x=x, y=stds, name=f"Std (width={width})", mode="lines+markers"),
            secondary_y=True,
        )

        # Update layout
        fig.update_layout(
            title=f"Activation Statistics (width={width})",
            xaxis_title="Layer",
            xaxis_ticktext=layer_names,
            xaxis_tickangle=45,
            xaxis_tickmode="array",
            xaxis_tickvals=x,
            showlegend=True,
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="Mean Activation", secondary_y=False)
        fig.update_yaxes(title_text="Activation Std Dev", secondary_y=True)

        # Save plot
        import os

        os.makedirs("plots", exist_ok=True)
        fig.write_html(f"plots/activation_stats_width{width}.html")

        activation_scales = {}


if __name__ == "__main__":
    # test_activation_scales()
    test_model()
    # test_repa_loss()
