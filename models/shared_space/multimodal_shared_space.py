"""
multimodal_shared_space.py
Entry point for fusing DeepEarth V‑JEPA vision backbone and DeepSeek‑V3 text backbone into a shared latent space.

Repository layout after the recent restructure **no longer has the `src/deepearth/` prefix** – the code now lives directly at the repo root.  Therefore:
 - Vision backbone is imported from `encoders.vision.v_jepa`.
 - Language backbone is imported from `encoders.text.deepseek_v3`.

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoders.vision.vjepa2_extractor import VJEPA2Extractor as VJEPAEncoder
# Language encoder
from encoders.language.deepseek_v3_encoder import DeepSeekV3Encoder as DeepSeekEncoder

# ------------------------------
# Utility layers
# ------------------------------

class ProjectionHead(nn.Module):
    """Linear projection to align modality‑specific hidden size to global *d_model*."""
    def __init__(self, in_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02 / (in_dim ** 0.5))  # near‑identity

    def forward(self, x):  # x: (B, N, in_dim)
        return self.dropout(self.ln(self.proj(x)))


class SharedLatentPool(nn.Module):
    """Concatenate projected vision & text tokens and run them through a Transformer encoder."""
    def __init__(self, d_model: int = 1024, depth: int = 12, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.vision_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, vision, text, *, return_sequence: bool = False):
        B = vision.size(0)
        seq = torch.cat([
            self.vision_token.expand(B, -1, -1),
            self.text_token.expand(B, -1, -1),
            vision,
            text,
        ], dim=1)  # (B, 2+N_v+N_t, d_model)
        seq = self.transformer(seq)
        v_pool, t_pool = seq[:, 0], seq[:, 1]
        if return_sequence:
            return v_pool, t_pool, seq
        return v_pool, t_pool


def clip_contrastive_loss(v_emb, t_emb, *, temperature: float = 0.07):
    v = F.normalize(v_emb, dim=-1)
    t = F.normalize(t_emb, dim=-1)
    logits = v @ t.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


class MultiModalSharedSpace(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        vision_hidden_size: int,
        text_hidden_size: int,
        *,
        d_model: int = 1024,
        depth: int = 12,
        heads: int = 8,
        freeze_backbones: bool = True,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        if freeze_backbones:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        self.proj_v = ProjectionHead(vision_hidden_size, d_model)
        self.proj_t = ProjectionHead(text_hidden_size, d_model)
        self.shared_pool = SharedLatentPool(d_model, depth, heads)
        self.pixel_head = nn.Linear(d_model, vision_hidden_size)
        self.token_head = nn.Linear(d_model, text_hidden_size)

    def forward(self, images, input_ids, attention_mask=None, *, compute_recon: bool = False):
        v_tokens = self.vision_encoder(images)                            # (B, N_v, D_v)
        t_tokens = self.text_encoder(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     output_hidden_states=True).last_hidden_state  # (B, N_t, D_t)
        v_proj = self.proj_v(v_tokens)                                    # (B, N_v, d_model)
        t_proj = self.proj_t(t_tokens)                                    # (B, N_t, d_model)
        v_pool, t_pool, seq = self.shared_pool(v_proj, t_proj, return_sequence=True)

        out = {"vision_pool": v_pool, "text_pool": t_pool}
        if compute_recon:
            n_v = v_tokens.size(1)
            out["vision_recon"] = self.pixel_head(seq[:, 2:2 + n_v])
            out["text_recon"] = self.token_head(seq[:, 2 + n_v:])
        return out


def training_step(model: MultiModalSharedSpace, batch, *, mae_w: float = 1.0, clip_w: float = 1.0, temp: float = 0.07):
    out = model(batch["images"], batch["input_ids"], batch["attention_mask"], compute_recon=True)
    loss_clip = clip_contrastive_loss(out["vision_pool"], out["text_pool"], temperature=temp)

    v_pred = out["vision_recon"][batch["vision_mask"]]
    v_tgt = model.vision_encoder.pixel_target(batch["images"], batch["vision_mask"])
    loss_v = F.mse_loss(v_pred, v_tgt)

    t_pred = out["text_recon"][batch["text_mask"]]
    t_tgt = batch["input_ids"][batch["text_mask"]]
    loss_t = F.cross_entropy(t_pred.reshape(-1, t_pred.size(-1)), t_tgt.reshape(-1), ignore_index=-100)

    total = clip_w * loss_clip + mae_w * (loss_v + loss_t)
    return total, {"total": total.detach(), "clip": loss_clip.detach(), "recon_v": loss_v.detach(), "recon_t": loss_t.detach()}


def build_model(d_model: int = 1024, *, freeze_backbones: bool = True):
    """Return a ready-to-train multimodal model that matches the repo’s current API."""
    # Vision ───────────────────────────────────────────────
    v_enc = VJEPAEncoder()                  # ← no `pretrained`, no `return_patches`
    # expose its hidden width for the projection head
    vision_hidden = getattr(
        v_enc,                              # try common attribute names
        "hidden_size",
        getattr(v_enc, "embed_dim", None),
    )
    if vision_hidden is None:
        raise AttributeError(
            "Could not infer the vision hidden size. "
            "Open encoders/vision/vjepa2_extractor.py and either: "
            "(a) add `self.hidden_size = <width>` inside __init__, "
            "or (b) tell me the attribute name so I can patch this line."
        )

    # Language ─────────────────────────────────────────────
    t_enc = DeepSeekEncoder(out_dim=d_model, freeze=True)
    # make sure the wrapper exposes its width
    text_hidden = getattr(t_enc, "hidden_size", d_model)

    return MultiModalSharedSpace(
        vision_encoder=v_enc,
        text_encoder=t_enc,
        vision_hidden_size=vision_hidden,
        text_hidden_size=text_hidden,
        d_model=d_model,
        depth=12,
        heads=8,
        freeze_backbones=freeze_backbones,
    )


if __name__ == "__main__":
    imgs = torch.randn(2, 3, 224, 224)
    ids = torch.randint(0, 1000, (2, 32))
    mask = torch.ones_like(ids)
    mdl = build_model()
    batch = {
        "images": imgs,
        "input_ids": ids,
        "attention_mask": mask,
        "vision_mask": torch.zeros(2, mdl.vision_encoder.num_patches, dtype=torch.bool),
        "text_mask": torch.zeros_like(ids, dtype=torch.bool),
    }
    with torch.no_grad():
        pools = mdl(batch["images"], batch["input_ids"], batch["attention_mask"])
        print(pools["vision_pool"].shape, pools["text_pool"].shape)  # → torch.Size([2, 1024]) torch.Size([2, 1024])

