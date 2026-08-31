"""DINOv3-SAT patch-token extraction for 512px remote-sensing chips."""
import os
import sys

import numpy as np
import torch


DINO_SAT = "facebook/dinov3-vitl16-pretrain-sat493m"
SAT_MEAN = (0.430, 0.411, 0.296)
SAT_STD = (0.213, 0.156, 0.143)


class DINOv3Patch32:
    def __init__(self, model_id=DINO_SAT, device=None, batch=None, backend=None):
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = int(batch or os.environ.get("NAIP_EMBED_BATCH", 2))
        self.backend = backend or os.environ.get("DINOV3_BACKEND", "transformers")
        if self.backend == "hub":
            self.mdl = self._load_hub().eval().to(self.device)
            self.mean = np.array(SAT_MEAN, np.float32)[:, None, None]
            self.std = np.array(SAT_STD, np.float32)[:, None, None]
        elif self.backend == "transformers":
            from transformers import AutoModel
            self.mdl = AutoModel.from_pretrained(model_id).eval().to(self.device)
            self.nreg = self.mdl.config.num_register_tokens
            self.mean = np.array(SAT_MEAN, np.float32)[:, None, None]
            self.std = np.array(SAT_STD, np.float32)[:, None, None]
        else:
            raise ValueError("DINOV3_BACKEND must be 'transformers' or 'hub'")

    def _load_hub(self):
        repo = os.environ.get("DINOV3_REPO", "facebookresearch/dinov3")
        source = "local" if os.path.isdir(repo) else "github"
        model = os.environ.get("DINOV3_MODEL", "dinov3_vitl16")
        weights = os.environ.get("DINOV3_WEIGHTS", "SAT493M")
        if source == "local" and weights in {"SAT493M", "LVD1689M"}:
            sys.path.insert(0, repo)
            from dinov3.hub.backbones import Weights
            weights = getattr(Weights, weights)
        return torch.hub.load(repo, model, source=source, weights=weights)

    @torch.no_grad()
    def patch32(self, ims):
        """list/array of RGB [3,512,512] uint8 -> [N,32,32,1024] float32."""
        x = np.stack([(im.astype(np.float32) / 255.0 - self.mean) / self.std for im in ims])
        out = []
        use_amp = self.device.startswith("cuda")
        for i in range(0, len(x), self.batch):
            xt = torch.tensor(x[i:i + self.batch], dtype=torch.float32, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                h = self._patch_tokens(xt)
            if h.shape[1:] != (1024, 1024):
                raise RuntimeError(f"expected DINOv3 patch tokens [N,1024,1024], got {tuple(h.shape)}")
            out.append(h.float().reshape(h.shape[0], 32, 32, 1024).cpu().numpy())
        return np.concatenate(out)

    def _patch_tokens(self, xt):
        if self.backend == "hub":
            return self.mdl.forward_features(xt)["x_norm_patchtokens"]
        return self.mdl(pixel_values=xt).last_hidden_state[:, 1 + self.nreg:]

    def pool(self, ims):
        patch = self.patch32(ims)
        return patch.reshape(patch.shape[0], -1, patch.shape[-1]).mean(1)
