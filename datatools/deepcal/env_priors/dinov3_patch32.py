"""DINOv3-SAT patch-token extraction for 512px remote-sensing chips."""
import os

import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel


DINO_SAT = "facebook/dinov3-vitl16-pretrain-sat493m"


class DINOv3Patch32:
    def __init__(self, model_id=DINO_SAT, device=None, batch=None):
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = int(batch or os.environ.get("NAIP_EMBED_BATCH", 2))
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.mdl = AutoModel.from_pretrained(model_id).eval().to(self.device)
        self.nreg = self.mdl.config.num_register_tokens
        self.mean = np.array(self.proc.image_mean, np.float32)[:, None, None]
        self.std = np.array(self.proc.image_std, np.float32)[:, None, None]

    @torch.no_grad()
    def patch32(self, ims):
        """list/array of RGB [3,512,512] uint8 -> [N,32,32,1024] float32."""
        x = np.stack([(im.astype(np.float32) / 255.0 - self.mean) / self.std for im in ims])
        out = []
        use_amp = self.device.startswith("cuda")
        for i in range(0, len(x), self.batch):
            xt = torch.tensor(x[i:i + self.batch], dtype=torch.float32, device=self.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                h = self.mdl(pixel_values=xt).last_hidden_state[:, 1 + self.nreg:]
            if h.shape[1:] != (1024, 1024):
                raise RuntimeError(f"expected DINOv3 patch tokens [N,1024,1024], got {tuple(h.shape)}")
            out.append(h.float().reshape(h.shape[0], 32, 32, 1024).cpu().numpy())
        return np.concatenate(out)

    def pool(self, ims):
        patch = self.patch32(ims)
        return patch.reshape(patch.shape[0], -1, patch.shape[-1]).mean(1)
