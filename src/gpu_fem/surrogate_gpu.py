"""
surrogate_gpu.py
----------------
GPU-accelerated online-learning surrogate for element sensitivity prediction.

This module now supports two surrogate modes:
  - basic: per-element MLP on local state, matching the original TO3D surrogate
  - temporal_sr: a DLSS-like temporal super-resolution residual corrector that
    uses the previous exact FEM sensitivity field as an anchor and learns the
    fine-grid residual on top of coarse pooled context

Extends SensitivitySurrogate (TO3D) with:
  - CUDA device by default (falls back to CPU automatically)
  - predict_gpu(): returns torch.Tensor on device - no CPU round-trip
  - Larger MLP option for high-fidelity at 100k+ elements
  - Batched prediction for meshes too large to fit in VRAM at once

Features per element:
  - basic: [rho_phys_e, rho_f_e, penal, iter_ratio]
  - temporal_sr: [rho_phys_e, rho_f_e, anchor_dc_e, anchor_low_e,
                  anchor_detail_e, penal, iter_ratio]
Target:
  - basic: dc_phys
  - temporal_sr: residual correction to the previous exact FEM sensitivity
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Network definitions
# ─────────────────────────────────────────────────────────────────────────────

if _TORCH_AVAILABLE:
    def _make_mlp(in_features: int, hidden: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_features
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.SiLU())
            prev = width
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    class _SensNet(nn.Module):
        """Standard MLP: 4 -> 64 -> 64 -> 1."""

        def __init__(self):
            super().__init__()
            self.net = _make_mlp(4, [64, 64])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    class _SensNetLarge(nn.Module):
        """Larger MLP: 4 -> 128 -> 128 -> 64 -> 1."""

        def __init__(self):
            super().__init__()
            self.net = _make_mlp(4, [128, 128, 64])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    class _SensNetTemporalSR(nn.Module):
        """Temporal super-resolution MLP: 7 -> 96 -> 96 -> 1."""

        def __init__(self):
            super().__init__()
            self.net = _make_mlp(7, [96, 96])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)

    class _SensNetTemporalSRLarge(nn.Module):
        """Temporal SR large MLP: 7 -> 160 -> 160 -> 96 -> 1."""

        def __init__(self):
            super().__init__()
            self.net = _make_mlp(7, [160, 160, 96])

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# GPU Surrogate
# ─────────────────────────────────────────────────────────────────────────────

class SensitivitySurrogateGPU:
    """
    GPU-accelerated online surrogate for element sensitivity prediction.

    Parameters
    ----------
    device : str
        "cuda" (default) or "cpu".
    n_ensemble : int
        Number of MLP members.
    use_large_net : bool
        If True, use _SensNetLarge (recommended for n_elem > 50k).
    gpu_batch_size : int
        Predict in chunks of this size to avoid VRAM overflow on large meshes.
    buffer_size, min_obs, subsample_stride, lr, physics_loss_weight :
        Same as SensitivitySurrogate (TO3D).
    """

    def __init__(
        self,
        device: str = "cuda",
        n_ensemble: int = 3,
        use_large_net: bool = False,
        gpu_batch_size: int = 4096,
        mesh_shape: Optional[tuple[int, int, int]] = None,
        feature_mode: str = "basic",
        coarse_factor: int = 2,
        buffer_size: int = 16000,
        min_obs: int = 200,
        subsample_stride: int = 4,
        lr: float = 1e-3,
        physics_loss_weight: float = 0.1,
        freeze_after_obs: Optional[int] = None,
    ) -> None:
        self.n_ensemble = n_ensemble
        self.gpu_batch_size = gpu_batch_size
        self.mesh_shape = mesh_shape
        self.coarse_factor = max(1, int(coarse_factor))
        self.feature_mode = self._resolve_feature_mode(feature_mode, mesh_shape, use_large_net)
        self.buffer: deque[tuple[np.ndarray, float]] = deque(maxlen=buffer_size)
        self.min_obs = min_obs
        self.subsample_stride = subsample_stride
        self.uncertainty: float = 0.0
        self._ready: bool = False
        self._n_fem_obs: int = 0
        self.physics_loss_weight = physics_loss_weight
        self.freeze_after_obs = freeze_after_obs
        self._anchor_dc: Optional[np.ndarray] = None

        if _TORCH_AVAILABLE:
            # Resolve device
            if device == "cuda" and not torch.cuda.is_available():
                print("[SensitivitySurrogateGPU] CUDA not available, using CPU.")
                device = "cpu"
            self._device = torch.device(device)
            if self.feature_mode == "temporal_sr":
                NetClass = _SensNetTemporalSRLarge if use_large_net else _SensNetTemporalSR
            else:
                NetClass = _SensNetLarge if use_large_net else _SensNet
            self._nets = [NetClass().to(self._device) for _ in range(n_ensemble)]
            self._opts = [
                torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
                for net in self._nets
            ]
        else:
            self._device = None
            self._nets = None
            self._opts = None

    @staticmethod
    def _resolve_feature_mode(
        feature_mode: str,
        mesh_shape: Optional[tuple[int, int, int]],
        use_large_net: bool,
    ) -> str:
        if feature_mode != "auto":
            return feature_mode
        if mesh_shape is not None:
            nelx, nely, nelz = mesh_shape
            n_elem = nelx * nely * max(1, nelz)
            if n_elem >= 50_000 or use_large_net:
                return "temporal_sr"
        return "basic"

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def build_features(
        rho_phys: np.ndarray,
        rho_f: np.ndarray,
        penal: float,
        iter_ratio: float,
    ) -> np.ndarray:
        """Build (n_elem, 4) feature matrix."""
        n = len(rho_phys)
        return np.stack([
            rho_phys.astype(np.float32),
            rho_f.astype(np.float32),
            np.full(n, penal, dtype=np.float32),
            np.full(n, iter_ratio, dtype=np.float32),
        ], axis=1)

    def _anchor_or_zeros(self, n: int) -> np.ndarray:
        if self._anchor_dc is None or len(self._anchor_dc) != n:
            return np.zeros(n, dtype=np.float32)
        return self._anchor_dc.astype(np.float32, copy=False)

    def _build_temporal_features_np(
        self,
        rho_phys: np.ndarray,
        rho_f: np.ndarray,
        penal: float,
        iter_ratio: float,
    ) -> np.ndarray:
        anchor = self._anchor_or_zeros(len(rho_phys))
        anchor_low = self._coarse_to_fine_numpy(anchor)
        anchor_detail = anchor - anchor_low
        return np.stack([
            rho_phys.astype(np.float32),
            rho_f.astype(np.float32),
            anchor.astype(np.float32),
            anchor_low.astype(np.float32),
            anchor_detail.astype(np.float32),
            np.full(len(rho_phys), penal, dtype=np.float32),
            np.full(len(rho_phys), iter_ratio, dtype=np.float32),
        ], axis=1)

    def _build_temporal_features_torch(
        self,
        rho_phys_t: "torch.Tensor",
        rho_f_t: "torch.Tensor",
        penal: float,
        iter_ratio: float,
    ) -> "torch.Tensor":
        anchor = self._anchor_dc
        if anchor is None or len(anchor) != rho_phys_t.shape[0]:
            anchor_t = torch.zeros_like(rho_phys_t, dtype=torch.float32)
        else:
            anchor_t = torch.tensor(anchor, dtype=torch.float32, device=self._device)
        anchor_low = self._coarse_to_fine_torch(anchor_t)
        anchor_detail = anchor_t - anchor_low
        penal_t = torch.full_like(rho_phys_t, float(penal), dtype=torch.float32)
        ratio_t = torch.full_like(rho_phys_t, float(iter_ratio), dtype=torch.float32)
        return torch.stack([
            rho_phys_t.float(),
            rho_f_t.float(),
            anchor_t,
            anchor_low,
            anchor_detail,
            penal_t,
            ratio_t,
        ], dim=1)

    def _coarse_to_fine_numpy(self, field: np.ndarray) -> np.ndarray:
        if self.mesh_shape is None:
            return field.astype(np.float32, copy=False)
        nelx, nely, nelz = self.mesh_shape
        if nelz and nelz > 1:
            arr = field.reshape(nelx, nely, nelz)
            fx = fy = fz = self.coarse_factor
            px = (-nelx) % fx
            py = (-nely) % fy
            pz = (-nelz) % fz
            if px or py or pz:
                arr = np.pad(arr, ((0, px), (0, py), (0, pz)), mode="edge")
            cx, cy, cz = arr.shape[0] // fx, arr.shape[1] // fy, arr.shape[2] // fz
            coarse = arr.reshape(cx, fx, cy, fy, cz, fz).mean(axis=(1, 3, 5))
            up = np.repeat(np.repeat(np.repeat(coarse, fx, axis=0), fy, axis=1), fz, axis=2)
            return up[:nelx, :nely, :nelz].reshape(-1).astype(np.float32, copy=False)
        arr = field.reshape(nelx, nely)
        fx = fy = self.coarse_factor
        px = (-nelx) % fx
        py = (-nely) % fy
        if px or py:
            arr = np.pad(arr, ((0, px), (0, py)), mode="edge")
        cx, cy = arr.shape[0] // fx, arr.shape[1] // fy
        coarse = arr.reshape(cx, fx, cy, fy).mean(axis=(1, 3))
        up = np.repeat(np.repeat(coarse, fx, axis=0), fy, axis=1)
        return up[:nelx, :nely].reshape(-1).astype(np.float32, copy=False)

    def _coarse_to_fine_torch(self, field: "torch.Tensor") -> "torch.Tensor":
        if self.mesh_shape is None:
            return field.float()
        nelx, nely, nelz = self.mesh_shape
        if nelz and nelz > 1:
            x = field.float().reshape(1, 1, nelx, nely, nelz)
            pooled = F.avg_pool3d(x, kernel_size=self.coarse_factor, stride=self.coarse_factor, ceil_mode=True)
            up = F.interpolate(pooled, size=(nelx, nely, nelz), mode="trilinear", align_corners=False)
            return up.reshape(-1)
        x = field.float().reshape(1, 1, nelx, nely)
        pooled = F.avg_pool2d(x, kernel_size=self.coarse_factor, stride=self.coarse_factor, ceil_mode=True)
        up = F.interpolate(pooled, size=(nelx, nely), mode="bilinear", align_corners=False)
        return up.reshape(-1)

    def _residual_target(self, dc_phys: np.ndarray) -> np.ndarray:
        if self.feature_mode == "temporal_sr" and self._anchor_dc is not None and len(self._anchor_dc) == len(dc_phys):
            return (dc_phys.astype(np.float32) - self._anchor_dc.astype(np.float32))
        return dc_phys.astype(np.float32)

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update(
        self,
        rho_phys: np.ndarray,
        rho_f: np.ndarray,
        penal: float,
        iter_ratio: float,
        dc_phys: np.ndarray,
        n_train_steps: int = 8,
    ) -> None:
        if not _TORCH_AVAILABLE or self._nets is None:
            return
        if self.freeze_after_obs is not None and self._n_fem_obs >= self.freeze_after_obs:
            return

        if self.feature_mode == "temporal_sr":
            X = self._build_temporal_features_np(rho_phys, rho_f, penal, iter_ratio)
            Y = self._residual_target(dc_phys)
        else:
            X = self.build_features(rho_phys, rho_f, penal, iter_ratio)
            Y = dc_phys.astype(np.float32)

        idx = np.arange(0, len(X), self.subsample_stride)
        for i in idx:
            self.buffer.append((X[i], Y[i]))

        self._n_fem_obs += 1
        if len(self.buffer) < self.min_obs:
            self._anchor_dc = dc_phys.astype(np.float32).copy()
            return

        self._ready = True
        self._train(n_train_steps)
        self._anchor_dc = dc_phys.astype(np.float32).copy()

    def _train(self, n_steps: int) -> None:
        batch_size = min(512, len(self.buffer))
        for _ in range(n_steps):
            batch = random.sample(self.buffer, batch_size)
            Xb = torch.tensor(
                np.array([b[0] for b in batch]), device=self._device
            )
            Yb = torch.tensor(
                np.array([b[1] for b in batch]), device=self._device
            )
            for net, opt in zip(self._nets, self._opts):
                pred = net(Xb)
                loss = nn.functional.mse_loss(pred, Yb)

                if self.physics_loss_weight > 0.0:
                    total_pred = pred + Xb[:, 2] if self.feature_mode == "temporal_sr" else pred
                    L_sign = nn.functional.relu(total_pred).mean()
                    Xb_r = Xb.detach().requires_grad_(True)
                    pred_r = net(Xb_r)
                    if self.feature_mode == "temporal_sr":
                        pred_r = pred_r + Xb_r[:, 2]
                    grad_rho = torch.autograd.grad(pred_r.sum(), Xb_r, create_graph=True)[0][:, 0]
                    L_mono = nn.functional.relu(grad_rho).mean()
                    loss = loss + self.physics_loss_weight * (L_sign + L_mono)

                opt.zero_grad()
                loss.backward()
                opt.step()

    # ------------------------------------------------------------------
    # Prediction (numpy interface — same as TO3D surrogate)
    # ------------------------------------------------------------------

    def predict(
        self,
        rho_phys: np.ndarray,
        rho_f: np.ndarray,
        penal: float,
        iter_ratio: float,
    ) -> tuple[Optional[np.ndarray], float]:
        """Predict dc_phys. Returns (prediction, uncertainty). prediction is None if not ready."""
        if not self._ready or not _TORCH_AVAILABLE or self._nets is None:
            return None, 1.0

        if self.feature_mode == "temporal_sr":
            X = self._build_temporal_features_np(rho_phys, rho_f, penal, iter_ratio)
        else:
            X = self.build_features(rho_phys, rho_f, penal, iter_ratio)
        result_gpu = self.predict_gpu_from_numpy(X)
        preds_arr = result_gpu  # (n_ensemble, n_elem)
        mean_pred = preds_arr.mean(axis=0)
        std_pred = preds_arr.std(axis=0)

        denom = np.abs(mean_pred).mean()
        uncertainty = float(std_pred.mean() / max(denom, 1e-10))
        self.uncertainty = uncertainty

        if self.feature_mode == "temporal_sr" and self._anchor_dc is not None and len(self._anchor_dc) == len(mean_pred):
            mean_pred = mean_pred + self._anchor_dc.astype(np.float32)

        return mean_pred, uncertainty

    def predict_gpu_from_numpy(self, X: np.ndarray) -> np.ndarray:
        """
        Predict from numpy feature matrix X (n_elem, 4).
        Returns (n_ensemble, n_elem) numpy array.
        Runs in GPU batches of size gpu_batch_size.
        """
        n_elem = len(X)
        preds = [np.empty(n_elem, dtype=np.float32) for _ in range(self.n_ensemble)]
        batch_size = self.gpu_batch_size

        with torch.no_grad():
            for start in range(0, n_elem, batch_size):
                end = min(start + batch_size, n_elem)
                Xb = torch.tensor(X[start:end], device=self._device)
                for k, net in enumerate(self._nets):
                    preds[k][start:end] = net(Xb).cpu().numpy()

        return np.stack(preds, axis=0)  # (n_ensemble, n_elem)

    def predict_gpu(
        self,
        rho_phys_t: "torch.Tensor",
        rho_f_t: "torch.Tensor",
        penal: float,
        iter_ratio: float,
    ) -> "torch.Tensor":
        """
        GPU-native predict: input and output are CUDA tensors.
        No CPU round-trip — for use in fully GPU-resident SIMP loop.

        Returns mean_dc_phys tensor on same device as inputs.
        Also updates self.uncertainty (scalar, CPU).
        """
        if not self._ready or not _TORCH_AVAILABLE or self._nets is None:
            return None

        n = rho_phys_t.shape[0]
        if self.feature_mode == "temporal_sr":
            X = self._build_temporal_features_torch(rho_phys_t, rho_f_t, penal, iter_ratio)
        else:
            penal_t = torch.full((n,), penal, dtype=torch.float32, device=self._device)
            ratio_t = torch.full((n,), iter_ratio, dtype=torch.float32, device=self._device)
            X = torch.stack([
                rho_phys_t.float(),
                rho_f_t.float(),
                penal_t,
                ratio_t,
            ], dim=1)  # (n_elem, 4)

        preds = []
        batch_size = self.gpu_batch_size
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb = X[start:end]
                batch_preds = torch.stack([net(Xb) for net in self._nets], dim=0)  # (n_ens, batch)
                preds.append(batch_preds)

        preds_t = torch.cat(preds, dim=1)  # (n_ensemble, n_elem)
        mean_pred = preds_t.mean(dim=0)    # (n_elem,)
        std_pred = preds_t.std(dim=0)

        denom = mean_pred.abs().mean()
        uncertainty = float((std_pred.mean() / (denom + 1e-10)).cpu())
        self.uncertainty = uncertainty

        if self.feature_mode == "temporal_sr" and self._anchor_dc is not None and len(self._anchor_dc) == mean_pred.shape[0]:
            anchor = torch.tensor(self._anchor_dc, dtype=mean_pred.dtype, device=mean_pred.device)
            mean_pred = mean_pred + anchor

        return mean_pred  # stays on GPU

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def n_fem_observations(self) -> int:
        return self._n_fem_obs

    @property
    def torch_available(self) -> bool:
        return _TORCH_AVAILABLE

    @property
    def device(self) -> str:
        if self._device is None:
            return "cpu"
        return str(self._device)
