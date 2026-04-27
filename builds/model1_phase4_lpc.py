from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch
from torch import Tensor


def autocorrelation(x: Tensor, lag: int) -> Tensor:
    if lag <= 0 or lag >= x.numel():
        raise ValueError("lag must be in [1, len(x)-1]")
    x0 = x[:-lag] - x[:-lag].mean()
    x1 = x[lag:] - x[lag:].mean()
    denom = x0.std().clamp_min(1e-6) * x1.std().clamp_min(1e-6)
    return (x0 * x1).mean() / denom


def levinson_durbin(r: Tensor, order: int) -> Tensor:
    a = torch.zeros(order + 1, dtype=r.dtype, device=r.device)
    e = r[0].clamp_min(1e-6)
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = torch.dot(a[1:i], r[1:i].flip(0)) if i > 1 else torch.tensor(0.0, dtype=r.dtype, device=r.device)
        k = -(r[i] + acc) / e
        a_prev = a.clone()
        a[1:i] = a_prev[1:i] + k * a_prev[1:i].flip(0)
        a[i] = k
        e = e * (1 - k * k).clamp_min(1e-6)
    return -a[1:]


@dataclass
class LPCResult:
    coeffs: Tensor
    residual: Tensor


class LinearPredictiveCoder:
    def __init__(self, order: int = 8):
        self.order = order
        self.coeffs: Tensor | None = None

    def fit(self, token_ids: Tensor) -> Tensor:
        x = token_ids.float()
        x = x - x.mean()
        r = torch.stack([(x[: x.numel() - lag] * x[lag:]).mean() for lag in range(self.order + 1)])
        self.coeffs = levinson_durbin(r, self.order)
        return self.coeffs

    def residualize(self, token_ids: Tensor) -> Tensor:
        if self.coeffs is None:
            raise RuntimeError("fit must be called before residualize")
        x = token_ids.float()
        residual = x.clone()
        for idx in range(self.order, x.numel()):
            pred = torch.dot(self.coeffs, x[idx - self.order : idx].flip(0))
            residual[idx] = x[idx] - pred
        return residual

    def fit_transform(self, token_ids: Tensor) -> LPCResult:
        coeffs = self.fit(token_ids)
        return LPCResult(coeffs=coeffs, residual=self.residualize(token_ids))


def verify_lpc() -> dict[str, float]:
    torch.manual_seed(0)
    x = torch.arange(0, 256, dtype=torch.float32)
    signal = x * 0.6 + torch.roll(x, 1) * 0.3 + torch.sin(x / 8.0) * 2.0
    lpc = LinearPredictiveCoder(order=4)
    result = lpc.fit_transform(signal)
    raw_corr = autocorrelation(signal, 1).item()
    residual_corr = autocorrelation(result.residual[4:], 1).item()
    if abs(residual_corr) >= abs(raw_corr):
        raise AssertionError(f"LPC residual did not lower autocorrelation: raw={raw_corr}, residual={residual_corr}")
    return {"raw_autocorr_lag1": raw_corr, "residual_autocorr_lag1": residual_corr}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify LPC preprocessing for Model 1 Codec.")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = verify_lpc()
    print(json.dumps(result))
    if not args.verify_only:
        print("LPC preprocessing verified.")


if __name__ == "__main__":
    main()
