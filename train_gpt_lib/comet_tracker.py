from __future__ import annotations
# pyright: reportMissingImports=false

import json
import math
import os
from typing import Any

from .config import Hyperparameters


class CometTracker:
    def __init__(self, args: Hyperparameters, enabled: bool, log0):
        self.enabled = enabled
        self.log0 = log0
        self.experiment: Any | None = None
        if not enabled:
            return
        api_key = (args.comet_api_key or os.environ.get("COMET_API_KEY", "")).strip()
        if not api_key:
            self.log0(
                "comet:disabled reason=no_api_key "
                "(export COMET_API_KEY before torchrun, or use launch_grid_generated/.comet_api_key — see check_model_size.ipynb)"
            )
            self.enabled = False
            return
        try:
            from comet_ml import Experiment
        except Exception as exc:  # pragma: no cover
            self.log0(f"comet:disabled reason=import_error detail={exc}")
            self.enabled = False
            return
        try:
            kwargs: dict[str, Any] = {
                "project_name": args.comet_project_name,
                "api_key": api_key,
            }
            if args.comet_workspace:
                kwargs["workspace"] = args.comet_workspace
            self.experiment = Experiment(**kwargs)
            run_name = args.experiment_name if args.experiment_name else args.run_id
            self.experiment.set_name(run_name)
            params = args.as_dict()
            self.experiment.log_parameters(params)
            self.experiment.log_asset_data(
                json.dumps(params, indent=2, sort_keys=True),
                name="hyperparameters.json",
                overwrite=True,
            )
            self.log0("comet:enabled")
        except Exception as exc:  # pragma: no cover
            self.log0(f"comet:disabled reason=init_error detail={exc}")
            self.enabled = False
            self.experiment = None

    def log_train(self, step: int, train_loss: float) -> None:
        if not self.enabled or self.experiment is None:
            return
        self.experiment.log_metric("train_loss", train_loss, step=step)

    def log_val(
        self,
        step: int,
        val_loss_full_precision: float,
        val_bpb_full_precision: float,
        val_loss_quantized_model: float,
        val_bpb_quantized_model: float,
    ) -> None:
        """Log validation on full-precision weights and on int8-quantized→dequantized weights."""
        if not self.enabled or self.experiment is None:
            return
        # Metric name suffix: full precision vs quantized (dequantized) submission path.
        self.experiment.log_metric("val_loss_full_precision", val_loss_full_precision, step=step)
        self.experiment.log_metric("val_bpb_full_precision", val_bpb_full_precision, step=step)
        self.experiment.log_metric("val_loss_quantized_model", val_loss_quantized_model, step=step)
        self.experiment.log_metric("val_bpb_quantized_model", val_bpb_quantized_model, step=step)
        # Aliases for dashboards that expect a single val_* (default: full precision).
        self.experiment.log_metric("val_loss", val_loss_full_precision, step=step)
        self.experiment.log_metric("val_bpb", val_bpb_full_precision, step=step)
        self.experiment.log_metric("val_ppb", val_bpb_full_precision, step=step)

    def log_model_size(
        self,
        *,
        fp32_pt_bytes: int,
        int8_zlib_bytes: int,
        int8_payload_bytes: int,
    ) -> None:
        """Log submission-relevant size (int8+zlib file) as primary ``model_size_*`` metrics."""
        if not self.enabled or self.experiment is None:
            return
        mib = 1024.0 * 1024.0
        z_mb_ceil = int(math.ceil(int8_zlib_bytes / mib))
        pt_mb_ceil = int(math.ceil(fp32_pt_bytes / mib))
        pay_mb_ceil = int(math.ceil(int8_payload_bytes / mib))
        # Primary: то же, что файл ``final_model.int8.ptz`` (сабмит после сжатия)
        self.experiment.log_metric("model_size_bytes", float(int8_zlib_bytes), step=0)
        self.experiment.log_metric("model_size_mb_ceil", float(z_mb_ceil), step=0)
        self.experiment.log_metric("int8_payload_bytes", float(int8_payload_bytes), step=0)
        self.experiment.log_metric("int8_payload_mb_ceil", float(pay_mb_ceil), step=0)
        self.experiment.log_metric("model_fp32_pt_bytes", float(fp32_pt_bytes), step=0)
        self.experiment.log_metric("model_fp32_pt_mb_ceil", float(pt_mb_ceil), step=0)

    def end(self) -> None:
        if not self.enabled or self.experiment is None:
            return
        self.experiment.end()
