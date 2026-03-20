from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from safetensors.torch import load_file


@dataclass
class I2LValidationReport:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _normalize_key(key: str) -> str:
    # DiffSynth-style exported keys often use diffusion_model.*
    if key.startswith("diffusion_model."):
        return key.replace("diffusion_model.", "transformer.", 1)
    return key


def validate_lora_for_network(lora_path: str, network) -> I2LValidationReport:
    report = I2LValidationReport(is_valid=True)

    lora_sd = load_file(lora_path)
    net_sd = network.state_dict()

    has_lora_pair = False
    for key, tensor in lora_sd.items():
        mapped_key = _normalize_key(key)
        if ".lora_A." in mapped_key or ".lora_B." in mapped_key:
            has_lora_pair = True

        if mapped_key not in net_sd:
            report.warnings.append(f"Key missing in target network: {mapped_key}")
            continue

        tgt = net_sd[mapped_key]
        if tensor.shape != tgt.shape:
            # Network load path can expand/shrink some lora A/B shapes, so keep this as warning.
            report.warnings.append(
                f"Shape mismatch for {mapped_key}: src={tuple(tensor.shape)} tgt={tuple(tgt.shape)}"
            )

    if not has_lora_pair:
        report.is_valid = False
        report.errors.append("No LoRA A/B weights were found in the provided safetensors file.")

    return report
