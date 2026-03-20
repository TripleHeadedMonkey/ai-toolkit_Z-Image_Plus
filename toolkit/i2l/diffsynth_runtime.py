from __future__ import annotations

import os
from typing import List


def generate_zimage_i2l_lora(
    image_paths: List[str],
    output_path: str,
    i2l_model_name_or_path: str = "DiffSynth-Studio/Z-Image-i2L",
    encoders_name_or_path: str = "DiffSynth-Studio/General-Image-Encoders",
    base_model_name_or_path: str = "Tongyi-MAI/Z-Image",
    turbo_model_name_or_path: str = "Tongyi-MAI/Z-Image-Turbo",
    device: str = "cuda",
):
    try:
        import torch
        from PIL import Image
        from safetensors.torch import save_file
        from diffsynth.pipelines.z_image import (
            ZImagePipeline,
            ModelConfig,
            ZImageUnit_Image2LoRAEncode,
            ZImageUnit_Image2LoRADecode,
        )
    except Exception as e:
        raise ImportError(
            "DiffSynth runtime dependencies are required for on-the-fly I2L generation. "
            "Install DiffSynth-Studio in this environment or provide train.zimage_init_lora_path."
        ) from e

    if len(image_paths) == 0:
        raise ValueError("No image paths provided for I2L generation.")

    if output_path is None:
        raise ValueError("output_path is required for I2L generation.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": device,
        "onload_dtype": torch.bfloat16,
        "onload_device": device,
        "preparing_dtype": torch.bfloat16,
        "preparing_device": device,
        "computation_dtype": torch.bfloat16,
        "computation_device": device,
    }

    pipe = ZImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(model_id=base_model_name_or_path, origin_file_pattern="transformer/config.json"),
            ModelConfig(
                model_id=base_model_name_or_path,
                origin_file_pattern="transformer/diffusion_pytorch_model.safetensors.index.json",
            ),
            ModelConfig(model_id=base_model_name_or_path, origin_file_pattern="transformer/*.safetensors", **vram_config),
            ModelConfig(model_id=turbo_model_name_or_path, origin_file_pattern="transformer/config.json"),
            ModelConfig(
                model_id=turbo_model_name_or_path,
                origin_file_pattern="transformer/diffusion_pytorch_model.safetensors.index.json",
            ),
            ModelConfig(model_id=turbo_model_name_or_path, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=turbo_model_name_or_path, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=turbo_model_name_or_path, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ModelConfig(model_id=encoders_name_or_path, origin_file_pattern="SigLIP2-G384/model.safetensors"),
            ModelConfig(model_id=encoders_name_or_path, origin_file_pattern="DINOv3-7B/model.safetensors"),
            ModelConfig(model_id=i2l_model_name_or_path, origin_file_pattern="model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id=turbo_model_name_or_path, origin_file_pattern="tokenizer/"),
    )

    images = [Image.open(path).convert("RGB") for path in image_paths]

    with torch.no_grad():
        embs = ZImageUnit_Image2LoRAEncode().process(pipe, image2lora_images=images)
        lora = ZImageUnit_Image2LoRADecode().process(pipe, **embs)["lora"]

    save_file(lora, output_path)
    return output_path