diff --git a/extensions_built_in/sd_trainer/SDTrainer.py b/extensions_built_in/sd_trainer/SDTrainer.py
index f7792117956ff23d424c0f6375bf03a9e015c36f..c82a6b58d4e4a5d8cb2cf112e2f9dc874461e79c 100644
--- a/extensions_built_in/sd_trainer/SDTrainer.py
+++ b/extensions_built_in/sd_trainer/SDTrainer.py
@@ -1,64 +1,67 @@
 import os
 import random
+import json
+import hashlib
 from collections import OrderedDict
 from typing import Union, Literal, List, Optional
 
 import numpy as np
 from diffusers import T2IAdapter, AutoencoderTiny, ControlNetModel
 
 import torch.functional as F
 from safetensors.torch import load_file
 from torch.utils.data import DataLoader, ConcatDataset
 
 from toolkit import train_tools
 from toolkit.basic import value_map, adain, get_mean_std
 from toolkit.clip_vision_adapter import ClipVisionAdapter
 from toolkit.config_modules import GenerateImageConfig
 from toolkit.data_loader import get_dataloader_datasets
 from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
 from toolkit.guidance import get_targeted_guidance_loss, get_guidance_loss, GuidanceType
 from toolkit.image_utils import show_tensors, show_latents
 from toolkit.ip_adapter import IPAdapter
 from toolkit.custom_adapter import CustomAdapter
 from toolkit.print import print_acc
 from toolkit.prompt_utils import PromptEmbeds, concat_prompt_embeds
 from toolkit.reference_adapter import ReferenceAdapter
 from toolkit.stable_diffusion_model import StableDiffusion, BlankNetwork
 from toolkit.train_tools import get_torch_dtype, apply_snr_weight, add_all_snr_to_noise_scheduler, \
     apply_learnable_snr_gos, LearnableSNRGamma
 import gc
 import torch
 from jobs.process import BaseSDTrainProcess
 from torchvision import transforms
 from diffusers import EMAModel
 import math
 from toolkit.train_tools import precondition_model_outputs_flow_match
 from toolkit.models.diffusion_feature_extraction import DiffusionFeatureExtractor, load_dfe
 from toolkit.util.losses import wavelet_loss, stepped_loss
 import torch.nn.functional as F
 from toolkit.unloader import unload_text_encoder
+from toolkit.i2l import validate_lora_for_network, generate_zimage_i2l_lora
 from PIL import Image
 from torchvision.transforms import functional as TF
 
 
 def flush():
     torch.cuda.empty_cache()
     gc.collect()
 
 
 adapter_transforms = transforms.Compose([
     transforms.ToTensor(),
 ])
 
 
 class SDTrainer(BaseSDTrainProcess):
 
     def __init__(self, process_id: int, job, config: OrderedDict, **kwargs):
         super().__init__(process_id, job, config, **kwargs)
         self.assistant_adapter: Union['T2IAdapter', 'ControlNetModel', None]
         self.do_prior_prediction = False
         self.do_long_prompts = False
         self.do_guided_loss = False
         self.taesd: Optional[AutoencoderTiny] = None
 
         self._clip_image_embeds_unconditional: Union[List[str], None] = None
@@ -94,50 +97,259 @@ class SDTrainer(BaseSDTrainProcess):
             if self.network_config is None:
                 raise ValueError("diff_output_preservation requires a network to be set")
             if self.train_config.train_text_encoder:
                 raise ValueError("diff_output_preservation is not supported with train_text_encoder")
         
         if self.train_config.blank_prompt_preservation:
             if self.network_config is None:
                 raise ValueError("blank_prompt_preservation requires a network to be set")
         
         if self.train_config.blank_prompt_preservation or self.train_config.diff_output_preservation:
             # always do a prior prediction when doing output preservation
             self.do_prior_prediction = True
         
         # store the loss target for a batch so we can use it in a loss
         self._guidance_loss_target_batch: float = 0.0
         if isinstance(self.train_config.guidance_loss_target, (int, float)):
             self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target)
         elif isinstance(self.train_config.guidance_loss_target, list):
             self._guidance_loss_target_batch = float(self.train_config.guidance_loss_target[0])
         else:
             raise ValueError(f"Unknown guidance loss target type {type(self.train_config.guidance_loss_target)}")
 
 
     def before_model_load(self):
         pass
+
+    def _maybe_run_zimage_i2l_stage(self):
+        if self.sd is None or getattr(self.sd, "arch", None) != "zimage":
+            return
+
+        do_i2l = bool(self.train_config.zimage_i2l_only or self.train_config.zimage_bootstrap_enabled)
+        if not do_i2l:
+            return
+
+        init_lora_path = self.train_config.zimage_init_lora_path
+        if init_lora_path is None or not os.path.exists(init_lora_path):
+            image_paths = self._collect_zimage_i2l_image_paths()
+            cached_path = self._get_i2l_cached_lora_path(image_paths)
+            if cached_path is not None and os.path.exists(cached_path):
+                init_lora_path = cached_path
+                print_acc(f"Using cached I2L LoRA: {init_lora_path}")
+            else:
+                output_path = self.train_config.zimage_i2l_output_path
+                if output_path is None:
+                    output_path = self._get_i2l_cache_output_path(image_paths)
+
+                i2l_device = getattr(self.train_config, "zimage_i2l_device", "cpu")
+                # free training-model VRAM before running I2L generation
+                self.sd.set_device_state({"device": "cpu", "train_unet": False, "require_grads": False})
+                flush()
+
+                print_acc(f"Generating Z-Image I2L LoRA from {len(image_paths)} image(s)...")
+                try:
+                    init_lora_path = generate_zimage_i2l_lora(
+                        image_paths=image_paths,
+                        output_path=output_path,
+                        i2l_model_name_or_path=self.train_config.zimage_i2l_model_name_or_path,
+                        encoders_name_or_path=self.train_config.zimage_i2l_encoders_name_or_path,
+                        base_model_name_or_path=self.train_config.zimage_i2l_base_model_name_or_path,
+                        turbo_model_name_or_path=self.train_config.zimage_i2l_turbo_model_name_or_path,
+                        device=i2l_device,
+                    )
+                finally:
+                    self.sd.set_device_state(self.train_device_state_preset)
+                    flush()
+                self._record_i2l_cache_entry(image_paths, init_lora_path)
+            self.train_config.zimage_init_lora_path = init_lora_path
+
+        if self.network is None:
+            raise ValueError("Z-Image I2L mode requires a trainable LoRA network to be initialized.")
+
+        report = validate_lora_for_network(init_lora_path, self.network)
+        if not report.is_valid:
+            raise ValueError(f"I2L LoRA validation failed: {'; '.join(report.errors)}")
+
+        if len(report.warnings) > 0:
+            print_acc("I2L compatibility warnings:")
+            for warning in report.warnings[:20]:
+                print_acc(f" - {warning}")
+
+        if self.train_config.zimage_init_strategy != "copy":
+            raise ValueError(
+                f"Unsupported zimage_init_strategy '{self.train_config.zimage_init_strategy}'. "
+                "Currently only 'copy' is implemented."
+            )
+
+        init_scale = float(self.train_config.zimage_init_lora_scale)
+        init_scale = max(0.0, init_scale)
+        if init_scale > 1.0:
+            print_acc(f"zimage_init_lora_scale {init_scale} > 1.0, clamping to 1.0 for stable warm start.")
+            init_scale = 1.0
+
+        load_payload = init_lora_path
+        if init_scale < 0.9999:
+            load_payload = self._build_scaled_i2l_state_dict(init_lora_path, init_scale)
+
+        self.network.load_weights(load_payload)
+        self.network._update_torch_multiplier()
+        print_acc(f"Loaded I2L initialization LoRA from: {init_lora_path} (scale={init_scale})")
+
+        if self.train_config.zimage_i2l_only:
+            print_acc("zimage_i2l_only enabled: generated/loaded I2L LoRA and skipping conventional training loop.")
+            self.train_config.disable_sampling = True
+            self.exit_after_hook_before_train_loop = True
+
+    def _collect_zimage_i2l_image_paths(self):
+        max_images = int(self.train_config.zimage_i2l_max_images)
+        max_images = min(max(max_images, 1), 6)
+
+        dataset_path = getattr(self.train_config, "zimage_i2l_dataset_path", None)
+        if dataset_path is not None and dataset_path != "":
+            if not os.path.isdir(dataset_path):
+                raise ValueError(f"train.zimage_i2l_dataset_path does not exist: {dataset_path}")
+            image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
+            files = [
+                os.path.join(dataset_path, f)
+                for f in sorted(os.listdir(dataset_path))
+                if f.lower().endswith(image_exts)
+            ]
+            if len(files) == 0:
+                raise ValueError(f"No images found in train.zimage_i2l_dataset_path: {dataset_path}")
+            return files[:max_images]
+
+        if self.train_config.zimage_i2l_image_paths is not None and len(self.train_config.zimage_i2l_image_paths) > 0:
+            valid_paths = [p for p in self.train_config.zimage_i2l_image_paths if os.path.exists(p)]
+            if len(valid_paths) == 0:
+                raise ValueError("train.zimage_i2l_image_paths was provided, but no files exist.")
+            return valid_paths[:max_images]
+
+        candidate_paths = []
+        for dataset in (self.datasets or []):
+            if not hasattr(dataset, "file_list"):
+                continue
+            for item in dataset.file_list:
+                path = getattr(item, "path", None)
+                if path is None and isinstance(item, str):
+                    path = item
+                if path is None:
+                    continue
+                if os.path.exists(path):
+                    candidate_paths.append(path)
+                if len(candidate_paths) >= max_images:
+                    break
+            if len(candidate_paths) >= max_images:
+                break
+
+        if len(candidate_paths) == 0:
+            raise ValueError(
+                "Could not find images for I2L generation. Set train.zimage_i2l_image_paths explicitly."
+            )
+        unique_paths = sorted(set(candidate_paths))
+        if len(unique_paths) <= max_images:
+            return unique_paths
+
+        seed = getattr(self.train_config, "zimage_i2l_image_select_seed", None)
+        rng = random.Random()
+        if seed is None:
+            rng.seed()
+        else:
+            rng.seed(seed)
+        return rng.sample(unique_paths, max_images)
+
+    def _build_scaled_i2l_state_dict(self, lora_path: str, scale: float):
+        lora_sd = load_file(lora_path)
+        current_sd = self.network.state_dict()
+
+        blended_sd = {}
+        for key, value in lora_sd.items():
+            mapped_key = key
+            if mapped_key.startswith("diffusion_model."):
+                mapped_key = mapped_key.replace("diffusion_model.", "transformer.", 1)
+
+            if mapped_key in current_sd and current_sd[mapped_key].shape == value.shape:
+                cur = current_sd[mapped_key].detach().to(dtype=value.dtype, device=value.device)
+                blended_sd[mapped_key] = cur * (1.0 - scale) + value * scale
+            else:
+                blended_sd[mapped_key] = value
+
+        return blended_sd
+
+    def _get_i2l_cache_index_path(self):
+        return os.path.join(self.save_root, "i2l", "cache_index.json")
+
+    def _compute_i2l_cache_key(self, image_paths: List[str]):
+        digest = hashlib.sha256()
+        digest.update(self.train_config.zimage_i2l_model_name_or_path.encode("utf-8"))
+        digest.update(self.train_config.zimage_i2l_base_model_name_or_path.encode("utf-8"))
+        digest.update(self.train_config.zimage_i2l_turbo_model_name_or_path.encode("utf-8"))
+
+        for path in sorted(image_paths):
+            with open(path, "rb") as f:
+                digest.update(hashlib.sha256(f.read()).digest())
+        return digest.hexdigest()
+
+    def _load_i2l_cache_index(self):
+        index_path = self._get_i2l_cache_index_path()
+        if not os.path.exists(index_path):
+            return {}
+        try:
+            with open(index_path, "r", encoding="utf-8") as f:
+                return json.load(f)
+        except Exception:
+            return {}
+
+    def _save_i2l_cache_index(self, data):
+        index_path = self._get_i2l_cache_index_path()
+        os.makedirs(os.path.dirname(index_path), exist_ok=True)
+        with open(index_path, "w", encoding="utf-8") as f:
+            json.dump(data, f, indent=2)
+
+    def _get_i2l_cache_output_path(self, image_paths: List[str]):
+        key = self._compute_i2l_cache_key(image_paths)
+        return os.path.join(self.save_root, "i2l", "cache", f"{key}.safetensors")
+
+    def _get_i2l_cached_lora_path(self, image_paths: List[str]):
+        key = self._compute_i2l_cache_key(image_paths)
+        index = self._load_i2l_cache_index()
+        entry = index.get(key, None)
+        if entry is None:
+            return None
+        lora_path = entry.get("output_path", None)
+        if lora_path is None or not os.path.exists(lora_path):
+            return None
+        return lora_path
+
+    def _record_i2l_cache_entry(self, image_paths: List[str], output_path: str):
+        key = self._compute_i2l_cache_key(image_paths)
+        index = self._load_i2l_cache_index()
+        index[key] = {
+            "output_path": output_path,
+            "image_paths": image_paths,
+            "model_id": self.train_config.zimage_i2l_model_name_or_path,
+        }
+        self._save_i2l_cache_index(index)
     
     def cache_sample_prompts(self):
         if self.train_config.disable_sampling:
             return
         if self.sample_config is not None and self.sample_config.samples is not None and len(self.sample_config.samples) > 0:
             # cache all the samples
             self.sd.sample_prompts_cache = []
             sample_folder = os.path.join(self.save_root, 'samples')
             output_path = os.path.join(sample_folder, 'test.jpg')
             for i in range(len(self.sample_config.prompts)):
                 sample_item = self.sample_config.samples[i]
                 prompt = self.sample_config.prompts[i]
 
                 # needed so we can autoparse the prompt to handle flags
                 gen_img_config = GenerateImageConfig(
                     prompt=prompt, # it will autoparse the prompt
                     negative_prompt=sample_item.neg,
                     output_path=output_path,
                     ctrl_img=sample_item.ctrl_img,
                     ctrl_img_1=sample_item.ctrl_img_1,
                     ctrl_img_2=sample_item.ctrl_img_2,
                     ctrl_img_3=sample_item.ctrl_img_3,
                 )
                 
                 has_control_images = False
@@ -222,50 +434,62 @@ class SDTrainer(BaseSDTrainProcess):
                     adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                 ).to(self.device_torch)
             elif self.train_config.adapter_assist_type == "control_net":
                 self.assistant_adapter = ControlNetModel.from_pretrained(
                     adapter_path, torch_dtype=get_torch_dtype(self.train_config.dtype)
                 ).to(self.device_torch, dtype=get_torch_dtype(self.train_config.dtype))
             else:
                 raise ValueError(f"Unknown adapter assist type {self.train_config.adapter_assist_type}")
 
             self.assistant_adapter.eval()
             self.assistant_adapter.requires_grad_(False)
             flush()
         if self.train_config.train_turbo and self.train_config.show_turbo_outputs:
             if self.model_config.is_xl:
                 self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesdxl",
                                                              torch_dtype=get_torch_dtype(self.train_config.dtype))
             else:
                 self.taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd",
                                                              torch_dtype=get_torch_dtype(self.train_config.dtype))
             self.taesd.to(dtype=get_torch_dtype(self.train_config.dtype), device=self.device_torch)
             self.taesd.eval()
             self.taesd.requires_grad_(False)
 
     def hook_before_train_loop(self):
         super().hook_before_train_loop()
+        should_do_i2l = (
+            getattr(self.sd, "arch", None) == "zimage"
+            and (self.train_config.zimage_i2l_only or self.train_config.zimage_bootstrap_enabled)
+        )
+        if should_do_i2l and not self.train_config.disable_sampling:
+            print_acc("Generating pre-I2L sample (step 0)")
+            self.sample(0)
+        self._maybe_run_zimage_i2l_stage()
+        if should_do_i2l and not self.train_config.disable_sampling:
+            print_acc("Generating post-I2L sample (step 0)")
+            self.sample(0)
+            self.did_preloop_baseline_sample = True
         if self.is_caching_text_embeddings:
             # make sure model is on cpu for this part so we don't oom.
             self.sd.unet.to('cpu')
         
         # cache unconditional embeds (blank prompt)
         with torch.no_grad():
             kwargs = {}
             if self.sd.encode_control_in_text_embeddings:
                 # just do a blank image for unconditionals
                 control_image = torch.zeros((1, 3, 224, 224), device=self.sd.device_torch, dtype=self.sd.torch_dtype)
                 if self.sd.has_multiple_control_images:
                     control_image = [control_image]
                 
                 kwargs['control_images'] = control_image
             self.unconditional_embeds = self.sd.encode_prompt(
                 [self.train_config.unconditional_prompt],
                 long_prompts=self.do_long_prompts,
                 **kwargs
             ).to(
                 self.device_torch,
                 dtype=self.sd.torch_dtype
             ).detach()
         
         if self.train_config.do_prior_divergence:
             self.do_prior_prediction = True
