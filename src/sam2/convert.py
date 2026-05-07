"""Convert SAM2.1 PyTorch checkpoint to MLX safetensors format.

Usage:
    python -m src.sam2.convert --src weights/sam2.1_hiera_base_plus.pt --dst weights/sam2.1_hiera_base_plus.safetensors
"""

import argparse
import os
import re

import mlx.core as mx
import numpy as np
import safetensors.numpy
import torch


# MLX nn.Sequential uses .layers.N prefix; PyTorch uses direct .N
SEQUENTIAL_MODULES = [
    "output_upscaling",
    "mask_input",
    "mask_downscaling",
    "mask_downsampler",
]

PREFIX_RENAMES = {
    "sam_mask_decoder": "mask_decoder",
    "sam_prompt_encoder": "prompt_encoder",
}

SUBKEY_RENAMES = {
    "final_attn_token_to_image": "final_attn",
    "norm_final_attn": "norm_final",
    "cross_attn_image": "cross_attn",
    "cross_attn_to_token": "cross_attn_image_to_token",
    "mask_downscaling": "mask_input",
    "pwconv1": "pw1",
    "pwconv2": "pw2",
    "positional_encoding_gaussian_matrix": "gaussian_matrix",
}

# ConvTranspose2d keys have [I, O, H, W] layout (vs Conv2d [O, I, H, W])
CONV_TRANSPOSE_KEYS = {
    "mask_decoder.output_upscaling.layers.0.weight",
    "mask_decoder.output_upscaling.layers.3.weight",
}


def convert(src: str, dst: str):
    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    pt = ckpt["model"]
    print(f"Loaded {len(pt)} keys from {src}")

    out = {}
    for pt_key, tensor in pt.items():
        w = mx.array(tensor.detach().cpu().numpy().astype(np.float32))
        ndim = tensor.ndim
        mk = pt_key

        # Prefix renames
        for old, new in PREFIX_RENAMES.items():
            if mk.startswith(old + "."):
                mk = new + mk[len(old):]
                break

        # Sub-key renames
        for old, new in SUBKEY_RENAMES.items():
            if old in mk:
                mk = mk.replace(old, new)

        # Sequential layer remap
        for seq_name in SEQUENTIAL_MODULES:
            if f".{seq_name}." in mk:
                m = re.match(rf"(.+\.{seq_name})\.(\d+)\.(.+)", mk)
                if m:
                    mk = f"{m.group(1)}.layers.{m.group(2)}.{m.group(3)}"
                break

        # patch_embed.proj -> patch_embed
        mk = mk.replace("image_encoder.trunk.patch_embed.proj.", "image_encoder.trunk.patch_embed.")

        # neck.convs.N.conv -> neck.convs.N
        m = re.match(r"image_encoder\.neck\.convs\.(\d+)\.conv\.(.+)", mk)
        if m:
            mk = f"image_encoder.neck.convs.{m.group(1)}.{m.group(2)}"

        # mask_downsampler.encoder -> mask_downsampler.layers
        mk = mk.replace("mask_downsampler.encoder.", "mask_downsampler.layers.")
        if mk in (
            "memory_encoder.mask_downsampler.layers.12.weight",
            "memory_encoder.mask_downsampler.layers.12.bias",
        ):
            mk = mk.replace("mask_downsampler.layers.12", "mask_downsampler.final_conv")

        # Weight layout: PyTorch [O,I,H,W] -> MLX [O,H,W,I]
        if ndim == 4 and mk.endswith(".weight"):
            if mk in CONV_TRANSPOSE_KEYS:
                w = w.transpose(1, 2, 3, 0)
            else:
                w = w.transpose(0, 2, 3, 1)

        out[mk] = w

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    arrays = {k: np.array(v, dtype=np.float32) for k, v in out.items()}
    safetensors.numpy.save_file(arrays, dst)
    print(f"Wrote {len(out)} weights -> {dst}")


def main():
    parser = argparse.ArgumentParser(description="Convert SAM2.1 PyTorch weights to MLX safetensors")
    parser.add_argument("--src", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dst", required=True, help="Output .safetensors path")
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
