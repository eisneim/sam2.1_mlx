# SAM2.1 MLX

SAM2.1 (Segment Anything Model 2.1) ported to Apple MLX for native inference on Apple Silicon.

Supports video object tracking and single-image segmentation. Click on an object in the first frame, and the model tracks it through the entire video.

<video src="https://github.com/user-attachments/assets/63d26cc3-4146-4377-b352-3b44b0b76627">
</video>



## Features

- SAM2.1 small / base_plus / large architectures on MLX
- Video object tracking with memory bank propagation
- Single-image segmentation with point prompts
- Interactive click-to-select via OpenCV
- Preloaded features for fast propagation (~130 fps on M-series)

## Requirements

- macOS with Apple Silicon (MLX requires Metal GPU)
- Python 3.10+

## Install

```bash
git clone https://github.com/eisneim/sam2.1_mlx.git
cd sam2.1_mlx
pip install mlx opencv-python safetensors numpy
```

## Weights

Download pre-converted MLX weights from Hugging Face:

```bash
# Base Plus (recommended)
huggingface-cli download eisneim/sam2.1_mlx_weights sam2.1_hiera_base_plus.safetensors --local-dir weights/

# Small (faster)
huggingface-cli download eisneim/sam2.1_mlx_weights sam2.1_hiera_small.safetensors --local-dir weights/
```

Or convert from PyTorch yourself:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -P weights/
python -m src.sam2.convert --src weights/sam2.1_hiera_base_plus.pt --dst weights/sam2.1_hiera_base_plus.safetensors
```

## Usage

### Video Tracking

Click on an object in the first frame, then the model tracks it through all frames:

```bash
python inference_video.py -i path/to/video.mp4
python inference_video.py -i path/to/video.mp4 --model small
```

Options:
- `-i, --video` — input video file
- `--model` — `small`, `base_plus` (default), or `large`
- `--weights` — path to `.safetensors` weights (auto-selected by model)
- `--output` — output directory (default: `output/`)

Outputs:
- `tracked_overlay.mp4` — video with green mask overlay
- `tracked_mask.mp4` — binary mask video

### Image Segmentation

Click on an object in the image to get its segmentation mask:

```bash
python inference_image.py -i path/to/image.jpg
python inference_image.py -i path/to/image.jpg --model small
```

Outputs:
- `{name}_overlay.png` — image with green mask overlay
- `{name}_mask.png` — binary mask

## Architecture

```
src/sam2/
├── __init__.py          # Package exports
├── build.py             # Model configs and builder
├── convert.py           # PyTorch → MLX weight converter
├── hiera.py             # Hiera vision backbone (windowed attention)
├── image_encoder.py     # FPN neck + image encoder wrapper
├── mask_decoder.py      # Mask decoder, prompt encoder, two-way transformer
├── memory.py            # Memory encoder, memory attention (cross-attn with RoPE)
├── position_encoding.py # Sinusoidal PE, random PE, 2D axial RoPE
├── predictor.py         # SAM2VideoPredictor, SAM2ImagePredictor
├── sam2_base.py         # Core SAM2Base orchestrator
└── sam2_utils.py        # MLP, DropPath, LayerNorm2d, window ops
```

Key design choices:
- All conv/linear weights use MLX channels-last layout `[O, H, W, I]`
- ImageNet normalization (not `img * 2 - 1`) to match Hiera's training statistics
- Square resize (not longest-edge + pad) to match training geometry
- Conditioning frames keep `t_pos=0` temporal encoding throughout propagation

## Tested On

| Video | Frames | Mean IoU | All >= 0.5 |
|-------|--------|----------|------------|
| 1     | 60     | 0.608    | 60/60      |
| 2     | 60     | 0.897    | 60/60      |
| v     | 64     | 0.981    | 64/64      |

## Acknowledgments

- [SAM2](https://github.com/facebookresearch/sam2) by Meta FAIR
- [MLX](https://github.com/ml-explore/mlx) by Apple

## License

This project follows the same license as the original SAM2 (Apache 2.0).
