# Denoising Diffusion Probabilistic Models (DDPM)

Implementation of [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho, Jain, Abbeel, 2020).

## What this implements

The DDPM training and sampling procedure — a class of generative models that learn to reverse a gradual noising process. The core contribution is the training objective (simplified variational bound) and the reverse sampling algorithm, NOT the U-Net architecture (which is adapted from prior work). This implementation covers the forward diffusion process, the simplified training objective (L_simple), the noise schedule, and the reverse sampling algorithm from Algorithm 1 and Algorithm 2.

## Quick start

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip --isolated install --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r requirements-win-py313.lock
```

The security-supported target is 64-bit Windows on CPython 3.13 using the canonical PyPI index. `requirements.in` is the reviewed direct-input manifest; it is not a lock. `requirements-win-py313.lock` contains the complete transitive closure and artifact hashes, including the notebook, FID, and safetensors dependencies. Review and audit that exact lock before installation. Use a disposable environment without credentials or sensitive data. Other platforms are not security-supported by this example.

Checkpoints are two-file pairs: tensor data in `<name>.safetensors` and strict bounded configuration/metadata in `<name>.json`. The loader rejects links/reparse points, non-regular or oversized files, invalid headers, unsupported dtypes, excessive tensor counts/dimensions/elements/bytes, and any key or shape that differs from the validated U-Net schema before it materializes the selected state on CPU or moves the model to an accelerator. Python pickle checkpoints are intentionally unsupported. Optimizer state is not persisted, so these files support evaluation and weight restoration rather than exact optimizer resume.

On Windows, Python does not expose POSIX-equivalent directory-handle/no-follow semantics for every operation. Reparse, regular-file, and identity checks reduce risk but cannot eliminate a path-swap race by another process running as the same Windows principal. Store checkpoints in an ACL-isolated directory without concurrent untrusted same-account processes.

Training requires a CIFAR-10 copy already present in the configured `data_dir`. The data loader uses `download=False` and fails if the dataset is missing. Acquire and verify dataset files separately before training.

FID evaluation also stays offline. Manually obtain the [upstream FID Inception weights](https://github.com/mseitzer/pytorch-fid/releases/tag/fid_weights) in an isolated environment, then run the repository's `scripts/convert_fid_inception.py` from the repository root with the downloaded `.pth` path and a new `.safetensors` destination. The converter accepts only the pinned 95,628,359-byte file with SHA-256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`, loads it with `weights_only=True`, and writes safetensors. Evaluation requires `--fid_weights` pointing to that local safetensors file and `--real_stats` pointing to local statistics or images. Without these inputs, FID fails closed; it never asks `pytorch-fid` to download its default pickle weights.

Before calling `pytorch-fid`, evaluation checks both FID inputs. Stats must be a `.npz` containing only bounded float32/float64 `mu` and `sigma` arrays with shapes matching `dims` (including the standard 2048-dimensional stats); ZIP64 metadata is unsupported. An image directory is limited to 50,100 supported images, 16 MiB and 4 megapixels per image, 2 GiB and 250 million pixels in total; malformed supported images fail closed. Evaluation also rejects explicit Windows UNC/device paths, remote mapped-drive roots, unknown drive roots, and symlink or reparse components for FID inputs and weights. This does not detect every network-backed mount nested under a fixed Windows drive; on POSIX, caller-selected mounts are trusted. No implicit HTTP access occurs. A same-account writer could still replace a path after preflight and before `pytorch-fid` reopens it. Keep these files in an ACL-isolated directory without concurrent untrusted writers.

```powershell
python scripts/convert_fid_inception.py <downloaded-weights.pth> <local-fid-weights.safetensors>
```

```python
from src.model import UNet, UNetConfig
from src.loss import DDPMLoss
from src.utils import linear_noise_schedule

config = UNetConfig()
model = UNet(config)
noise_schedule = linear_noise_schedule(timesteps=1000)

# Example: predict noise from noisy image at timestep t
import torch
x_t = torch.randn(2, 3, 32, 32)    # noisy image
t = torch.randint(0, 1000, (2,))    # timesteps
predicted_noise = model(x_t, t)
print(predicted_noise.shape)  # (2, 3, 32, 32)
```

## File structure

```
ddpm/
├── README.md                 # This file
├── REPRODUCTION_NOTES.md     # Ambiguity audit — what's specified vs. assumed
├── requirements.in           # Reviewed direct dependency inputs
├── requirements-win-py313.lock # Complete target-specific hash lock
├── src/
│   ├── checkpoint.py         # Strict safetensors + bounded JSON persistence
│   ├── model.py              # U-Net noise prediction network (§3.3, Appendix B)
│   ├── loss.py               # DDPM simplified loss L_simple (§3.4, Eq. 14)
│   ├── data.py               # Dataset skeleton for image data
│   ├── train.py              # Training loop — Algorithm 1
│   ├── evaluate.py           # FID score computation
│   └── utils.py              # Noise schedule, forward process, sampling (Algorithm 2)
├── configs/
│   └── base.yaml             # All hyperparameters from §4 and Appendix B
└── notebooks/
    └── walkthrough.ipynb     # Paper sections → code → sanity checks
```

## Important: Read REPRODUCTION_NOTES.md

This implementation flags every choice that the paper does not specify.
Before using this code for research, read [REPRODUCTION_NOTES.md](REPRODUCTION_NOTES.md)
to understand which implementation details are from the paper and which are our choices.

## Citation

```bibtex
@article{ho2020denoising,
  title={Denoising diffusion probabilistic models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={6840--6851},
  year={2020}
}
```

---

*Generated by [paper2code](https://github.com/PrathamLearnsToCode/paper2code) — citation-anchored paper implementation.*
