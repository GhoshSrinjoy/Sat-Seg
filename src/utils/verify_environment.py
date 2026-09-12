"""Run with python -m src.utils.verify_environment; fail if CUDA is unavailable."""

import importlib
import json
import platform
import sys
from importlib.metadata import version

import torch


def main() -> None:
    packages = {
        "numpy": "numpy", "pandas": "pandas", "matplotlib": "matplotlib",
        "opencv-python": "cv2", "pillow": "PIL", "scikit-learn": "sklearn",
        "tqdm": "tqdm", "jupyter": "jupyter", "jupyterlab": "jupyterlab",
        "ipykernel": "ipykernel", "rasterio": "rasterio", "geopandas": "geopandas",
        "shapely": "shapely", "pyproj": "pyproj", "pyyaml": "yaml",
        "transformers": "transformers", "accelerate": "accelerate",
        "huggingface-hub": "huggingface_hub", "safetensors": "safetensors",
        "torchvision": "torchvision",
    }
    for module in packages.values():
        importlib.import_module(module)
    print(f"Python: {platform.python_version()} ({sys.executable})", flush=True)
    print(f"PyTorch: {torch.__version__}; bundled CUDA: {torch.version.cuda}", flush=True)
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}", flush=True)
    if not available:
        raise RuntimeError("CUDA is required. Install the CUDA wheel and check the NVIDIA driver.")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    # Execute a real GPU operation, not just a driver discovery check.
    tensor = torch.ones((32, 32), device="cuda")
    result = tensor @ tensor
    torch.cuda.synchronize()
    if not torch.allclose(result, torch.full_like(result, 32)):
        raise RuntimeError("CUDA matrix multiplication produced an unexpected result.")
    # Exercise torchvision's compiled CUDA extension too.
    from torchvision.ops import nms
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9]], dtype=torch.float32, device="cuda")
    keep = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    if keep.tolist() != [0]:
        raise RuntimeError("torchvision CUDA NMS verification failed.")
    print("CUDA matrix multiplication and torchvision NMS: PASS")
    print(json.dumps({name: version(name) for name in packages}, indent=2))


if __name__ == "__main__":
    main()
