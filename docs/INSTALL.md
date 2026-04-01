# Installation

## 1. Create Conda Environment
```bash
conda create -n dcap python=3.8 -y
conda activate dcap
```

## 2. Set CUDA Environment Variables

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDNN_PATH=$CUDA_HOME/lib64
```

## 3. Install PyTorch (cu118)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 4. Install Build Tools
```bash
pip install ninja packaging
```

## 5. Install dCAP
```bash
pip install -r requirements.txt
pip install -v -e .
```

## 6. Verify Installation
```bash
python - <<'PY'
import torch
import dcap
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('dcap import: OK')
PY
```
