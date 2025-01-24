## Prerequisites

Install the NVIDIA CUDA Toolkit

https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#download-the-nvidia-cuda-toolkit

I setup `cuda-keyring_1.1-1_all.deb` to add the nvidia repository to my apt repos list, and then I ran `sudo apt-get install cuda-toolkit`

## Installation instructions

0. Setup venv and activate

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

1. Install CUML from RapidsAI

```bash
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.12.*" "dask-cudf-cu12==24.12.*" "cuml-cu12==24.12.*" \
    "cugraph-cu12==24.12.*" "nx-cugraph-cu12==24.12.*" "cuspatial-cu12==24.12.*" \
    "cuproj-cu12==24.12.*" "cuxfilter-cu12==24.12.*" "cucim-cu12==24.12.*" \
    "pylibraft-cu12==24.12.*" "raft-dask-cu12==24.12.*" "cuvs-cu12==24.12.*" \
    "nx-cugraph-cu12==24.12.*"
```

2. Then, install deps from requirements

```bash
pip install --extra-index-url=https://pypi.nvidia.com -r requirements.txt
```
