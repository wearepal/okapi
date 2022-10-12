#!/bin/bash
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	# Determine whether any GPUs are available and thus whether 
	# to install the CUDA-enabled version of PyTorch.
	if [[ $(lshw -C display | grep vendor) =~ NVIDIA ]]
        then
            echo "At least one CUDA-compatible device detected: re-installing PyTorch with CUDA support."
            pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
        else
            pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
	fi
fi

poetry install
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12+cu116.html --no-cache --force
