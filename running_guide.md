## environment
- OS: `Linux user-SYS-7049GP-TRT 5.15.0-139-generic #149~20.04.1-Ubuntu SMP Wed Apr 16 08:29:56 UTC 2025 x86_64 x86_64 x86_64 GNU/Linux`
- Python: `python 3.12`
- CUDA: `CUDA 12.6`
- Hardware: `NVIDIA GeForce RTX 3090`

The training and inference script are designed to run on a single NVIDIA GeForce RTX 3090.
We recommend using `uv sync` for environment setup and package management. Alternatively, a `requirements.txt` is provided for standard pip installation.

## inference guide
To reproduce the public score on the Kaggle Leaderboard, first, download the dataset by running:

```bash
bash download_data.sh
```

Download the pretrained models:

```bash
wget https://huggingface.co/alan314159/DLP_Lab_models/resolve/main/lab2/best_resnet_unet_model.pth -O saved_models/best_resnet_unet_model.pth
wget https://huggingface.co/alan314159/DLP_Lab_models/resolve/main/lab2/best_unet_model.pth -O saved_models/best_unet_model.pth
```

For UNet, run:
```bash
python -m src.inference --model-type unet --split test_unet --model-path saved_models/best_unet_model.pth --threshold 0.3 --output-csv submission_unet.csv
``` 

For ResNet34_UNet, run:
```bash
python -m src.inference --model-type resnet --split test_resnet_unet --model-path saved_models/best_resnet_unet_model.pth --threshold 0.45 --output-csv submission_resnet_unet.csv
```

## training guide
First, download the dataset by running:

```bash
bash download_data.sh
```

For UNet, run:
```bash
python -m src.train_unet --config configs/baseline_v12.json
```

the final model will be saved in `logs/new_unet_baseline_v12/saved_models/best_model.pth`. Please use `src/inference.py` to evaluate the model and generate the submission CSV.

For ResNet34_UNet, run:
```bash
python -m src.train_resnet --config configs/resnet_v13.json
``` 

the final model will be saved in `logs/resnet_v13/saved_models/best_model.pth`. Please use `src/inference.py` to evaluate the model and generate the submission CSV.