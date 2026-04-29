# DaST: Data-Free Substitute Training for Adversarial Attacks

Code replicating the CVPR 2020 (Oral) paper, "DaST: Data-Free Substitute Training for Adversarial Attacks" by Zhou et al. (Monash University). [Paper here.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_DaST_Data-Free_Substitute_Training_for_Adversarial_Attacks_CVPR_2020_paper.pdf)

DaST attacks a black-box ML model **without requiring any real training data**. A multi-branch GAN generates synthetic images from noise, queries the target model for labels, and trains a substitute model on those responses to mimic the target's decision boundary. Standard white-box attacks (FGSM, BIM, PGD, CW) are then run on the substitute, and the resulting adversarial examples transfer to fool the real target.

The Authors provide attack code for two scenarios: **DaST-L** (label-only) and **DaST-P** (probability-based). All attacks are run on MNIST using PyTorch. An Azure-simulation mode (sklearn target) is also supported.

## Step by Step Guide

1. Install the packages listed in [docs/requirements.txt](docs/requirements.txt) (see **Software Installation** below).
2. Download the pretrained models listed in the **Models** section and place them in `DaST/pretrained/`.
3. If running on an HPC cluster, update `job.sh` with your own system's file paths (see **Running on HPC** below).
4. Train a DaST substitute model using one of the commands in the **How to Run** section.
5. Evaluate transfer attack performance with `evaluation.py` or `test_dast.py`.

## How to Run

**Train a DaST substitute against the MNIST target (DaST-P, 80 epochs):**
```bash
python dast.py --dataset=mnist --niter=80 --batchSize=500 --alpha=0.2 --beta=0.1 --G_type=1 --save_folder=saved_model
```
Best models are saved to `saved_model/` whenever a new best ASR or accuracy is achieved. Set `--beta=0` for DaST-L (label-only mode).`

**Evaluate transfer attack performance:**
```bash
python evaluation.py --mode=dast --adv=FGSM --cuda
```

`--mode` options: `white` (white-box upper bound), `black` (real-data baseline), `dast` (your trained substitute)

`--adv` options: `FGSM`, `BIM`, `PGD`, `CW`

Add `--target` for targeted attacks.

## How to Run on a HPC

A SLURM job script is provided at `job.sh`. Before submitting, update the two hardcoded paths to match your cluster's directory structure:

```bash
source /work/pi_csc592_uri_edu/Thomas/DaSTProject/.venv/bin/activate
cd /work/pi_csc592_uri_edu/Thomas/DaSTProject/DaST
```

Change these to point to wherever you placed the project and virtual environment on your system. The script requests 1 GPU, 4 CPUs, 12 GB RAM, and a 32-hour time limit on the `gpu-preempt` partition — adjust these as needed for your cluster.

## Software Installation

I used **Python 3.12** and **CUDA 12.1**. Full dependency versions and setup instructions are in [docs/requirements.txt](docs/requirements.txt).

Quick setup but highly suggest checking [docs/requirements.txt](docs/requirements.txt) first:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --no-cache-dir -r docs/requirements.txt
```

After installing, patch `advertorch` for PyTorch 2.x compatibility (the `zero_gradients` removal) — the patch command is documented at the top of [docs/requirements.txt](docs/requirements.txt).

Key packages:

- `python==3.12`
- `torch==2.10.0`
- `torchvision==0.25.0`
- `numpy==2.4.2`
- `advertorch==0.2.3`
- `foolbox==3.3.4`
- `scikit-learn==1.8.0`

## Models

The authors provide the following pretrained models in `DaST/pretrained/`:

| File | Role |
|------|------|
| `net_m.pth` | **Attack target** for MNIST (Net_m CNN) |
| `net_l.pth` | Black-box baseline substitute (trained on real MNIST data) |
| `net_s.pth` | Small MNIST CNN |
| `sklearn_mnist_model.pkl` | Azure remote model simulation (sklearn classifier) |
| `vgg16cifar10.pth` | Attack target for CIFAR-10 (VGG16, ~93.9% accuracy) |

The `net_m.pth` target model is required to run any MNIST attack. No pretrained DaST substitute is included — you train one yourself.

## Experiments & Results

All experiments I did were run on MNIST using `netD_epoch_52.pth`, the best model from Training Test 2 (80 epochs, DaST-P). The substitute achieved **71.70% accuracy** on real MNIST images despite being trained entirely on GAN-generated data.

| Attack | ASR | L2 Distance | Notes |
|--------|-----|-------------|-------|
| FGSM | 7.62% | 4.11 | Single-step; perturbations don't transfer well |
| BIM | 81.34% | 4.78 | Iterative; strong transfer |
| PGD | 83.55% | 5.00 | Iterative with random start, best ASR |
| CW | 2.45% | 0.89 | Minimal perturbations overfit to substitute |

Full results and notes are in [docs/Training&AttackTestingNotes.md](docs/Training&AttackTestingNotes.md).

## System Requirements

Training was run on the Unity HPC cluster (NVIDIA A40 GPU, CUDA 12.1). The MNIST experiments require modest GPU memory. CPU-only mode is supported but significantly slower.

# Citation:
```latex
@inproceedings{zhou2020dast,
  title={DaST: Data-free Substitute Training for Adversarial Attacks},
  author={Zhou, Mingyi and Wu, Jing and Liu, Yipeng and Liu, Shuaicheng and Zhu, Ce},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={234--243},
  year={2020}
}
```
