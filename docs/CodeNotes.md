# DaST Project — Code Notes

**Paper:** DaST: Data-Free Substitute Training for Adversarial Attacks (CVPR 2020, Oral)
**Authors:** Zhou et al. — Monash University

---

## What This Project Does

DaST attacks a black-box ML model **without requiring any real training data**. The core idea:

1. A GAN generates synthetic images from random noise.
2. Those synthetic images are sent to the **target (black-box) model** to get labels.
3. A **substitute model** is trained on those GAN-labeled synthetic images to mimic the target's decision boundary.
4. Standard white-box attacks (FGSM, BIM, PGD, CW) run on the substitute — adversarial examples **transfer** to fool the real target.

This is a **black-box adversarial attack** — the attacker never sees the target's weights, only its predictions.

---

## Project File Map

```
dast.py              — Main training script (MNIST + Azure targets)
dast_cifar10.py      — Training script adapted for CIFAR-10
evaluation.py        — Full evaluation (transfer attack ASR + L2 distance)
test_dast.py         — Alternate evaluation with L2-norm metrics, batch mode
net.py               — Three MNIST CNN architectures (Net_s, Net_m, Net_l)
vgg.py               — VGG11/13/16/19 for CIFAR-10
resnet.py            — ResNet18/34/50/101/152

pretrained/
  net_m.pth              — TARGET model for MNIST attacks (Net_m, medium CNN)
  net_l.pth              — Substitute trained on real data (black-box baseline)
  net_s.pth              — Small MNIST CNN
  sklearn_mnist_model.pkl — Simulates Microsoft Azure remote model
  vgg16cifar10.pth        — TARGET model for CIFAR-10 (VGG16, ~93.9% acc)

saved_model/
  notes.txt          — Run your trained DaST models here; evaluate with test_dast.py

dataset/
  MNIST/             — Already downloaded and processed
```

---

## Pre-trained Models — What's Already There

| File | Role | Notes |
|------|------|-------|
| `pretrained/net_m.pth` | **Attack target** (MNIST) | Net_m architecture, this is what you're trying to fool |
| `pretrained/net_l.pth` | Black-box baseline substitute | Trained on real MNIST data — upper bound for black-box |
| `pretrained/net_s.pth` | Small MNIST model | Available for experiments |
| `pretrained/sklearn_mnist_model.pkl` | Azure model simulation | Sklearn classifier, simulates remote API |
| `pretrained/vgg16cifar10.pth` | **Attack target** (CIFAR-10) | VGG16, ~93.9% accuracy |

**No pre-trained DaST substitute model is included** — you train one yourself via `dast.py`.

---

## Architecture: The Multi-Branch GAN

The generator has two parts:

1. **10 separate `pre_conv` branch blocks** — one per output class. Each takes a chunk of random noise (shape `[B, 128, 1, 1]`) and upsamples it into feature maps via ConvTranspose layers. This biases each branch toward generating samples for its class.

2. **Shared `Generator` (netG)** — takes the branch output and refines it into a final image (28×28 for MNIST, 32×32 for CIFAR-10). Uses Conv layers and a Sigmoid final activation.

The substitute model is called **netD** — it's `Net_l` for MNIST or `VGG13` for CIFAR-10.

**Two generator types** (`--G_type`):
- `1` (default): ConvTranspose upsampling in pre_conv → Conv refinement in Generator
- `2`: Different Conv-only architecture; neither is definitively better

---

## Training Loop (per epoch)

Each epoch runs `batch_num` inner iterations:

**Step 1 — Update substitute model (netD):**
- Sample noise, split into 10 class chunks
- Each chunk → its `pre_conv` branch → shared `Generator` → synthetic images
- Query target model for labels (hard labels) + probabilities (if DaST-P)
- Train netD on synthetic data with `CrossEntropyLoss + beta * MSE(probabilities)`

**Step 2 — Update Generator (netG + pre_conv blocks):**
- Compute two losses on the same batch:
  - `loss_imitate`: maximize how hard netD finds it to learn (uses `Loss_max` which negates and exponentiates the CE loss)
  - `loss_diversity`: `CrossEntropy(netD_output, class_branch_labels)` — forces coverage of all classes
- `errG = alpha * loss_diversity + loss_imitate`
- Adaptive alpha: if `loss_diversity < 0.1`, alpha is reduced to match it

**Per epoch evaluation:** adversarial attack success rate (ASR) is computed on a 2000-sample MNIST validation subset (indices 6000–8000). Best ASR model is saved.

---

## Two Attack Scenarios

| Mode | `--beta` | Description |
|------|----------|-------------|
| **DaST-L** | `0` | Label-only. Target model queried for hard class labels. |
| **DaST-P** | `> 0` (e.g. 0.1) | Probability-based. Also uses soft probability outputs. Performs better when the target exposes probabilities. |

---

## How to Run

### Environment (original paper)
```
Python 3.9.12
PyTorch 1.12.0
CUDA 11.3.1
GPU: A40
```

### Dependencies to install
```bash
pip install torch torchvision advertorch foolbox xlwt scikit-learn
```

> **Note:** `dast.py` uses `from sklearn.externals import joblib` which was removed after sklearn 0.23. Replace with `import joblib` if you get an ImportError.

### Train a DaST substitute (MNIST target)
```bash
python dast.py --dataset=mnist
```
Trains for 2000 epochs by default. Models saved to `saved_model/` when best ASR or accuracy is achieved.

### Train against Azure simulation
```bash
python dast.py --dataset=azure
```
Uses the sklearn model as the black-box target instead of `net_m.pth`.

### Evaluate transfer attack performance
```bash
python evaluation.py --mode=dast --adv=FGSM --cuda
```

`--mode` options:
- `white` — white-box attack on the real target (best possible ASR, upper bound)
- `black` — attacks using `net_l.pth` trained on real data (baseline)
- `dast` — attacks using your trained DaST substitute

`--adv` options: `FGSM`, `BIM`, `PGD`, `CW`

Add `--target` for targeted attacks (random target class).

> **Note:** `evaluation.py` hardcodes `root='/data/dataset/'` — change to `root='dataset/'` to use the local MNIST copy.

> **Note:** `evaluation.py` loads `saved_model_2/netD_epoch_670.pth` for `dast` mode. Change to your actual saved model path after training.

---

## Key Hyperparameters

| Flag | Default | Effect |
|------|---------|--------|
| `--alpha` | 0.2 | Weight of label-control (diversity) loss |
| `--beta` | 0.1 | Weight of probability-matching loss; 0 = DaST-L, >0 = DaST-P |
| `--G_type` | 1 | Generator architecture variant |
| `--batchSize` | 500 | Batch size; **must be divisible by 10** (one chunk per class) |
| `--niter` | 2000 | Training epochs |
| `--lr` | 0.0001 | Adam learning rate |

---

## Training Your Own Model — Checklist

1. Install dependencies (see above)
2. Fix `sklearn.externals` import if needed
3. Fix dataset path in `evaluation.py` (`/data/dataset/` → `dataset/`)
4. Run `python dast.py --dataset=mnist --beta=0.1 --G_type=1 --batchSize=500`
5. Check `saved_model/` for checkpoint files as training progresses
6. Edit the `dast` branch in `evaluation.py` to load your checkpoint path
7. Run `python evaluation.py --mode=dast --adv=FGSM --cuda` and compare against `--mode=white` and `--mode=black`

---

## CIFAR-10 Notes

`dast_cifar10.py` extends the method to CIFAR-10 (32×32 RGB images).

- Target model: `pretrained/vgg16cifar10.pth` (VGG16, ~93.9% acc)
- Substitute model: VGG13 (`netD`)
- Uses `Generator_cifar10` (output size 32×32 with 3 channels)
- Attacks via `foolbox` L2BasicIterativeAttack
- **Training is unstable** — the log file `dast_cifar10.log` shows one collapsed run and one successful run that reached ~80% ASR at epoch 50 (DaST-P). The authors acknowledge this.

---

## Known Issues / Compatibility Warnings

| Issue | Location | Fix |
|-------|----------|-----|
| `sklearn.externals.joblib` removed | `dast.py` line 11 | Replace with `import joblib` |
| Dataset path hardcoded to cluster | `evaluation.py` line 59, `test_dast.py` line 58 | Change to `dataset/` |
| Saved model path hardcoded | `evaluation.py` line 204 | Update to your actual checkpoint |
| CUDA required | Throughout | All `.cuda()` calls are unconditional; CPU-only runs will fail |
| Performance is non-deterministic | All scripts | Authors note results vary machine to machine even with seeds |
| `xlwt` writes `.xls` output | `dast.py` | Log of per-epoch accuracy; benign |

---

## Results Context (from paper / logs)

- MNIST DaST-P: achieves competitive attack success rate vs. black-box baseline trained on real data
- Azure attack: 98.35% misclassification rate on Microsoft Azure remote model (from paper)
- CIFAR-10: ~80% ASR after ~50 epochs in a successful run (DaST-P, beta=0.1)
