# MNIST Digit Classifier (PyTorch)

Small, original PyTorch project that trains a simple CNN on MNIST and saves accuracy/loss plots.

## How to run
```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py
```

Artifacts are saved into `artifacts/`:
- `mnist_cnn.pt` (model weights)
- `train_loss.png`
- `test_accuracy.png`

## What this shows
- Data loading with `torchvision.datasets.MNIST`
- A minimal CNN with 2 conv blocks
- Training loop + evaluation
- Saving model + plotting metrics