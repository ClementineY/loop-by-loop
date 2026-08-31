"""Generate the small, committed Colab notebooks linked by the lessons."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def build(filename: str, title: str, outcome: str, question: str, cells: list[dict]) -> None:
    notebook = {
        "cells": [
            markdown(f"# {title}\n\n**Outcome:** {outcome}\n\nRun cells with **Shift + Enter**. PyTorch is already installed in standard Colab runtimes."),
            markdown(f"## Experiment question\n\n> {question}\n\nBefore running code, write a prediction. Then observe the evidence, change one variable, and explain the difference."),
            code("import torch\nprint('PyTorch', torch.__version__)\nprint('device:', 'cuda' if torch.cuda.is_available() else 'cpu')"),
            *cells,
            markdown("## Reflection\n\n1. What did you predict?\n2. What evidence did the output provide?\n3. Which one variable did you change?\n4. How does the result connect to the lesson's mental model?"),
        ],
        "metadata": {
            "colab": {"name": filename, "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (ROOT / filename).write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    build("m1_tensors.ipynb", "M1 · Tensors", "Predict and verify tensor shapes.", "Can you predict every result shape before PyTorch prints it?", [
        code("x = torch.arange(6, dtype=torch.float32).reshape(2, 3)\nprint(x)\nprint('shape:', x.shape, 'dtype:', x.dtype, 'elements:', x.numel())"),
        code("# Broadcasting: center each feature across the batch\nmean = x.mean(dim=0)\ncentered = x - mean\nprint('mean shape:', mean.shape)\nprint(centered)"),
        markdown("## Try it\nChange the reshape target. Before running, decide whether the element count is preserved."),
    ])
    build("m2_autograd.ipynb", "M2 · Autograd", "Inspect a dynamic graph and its gradients.", "Can you derive the gradients before autograd reveals them?", [
        code("x = torch.tensor(2.)\nw = torch.tensor(-1., requires_grad=True)\nb = torch.tensor(.5, requires_grad=True)\ny = (x*w + b)**2\ny.backward()\nprint('y:', y.item(), 'dw:', w.grad.item(), 'db:', b.grad.item())"),
        code("# Gradients accumulate\ny = (x*w + b)**2\ny.backward()\nprint('after a second backward, dw:', w.grad.item())\nw.grad.zero_(); b.grad.zero_()"),
    ])
    build("m3_optimization.ipynb", "M3 · Optimization", "Fit a small curve with a complete training loop.", "How does the learning rate change convergence?", [
        code("x = torch.linspace(-2, 2, 80).unsqueeze(1)\ny = 1.5*x - .3 + .15*torch.randn_like(x)\nmodel = torch.nn.Linear(1, 1)\nloss_fn = torch.nn.MSELoss()\noptimizer = torch.optim.SGD(model.parameters(), lr=.08)"),
        code("for epoch in range(80):\n    optimizer.zero_grad()\n    loss = loss_fn(model(x), y)\n    loss.backward()\n    optimizer.step()\nprint('loss:', loss.item())\nprint('weight:', model.weight.item(), 'bias:', model.bias.item())"),
    ])
    build("m4_networks.ipynb", "M4 · Neural networks", "Build and inspect an nn.Module.", "Where does each trainable value live?", [
        code("class TinyNet(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = torch.nn.Sequential(\n            torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)\n        )\n    def forward(self, x):\n        return self.layers(x).squeeze(-1)\n\nmodel = TinyNet()\nprint(model)\nprint('parameters:', sum(p.numel() for p in model.parameters()))"),
    ])
    build("m5_data.ipynb", "M5 · Data", "Create a Dataset and inspect shuffled batches.", "What changes when shuffle or batch size changes?", [
        code("from torch.utils.data import TensorDataset, DataLoader\nX = torch.randn(100, 2)\ny = (X[:, 0] + X[:, 1] > 0).long()\ndataset = TensorDataset(X, y)\nloader = DataLoader(dataset, batch_size=16, shuffle=True)\nfeatures, labels = next(iter(loader))\nprint(features.shape, labels.shape)\nprint(labels.bincount(minlength=2))"),
    ])
    build("m6_convolution.ipynb", "M6 · Convolution", "Track shapes through a small CNN.", "Can you write the complete shape story without running the cell?", [
        code("model = torch.nn.Sequential(\n    torch.nn.Conv2d(3, 16, 3, padding=1),\n    torch.nn.ReLU(),\n    torch.nn.MaxPool2d(2),\n    torch.nn.Conv2d(16, 32, 3, padding=1),\n    torch.nn.ReLU(),\n    torch.nn.AdaptiveAvgPool2d(1),\n    torch.nn.Flatten(),\n    torch.nn.Linear(32, 10),\n)\nimages = torch.randn(8, 3, 32, 32)\nprint(model(images).shape)"),
    ])
    build("m7_training_well.ipynb", "M7 · Training well", "Compare training and validation behavior reproducibly.", "What must be fixed before two runs are comparable?", [
        code("import random, numpy as np\nseed = 42\nrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\nprint('seeded:', seed)"),
        markdown("## Experiment record\nWrite down your hypothesis, code revision, data split, seed, model size, optimizer, learning rate, batch size, and both training and validation curves."),
    ])
    build("m8_ship.ipynb", "M8 · Ship a model", "Save, restore, and verify a model checkpoint.", "Can another learner reproduce your evaluation from the saved artifact?", [
        code("model = torch.nn.Linear(4, 2)\nfixed_input = torch.randn(3, 4)\nexpected = model(fixed_input).detach()\ntorch.save(model.state_dict(), 'model.pt')\nrestored = torch.nn.Linear(4, 2)\nrestored.load_state_dict(torch.load('model.pt', weights_only=True))\nrestored.eval()\nactual = restored(fixed_input)\nprint('round trip matches:', torch.allclose(expected, actual))"),
        markdown("## Capstone checklist\nTrain a small CIFAR-10 CNN, preserve the best validation checkpoint, report a confusion matrix, inspect errors, and write a miniature model card."),
    ])


if __name__ == "__main__":
    main()
