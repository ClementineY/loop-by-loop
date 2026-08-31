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
        code("# Name axes before reducing them\nimages = torch.randn(8, 3, 32, 32)\nchannel_means = images.mean(dim=(0, 2, 3))\nprint('images:', images.shape, 'one mean per channel:', channel_means.shape)"),
        markdown("## Try it\nChange the reshape target. Before running, decide whether the element count is preserved."),
    ])
    build("m2_autograd.ipynb", "M2 · Autograd", "Inspect the autograd lifecycle and control where gradients stop.", "Which leaves will receive gradients, and which route will detach remove?", [
        markdown("## 1 · Record a forward trail\nOnly the values that may learn need `requires_grad=True`. Inspect `is_leaf` and `grad_fn` before calling backward."),
        code("x = torch.tensor(2.)\nw = torch.tensor(-1., requires_grad=True)\nb = torch.tensor(.5, requires_grad=True)\nloss = (x*w + b)**2\nprint('w is leaf:', w.is_leaf, 'w.grad:', w.grad)\nprint('loss is leaf:', loss.is_leaf, 'loss.grad_fn:', loss.grad_fn)"),
        markdown("## 2 · Walk backward and clear the answers\nThe graph is recorded during forward; `.backward()` fills gradients on the leaves."),
        code("loss.backward()\nprint('loss:', loss.item(), 'dw:', w.grad.item(), 'db:', b.grad.item())\nw.grad.zero_(); b.grad.zero_()\nprint('after clearing:', w.grad.item(), b.grad.item())"),
        markdown("## 3 · Detach one branch\nThe two modes produce the same forward values. Compare which parameters receive gradients."),
        code("x = torch.tensor(3.)\nteacher_w = torch.tensor(2., requires_grad=True)\nstudent_w = torch.tensor(1., requires_grad=True)\nteacher_prediction = teacher_w * x\ntarget = teacher_prediction.detach()\nstudent_prediction = student_w * x\nloss = (student_prediction - target).pow(2)\nloss.backward()\nprint('target:', target.item(), 'loss:', loss.item())\nprint('student grad:', student_w.grad.item())\nprint('teacher grad:', teacher_w.grad)  # None: detach cut this route"),
        markdown("## Try it\nRemove `.detach()`, rebuild the tensors, and run again. Predict the teacher gradient before printing it. Then wrap both model calls in `torch.no_grad()` and inspect `requires_grad` on their outputs."),
    ])
    build("m3_optimization.ipynb", "M3 · Optimization", "Compare search directions, learning rates, and momentum before fitting a model.", "Which information makes an update efficient and stable?", [
        markdown("## 1 · Random directions versus the gradient\nOn a quadratic, compare equal-length moves. As dimension grows, a random unit vector is usually poorly aligned with the negative gradient."),
        code("torch.manual_seed(7)\nfor dimension in [2, 10, 100, 1000]:\n    w = torch.ones(dimension) / dimension**0.5\n    directions = torch.randn(120, dimension)\n    directions = directions / directions.norm(dim=1, keepdim=True)\n    random_change = 0.1 * (directions @ w) + 0.5 * 0.1**2\n    gradient_change = 0.5 * (1 - 0.1)**2 - 0.5\n    print(dimension, 'random best:', random_change.min().item(), 'gradient:', gradient_change)"),
        markdown("## 2 · Learning rate and momentum\nInspect the gradient and the stored velocity separately. Predict the next parameter before running the loop."),
        code("p = torch.tensor(1.0)\nvelocity = torch.tensor(0.0)\nlr, beta = 0.1, 0.9\nfor step in range(3):\n    gradient = p.clone()       # gradient of 1/2 p²\n    velocity = beta * velocity + gradient\n    p = p - lr * velocity\n    print(step + 1, 'gradient:', gradient.item(), 'velocity:', velocity.item(), 'p:', p.item())"),
        markdown("## 3 · Put the optimizer inside a training loop"),
        code("x = torch.linspace(-2, 2, 80).unsqueeze(1)\ny = 1.5*x - .3 + .15*torch.randn_like(x)\nmodel = torch.nn.Linear(1, 1)\nloss_fn = torch.nn.MSELoss()\noptimizer = torch.optim.SGD(model.parameters(), lr=.08)"),
        code("for epoch in range(80):\n    optimizer.zero_grad()\n    loss = loss_fn(model(x), y)\n    loss.backward()\n    optimizer.step()\nprint('loss:', loss.item())\nprint('weight:', model.weight.item(), 'bias:', model.bias.item())"),
    ])
    build("m4_networks.ipynb", "M4 · Neural networks", "See why activations matter, then build and inspect an nn.Module.", "What changes forward values and what controls backward slopes?", [
        markdown("## 1 · One neuron: weighted votes, bias, activation\nInspect every contribution before assembling whole layers."),
        code("x = torch.tensor([1.5, -1.0])\nw = torch.tensor([0.8, -0.4])\nb = torch.tensor(0.2)\nz = x @ w + b\noutput = torch.relu(z)\nprint('contributions:', x*w, 'z:', z.item(), 'output:', output.item())"),
        markdown("## 2 · A linear stack still collapses\nVerify that two affine transformations equal one affine transformation, then insert ReLU and compare."),
        code("x = torch.linspace(-2, 2, 9)\nw1, b1, w2, b2 = 1.4, 0.2, -1.1, 0.4\nstacked = w2 * (w1*x + b1) + b2\ncollapsed = (w2*w1)*x + (w2*b1 + b2)\nbent = w2 * torch.relu(w1*x + b1) + b2\nprint('linear stack matches:', torch.allclose(stacked, collapsed))\nprint('linear:', stacked)\nprint('with ReLU:', bent)"),
        markdown("## 3 · Forward value and local slope\nUse autograd to compare how much gradient survives at several inputs."),
        code("for name, fn in [('ReLU', torch.relu), ('sigmoid', torch.sigmoid), ('tanh', torch.tanh)]:\n    x = torch.tensor([-6., 0., 6.], requires_grad=True)\n    y = fn(x)\n    y.sum().backward()\n    print(name, 'output:', y.detach().tolist(), 'slope:', x.grad.tolist())"),
        markdown("## 4 · Package the layers"),
        code("class TinyNet(torch.nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = torch.nn.Sequential(\n            torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)\n        )\n    def forward(self, x):\n        return self.layers(x).squeeze(-1)\n\nmodel = TinyNet()\nprint(model)\nprint('parameters:', sum(p.numel() for p in model.parameters()))"),
        markdown("## 5 · Logits belong with a stable loss\nVerify that a shared offset changes neither probabilities nor cross-entropy."),
        code("logits = torch.tensor([[1.2, -0.3], [101.2, 99.7]])\nprobs = logits.softmax(dim=1)\nprint(probs)\nprint('same probabilities:', torch.allclose(probs[0], probs[1]))\nprint('loss:', torch.nn.functional.cross_entropy(logits[:1], torch.tensor([0])).item())"),
    ])
    build("m5_data.ipynb", "M5 · Data", "Create a Dataset, inspect shuffled batches, and measure gradient noise.", "How does batch size change the gradient estimate?", [
        code("example_gradients = torch.tensor([-6., -4., -2., -1., 0., 1., 3., 4., 5., 6., 8., 10.])\nprint('full-data gradient:', example_gradients.mean().item())\nfor batch_size in [1, 2, 4, 12]:\n    print(batch_size, 'example batch estimate:', example_gradients[:batch_size].mean().item())"),
        code("from torch.utils.data import TensorDataset, DataLoader, random_split\nX = torch.randn(100, 2)\ny = (X[:, 0] + X[:, 1] > 0).long()\ndataset = TensorDataset(X, y)\ntrain_set, val_set, test_set = random_split(dataset, [70, 15, 15])\nloader = DataLoader(train_set, batch_size=16, shuffle=True)\nfeatures, labels = next(iter(loader))\nprint('splits:', len(train_set), len(val_set), len(test_set))\nprint(features.shape, labels.shape)"),
    ])
    build("m6_convolution.ipynb", "M6 · Convolution", "Track shapes through a small CNN.", "Can you write the complete shape story without running the cell?", [
        code("model = torch.nn.Sequential(\n    torch.nn.Conv2d(3, 16, 3, padding=1),\n    torch.nn.ReLU(),\n    torch.nn.MaxPool2d(2),\n    torch.nn.Conv2d(16, 32, 3, padding=1),\n    torch.nn.ReLU(),\n    torch.nn.AdaptiveAvgPool2d(1),\n    torch.nn.Flatten(),\n    torch.nn.Linear(32, 10),\n)\nimages = torch.randn(8, 3, 32, 32)\nprint(model(images).shape)"),
        code("conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)\nfeature_maps = conv(torch.randn(8, 3, 32, 32))\nprint('feature maps:', feature_maps.shape)\nprint('conv parameters:', sum(p.numel() for p in conv.parameters()))  # 448"),
    ])
    build("m7_training_well.ipynb", "M7 · Training well", "Compare training and validation behavior reproducibly.", "What must be fixed before two runs are comparable?", [
        code("import random, numpy as np\nseed = 42\nrandom.seed(seed); np.random.seed(seed); torch.manual_seed(seed)\nprint('seeded:', seed)"),
        markdown("## Diagnose with two curves\nTraining loss asks whether the model can fit; validation loss asks whether that fit transfers. Preserve the checkpoint at the best validation epoch, not necessarily the final epoch."),
        markdown("## Experiment record\nWrite down your hypothesis, code revision, data split, seed, model size, optimizer, learning rate, batch size, and both training and validation curves."),
        markdown("## Numerical and hardware limits\nMixed precision can accelerate eligible operations and reduce activation storage, but standard AMP commonly retains FP32 parameters and optimizer state. Measure examples per second and validation quality rather than assuming a speedup."),
        code("shape = (64, 128, 768)\nelements = torch.tensor(shape).prod().item()\nprint('one FP32 tensor (MB):', elements * 4 / 1e6)\nprint('one 16-bit tensor (MB):', elements * 2 / 1e6)\nbatch, step_ms = 64, 80\nprint('throughput:', batch / (step_ms / 1000), 'examples/s')"),
        markdown("## Quantization belongs in the right phase\nPost-training quantization changes the exported inference model. Quantization-aware training simulates rounding during forward passes so parameters can adapt to the error."),
    ])
    build("m8_ship.ipynb", "M8 · Ship a model", "Save, restore, and verify a model checkpoint.", "Can another learner reproduce your evaluation from the saved artifact?", [
        code("model = torch.nn.Linear(4, 2)\nfixed_input = torch.randn(3, 4)\nexpected = model(fixed_input).detach()\ntorch.save(model.state_dict(), 'model.pt')\nrestored = torch.nn.Linear(4, 2)\nrestored.load_state_dict(torch.load('model.pt', weights_only=True))\nrestored.eval()\nactual = restored(fixed_input)\nprint('round trip matches:', torch.allclose(expected, actual))"),
        markdown("## Resume training completely\nA resumable checkpoint also includes `optimizer.state_dict()`, epoch, configuration, and the metric used to select the best model."),
        markdown("## Capstone checklist\nTrain a small CIFAR-10 CNN, preserve the best validation checkpoint, report a confusion matrix, inspect errors, and write a miniature model card."),
    ])


if __name__ == "__main__":
    main()
