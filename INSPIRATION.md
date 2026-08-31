# Inspiration and project boundaries

Loop by Loop sits between an interactive visualization and an executable textbook. It owes a conceptual debt to several open-source projects that demonstrated different parts of that experience.

## Projects we learn from

- [TensorFlow Playground](https://github.com/tensorflow/playground) demonstrates how a single coherent model state can connect architecture, hyperparameters, training curves, and a decision boundary.
- [CNN Explainer](https://github.com/poloclub/cnn-explainer) demonstrates progressive disclosure: learners can move between a network overview, a selected operation, its numerical inputs, and the corresponding explanation.
- [Dive into Deep Learning](https://github.com/d2l-ai/d2l-en) demonstrates the editorial and contribution systems required to maintain an open technical curriculum at scale.
- [Torchvista](https://github.com/sachinhosmani/torchvista) demonstrates how actual PyTorch execution graphs can become navigable learning and debugging artifacts inside notebooks.
- [Neural Nets](https://github.com/NAME0x0/neural-nets) demonstrates a browser-native curriculum where equations, plain-language explanations, and direct-manipulation experiments reinforce one another.
- [PyTorch Neural Network Simulation](https://github.com/romizone/pytorch) explores live training, weight inspection, and the generation of corresponding PyTorch code.

## What this project does differently

Our scope is deliberately narrower than a deep-learning textbook and broader than a single playground:

1. Teach the shortest useful path from Python familiarity to reading and writing a real PyTorch training loop.
2. Give every foundational abstraction a small browser experiment that remains understandable without framework internals.
3. Follow each conceptual experiment with a matching, runnable PyTorch notebook in Colab.
4. Treat tensor shapes, gradient flow, debugging, reproducibility, and model handoff as first-class learning outcomes.

The browser simulations are explanatory models, not reimplementations of PyTorch. Real framework behavior is demonstrated and verified in the notebooks.

## Reuse policy

This repository's lessons, visual design, and implementations are original unless a file states otherwise. The projects above are references for interaction and editorial patterns; no upstream prose, diagrams, or source code should be copied without reviewing and complying with that project's current license.

When a contribution adapts an external algorithm, dataset, exercise, or meaningful code fragment, document the source and license in the pull request and add an attribution near the adapted material when required.
