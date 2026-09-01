# Loop by Loop

**From Python fluency to a real PyTorch training loop.**

Loop by Loop is the visual bridge from knowing Python to reading and writing a real PyTorch training loop. Explanations and browser simulations build intuition; matching Google Colab notebooks run genuine PyTorch without a local installation.

## Positioning

Loop by Loop does not try to out-cover executable textbooks or out-flex general-purpose neural-network sandboxes. Its job is narrower: make the path from familiar Python code to a complete, readable PyTorch training loop visual, inspectable, and runnable.

The learning sequence is the product: see a concept, manipulate a small model of it, run the corresponding PyTorch experiment, and then recognize the same idea inside the final loop.

## What is included

- 25 focused lessons from tensor shapes through a complete classifier
- Six accessible interactive labs for tensors, autograd, optimization, activations, training, and convolution
- Eight generated Colab notebooks containing runnable PyTorch examples
- A dedicated experiment catalog with one guided Colab investigation per module
- A framework-independent, unit-tested TypeScript learning engine
- Static Astro output suitable for GitHub Pages

## Local development

Requires Node.js 22.12 or newer and Python 3.10+ only when rebuilding notebooks.

```bash
npm ci
npm run dev
```

Open `http://localhost:4321/`.

```bash
npm test          # learning-engine unit tests
npm run notebooks:verify # notebook structure and references
npm run typecheck # Astro sync + TypeScript
npm run build     # production site
npm run verify    # all three
```

Generate the committed notebooks after changing `notebooks/build_notebooks.py`:

```bash
python3 notebooks/build_notebooks.py
```

## Architecture

The project is deliberately split into three layers:

1. `src/lib/` contains pure TypeScript implementations of the concepts being taught. It has no UI or framework dependencies and is covered by Vitest.
2. `src/components/widgets/` turns that logic into Svelte learning labs. Widgets own interaction and drawing, not the underlying mathematics.
3. `src/content/lessons/` contains the curriculum as MDX. Astro validates its frontmatter and generates routes, navigation, and metadata.

Real PyTorch stays in `notebooks/`. The browser code is an explanatory model, not a substitute runtime.

## Production

The production repository is `ClementineY/loop-by-loop`, separate from the account homepage repository. It publishes as a GitHub project page at `https://clementiney.github.io/loop-by-loop/`, leaving the root academic site untouched. The workflow derives its owner, canonical origin, and base path from the repository where it runs, so forks remain deployable without changing the build configuration.

To publish from a new GitHub account:

1. Create a repository and push this project to its `main` branch.
2. In **Settings → Pages**, choose **GitHub Actions** as the source.
3. Run **Deploy to GitHub Pages**, or push to `main`.

Both Pages layouts are supported: an account repository named `<owner>.github.io` is served at `/`, while any other repository is served at `/<repository>/`. Local and alternate-host builds remain configurable with `SITE_URL` and `BASE_PATH`.

See [Architecture](docs/ARCHITECTURE.md), [Inspiration and boundaries](INSPIRATION.md), [Contributing](CONTRIBUTING.md), and [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Code and original course content are available under the [MIT License](LICENSE).
