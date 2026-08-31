# Contributing

Thank you for helping make machine learning easier to understand. Improvements to explanations, accessibility, tests, examples, and visual behavior are all valuable.

## Before opening a change

1. Open an issue for a new module, major dependency, or architecture change.
2. Keep one pull request focused on one learner outcome.
3. Run `npm run verify` before submitting.
4. Include a screenshot or short recording when visual behavior changes.

## Writing a lesson

Copy an existing file in `src/content/lessons/`. Required frontmatter is validated at build time:

```yaml
title: A concrete learner-facing title
module: M2 · Autograd
order: 4
summary: One sentence describing the outcome.
colab: m2_autograd.ipynb
concepts: [graph, gradient]
duration: 10
draft: false
```

Prefer one mental model, one runnable example, one important caveat, and one prediction question. Define jargon before using it. State tensor shapes and units explicitly. Avoid presenting a browser simulation as if it were PyTorch itself.

## Building a widget

Put reusable mathematics in `src/lib/` and write tests first. Put interaction and presentation in `src/components/widgets/`. Every control needs a visible label; color cannot be the only signal; canvas content needs a textual description; animation must stop when the component unmounts.

## Commit and review expectations

Use descriptive commits. Pull requests should explain the learner problem, the chosen model or interaction, alternatives considered, and verification performed. Maintainers may request content review separately from code review.
