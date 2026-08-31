# Architecture

## Product boundary

Loop by Loop is a static learning product. It does not execute PyTorch in the browser. Browser simulations expose the smallest useful version of an idea; generated Colab notebooks contain the corresponding real PyTorch workflow.

This boundary keeps hosting inexpensive, lessons fast, and the educational logic inspectable.

## Dependency direction

```text
MDX lessons ──embed──> Svelte widgets ──call──> TypeScript learning engine
     │                                               │
     └──link──> generated Colab notebooks            └──tested by Vitest

Astro content layer ──validates/sorts──> routes, sidebar, previous/next
```

Dependencies point inward. `src/lib` must not import Svelte, Astro, DOM APIs, or canvas. A widget may import learning-engine functions. Lessons may embed widgets but should not reimplement their mathematics.

## Content model

Lesson frontmatter is the curriculum database. `module` and `order` determine the global course map; `draft` controls publication; `duration` and `concepts` support discovery and future search. A build fails on malformed content.

## Rendering model

Astro produces static HTML for prose and navigation. Only widgets marked with a `client:*` directive hydrate JavaScript. The course remains readable if JavaScript fails; interactions are enhancements rather than the only carrier of an explanation.

## Testing strategy

- Unit tests cover deterministic math and data behavior.
- Type checking covers Astro, MDX integration, and Svelte component contracts.
- Production builds validate content and every generated route.
- Notebook validation checks syntax, experiment structure, and references from both lessons and the experiment catalog.
- Future browser tests should cover keyboard interaction, link integrity, and one representative workflow per widget rather than duplicating math tests.

## Extension points

- Add a lesson by creating one MDX file with valid frontmatter.
- Add a simulation by defining and testing pure logic, then building a Svelte adapter.
- Add notebook material in `notebooks/build_notebooks.py`; generated `.ipynb` files remain committed so Colab URLs are stable.
- Override `SITE_URL` and `BASE_PATH` at build time for forks or alternate hosting.

## Production model

GitHub is both the collaboration/notebook source and the static host. Pull requests run the complete verification pipeline. A push to `main` repeats those checks, builds with an owner- and repository-derived Pages path, uploads one immutable artifact, and deploys it through GitHub's protected `github-pages` environment.

The workflow handles account Pages (`/`) and project Pages (`/<repository>/`) without source edits. The static architecture remains the default; a backend should be introduced only when a validated feature requires server-side state or execution.

## Explicit non-goals for v1

Accounts, synchronized progress, a Python execution backend, analytics, search, and internationalization are deferred. Each adds operational or privacy cost and should be introduced only with a documented learner need.
