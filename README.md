# Lere's Corner

Source for [lere01.github.io](https://lere01.github.io/), Faith O. Oyedemi's
personal site for scientific software, quantum simulation, machine learning,
and neuronal dynamics.

The site brings together:

- research notes on neural quantum states, variational Monte Carlo, and
  computational physics;
- interactive demonstrations, including the Sampling Playground;
- scientific software projects written in Rust, Python, and WebAssembly; and
- a professional profile covering research, software engineering, and
  technical teaching.

## Technology

The site is built with [Hugo](https://gohugo.io/) and the
[Congo](https://jpanther.github.io/congo/) theme. Its house styles come from a
pinned release of the [`lere01/design`](https://github.com/lere01/design)
design system. GitHub Actions builds and deploys the production site to GitHub
Pages whenever `main` is pushed.

Content lives in `content/`, project cards in `data/projects.yaml`, and the
custom homepage in `layouts/partials/home/custom.html`.

## Local development

Requirements:

- Hugo Extended 0.163.3
- Go
- Git

Check the local toolchain and download the Hugo module:

```bash
make doctor
make deps
```

Start a live preview that includes draft posts:

```bash
make serve
```

Preview only production-eligible content:

```bash
make serve-production
```

## Writing

Create a new post bundle:

```bash
make new SLUG=my-article-title
```

Posts remain excluded from production while their front matter contains
`draft: true`. List all drafts with:

```bash
make list-drafts
```

Place article-specific images and other resources beside the article's
`index.md` file.

## Validation and publishing

Validate all content, including drafts:

```bash
make check
```

Build the minified production site into `public/`:

```bash
make build
```

Publishing is intentionally restricted to a clean `main` branch. Commit all
intended changes, then run:

```bash
make publish PUBLISH=1
```

The command performs a clean validation and production build before pushing
`main`. GitHub Actions then deploys the generated site. The `public/` directory
is generated output and should not be committed.

## Repository structure

```text
.
├── assets/                  # Site images and custom CSS
├── config/_default/         # Hugo and theme configuration
├── content/                 # Homepage, résumé, playground, and articles
├── data/projects.yaml       # Homepage project cards
├── layouts/partials/        # Custom homepage and head extensions
├── .github/workflows/       # GitHub Pages deployment
└── Makefile                 # Development and publishing commands
```
