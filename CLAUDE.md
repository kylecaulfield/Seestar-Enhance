# CLAUDE.md

Guidance for Claude Code and other AI assistants working in this repository.

## Project Overview

**Seestar-Enhance** — a website for enhancing images captured by the Seestar smart telescope (a ZWO Seestar S30/S50 astrophotography device). Intended functionality is client/server tooling to process, stack, and enhance astrophotography images exported from Seestar devices.

## Current State

The repository is in its initial state: only `README.md` exists. No application code, build tooling, dependencies, tests, or CI are configured yet. This document should be kept in sync as the project takes shape — update it whenever the stack, layout, or workflows change.

## Repository Layout

```
.
├── README.md     # One-line project description
└── CLAUDE.md     # This file
```

No `src/`, `package.json`, `pyproject.toml`, `Dockerfile`, or CI configuration exists yet. When introducing any of these, document them here.

## Development Workflow

### Branching

- Do development on feature branches named like `claude/<short-description>-<suffix>` when working via Claude.
- Do not push to `main` directly.
- Create the branch locally if it does not exist.

### Commits

- Write short, descriptive commit subjects (imperative mood, e.g. "Add image upload handler").
- Use the body for the *why* when non-obvious.
- Never use `--no-verify` or skip hooks without explicit instruction.
- Prefer creating new commits over amending existing ones.

### Pushing

- Push with `git push -u origin <branch-name>`.
- On transient network failures, retry up to 4 times with exponential backoff (2s, 4s, 8s, 16s).
- Do not force-push to shared branches (especially `main`).

### Pull Requests

- Do **not** open a PR unless the user explicitly asks for one.
- GitHub interactions go through the GitHub MCP tools (the `gh` CLI is not available).
- Repository scope is limited to `kylecaulfield/seestar-enhance`.

## Conventions

### General

- Prefer editing existing files to creating new ones.
- Do not create documentation files (`*.md`) unless explicitly requested.
- Do not add emojis to code, commits, or docs unless requested.
- Keep comments minimal — only explain the *why* when non-obvious.
- Do not add speculative abstractions, unused features, or backwards-compatibility shims for code paths that do not exist.

### When adding a stack

When the codebase grows beyond `README.md`, update this file with:

- **Language/runtime versions** (Node, Python, etc.)
- **Package manager** (npm/pnpm/yarn, pip/uv/poetry)
- **Commands**: install, build, dev server, test, lint, type-check
- **Entry points** and directory structure
- **Environment variables** and where they are loaded
- **Deployment target** (static host, container, serverless, etc.)

### Likely shape (to validate before assuming)

Given the project goal — a website for enhancing Seestar images — expect either:
1. A JS/TS frontend (React/Next.js/Vite) with client-side image processing, or
2. A full-stack app with a Python backend (FastAPI/Flask) handling stacking/denoise pipelines (OpenCV, astropy, numpy), or
3. A static site with WASM-powered processing.

Do not assume which; confirm with the user before scaffolding.

## Testing & Verification

No tests exist yet. Once a framework is chosen, document:
- How to run the full suite
- How to run a single test
- Any required fixtures or environment setup

For UI work, start the dev server and exercise the feature in a browser before declaring the task done. If you cannot run the UI, say so — do not claim success from type-check/lint alone.

## Security Notes

- Treat uploaded image files as untrusted input. Validate MIME types and sizes at the boundary.
- Do not log image contents or user identifiers.
- Never commit `.env`, credentials, or API keys. If such files appear staged, warn the user.
