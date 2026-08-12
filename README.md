# QwenDopamine
Testing the addition of "Dopamine" to LLMs as a way to improve agentic performance.

## Structure

- `configs/` — Hydra configurations for model, data, training, and experiments.
- `src/qwendopamine/` — package with models, training, data, distributed, evaluation, integrations, and CLI.
- `scripts/` — shell wrappers for common runs.
- `tests/` — shape, gradient, weight-loading, freezing, and checkpoint tests.
- `notebooks/` — inspection and activation analysis.
- `docs/` — architecture, experiments, checkpoint format.

## Setup

```bash
uv sync
uv run qwendopamine
```

## Notes

- Keep `logs/`, `data/`, and `checkpoints/` out of Git.
- Use pure PyTorch training loops; Hugging Face is for checkpoint/tokenizer interoperability.
