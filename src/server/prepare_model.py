from __future__ import annotations

from pathlib import Path

from src.model import cached_weights_path, load_model, validate_model_output


def main() -> None:
    import os

    torch_home = Path(os.environ.get("TORCH_HOME", "/app/model_cache"))
    torch_home.mkdir(parents=True, exist_ok=True)
    model, load_ms = load_model("cpu", require_cached=False)
    validate_model_output(model)
    weights_path = cached_weights_path(torch_home)
    if not weights_path.is_file():
        raise RuntimeError(f"Weights were not cached at {weights_path}")
    print(f"EfficientNet-B0 weights ready at {weights_path}; load_ms={load_ms:.3f}")


if __name__ == "__main__":
    main()

