from __future__ import annotations

from typing import Sequence

from absl import app

from diagnostics.diagnose_wrong_support_pretrained_direct_regression import (
    _CONFIG,
    diagnose,
)


def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Unexpected positional arguments.")

    cfg = _CONFIG.value
    cfg.mse.enable = False
    if int(cfg.task.num_rollout_episodes) <= 0:
        raise ValueError(
            "Simulation eval needs cfg.task.num_rollout_episodes > 0. "
            "Use configs/eval_wrong_support_pretrained_direct_regression.py or override it."
        )
    diagnose(cfg)


if __name__ == "__main__":
    app.run(main)
