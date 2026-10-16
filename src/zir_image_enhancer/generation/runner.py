"""Running one generation configuration over a list of garments.

Output layout, chosen so each candidate index can be scored by the official
``eval_val.py`` as if it were a team (it reads ``<preds>/<team>/<id>/front.png``)::

    <runs_root>/<run>/config.json
    <runs_root>/<run>/meta.jsonl            one line per candidate
    <runs_root>/<run>/cand_<k>/<id>/front.png
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from zir_image_enhancer.dataset import GarmentItem
from zir_image_enhancer.generation.fal_backend import (
    BudgetExceededError,
    GenerationRequest,
    Generator,
)
from zir_image_enhancer.generation.models import ModelSpec, get_model
from zir_image_enhancer.generation.prompts import build_prompt


class InputSelection(Enum):
    FIRST = "first"  # what the official baseline does
    ALL = "all"  # every photo of the garment, up to the model's reference limit


@dataclass(frozen=True)
class RunConfig:
    name: str
    model_key: str
    prompt_key: str
    inputs: InputSelection
    samples: int = 1
    seed_base: int | None = None

    def to_json(self) -> dict[str, object]:
        data = asdict(self)
        data["inputs"] = self.inputs.value
        return data


@dataclass(frozen=True)
class RunSummary:
    generated: int
    from_cache: int
    failed: int
    cost_usd: float
    seconds: float
    stopped_by_budget: bool


def select_inputs(
    item: GarmentItem, selection: InputSelection, model: ModelSpec
) -> tuple[Path, ...]:
    if selection is InputSelection.FIRST:
        return item.inputs[:1]
    return item.inputs[: model.max_references]


class GenerationRun:
    def __init__(self, config: RunConfig, generator: Generator | None, runs_root: Path) -> None:
        # generator can be None for planning only (dry runs), so no API key is needed.
        self._config = config
        self._generator = generator
        self._run_dir = runs_root / config.name
        self._model = get_model(config.model_key)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def plan(self, items: list[GarmentItem]) -> list[tuple[GarmentItem, GenerationRequest]]:
        planned: list[tuple[GarmentItem, GenerationRequest]] = []
        for item in items:
            images = select_inputs(item, self._config.inputs, self._model)
            prompt = build_prompt(self._config.prompt_key, len(images))
            for sample in range(self._config.samples):
                seed = None if self._config.seed_base is None else self._config.seed_base + sample
                planned.append((item, GenerationRequest(self._model, prompt, images, seed, sample)))
        return planned

    def estimated_cost_usd(self, items: list[GarmentItem]) -> float:
        """Upper bound: assumes nothing is cached yet."""
        return len(self.plan(items)) * self._model.est_cost_usd

    def execute(self, items: list[GarmentItem]) -> RunSummary:
        if self._generator is None:
            raise RuntimeError(
                "This run was built for planning only; pass a generator to execute it."
            )
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self._run_dir / "config.json").write_text(
            json.dumps(self._config.to_json(), indent=2), encoding="utf-8"
        )
        generated = cached = failed = 0
        cost = 0.0
        started = time.perf_counter()
        stopped_by_budget = False

        with (self._run_dir / "meta.jsonl").open("a", encoding="utf-8") as meta:
            for item, request in self.plan(items):
                entry: dict[str, object] = {
                    "item_id": item.item_id,
                    "sample": request.sample,
                    "model": self._model.key,
                    "prompt_key": self._config.prompt_key,
                    "seed": request.seed,
                    "inputs": [path.name for path in request.images],
                }
                try:
                    result = self._generator.generate(request)
                except BudgetExceededError as error:
                    # Stop the whole run: every later call would be refused anyway.
                    entry["error"] = str(error)
                    meta.write(json.dumps(entry) + "\n")
                    stopped_by_budget = True
                    break
                except Exception as error:  # noqa: BLE001
                    # One failed garment must not kill a batch; it is logged and skipped,
                    # like the official baseline does.
                    failed += 1
                    entry["error"] = f"{type(error).__name__}: {error}"
                    meta.write(json.dumps(entry) + "\n")
                    print(f"  ! {item.item_id} sample {request.sample}: {entry['error']}")
                    continue

                destination = (
                    self._run_dir / f"cand_{request.sample}" / item.item_id / f"{item.view}.png"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(result.png_bytes)
                cached += int(result.from_cache)
                generated += int(not result.from_cache)
                cost += result.cost_usd
                entry.update(
                    {
                        "from_cache": result.from_cache,
                        "cost_usd": result.cost_usd,
                        "seconds": round(result.seconds, 2),
                        "output": str(destination.relative_to(self._run_dir)),
                    }
                )
                meta.write(json.dumps(entry) + "\n")
                meta.flush()
                print(
                    f"  {item.item_id} sample {request.sample}: "
                    f"{'cache' if result.from_cache else f'{result.seconds:.1f}s'}"
                )

        return RunSummary(
            generated=generated,
            from_cache=cached,
            failed=failed,
            cost_usd=cost,
            seconds=time.perf_counter() - started,
            stopped_by_budget=stopped_by_budget,
        )
