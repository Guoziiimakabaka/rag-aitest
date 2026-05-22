from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml

from task_v2.utils import ensure_output_dir, load_yaml_config, require_field


def _to_repo_relative(path: Path, root: Path) -> str:
    return Path(path.relative_to(root)).as_posix()


def _build_variant_repeat_paths(
    root: Path,
    output_dir: Path,
    variant_name: str,
    source_eval_json: Path,
    run_count: int,
    overwrite: bool,
) -> List[Path]:
    variant_dir = ensure_output_dir(output_dir / variant_name)
    run_paths: List[Path] = []
    for run_idx in range(run_count):
        run_path = variant_dir / f"run_{run_idx + 1}.json"
        if run_path.exists() and not overwrite:
            raise FileExistsError(
                f"Repeated run file already exists: {run_path}. "
                "Use --overwrite to replace existing files."
            )
        shutil.copy2(source_eval_json, run_path)
        run_paths.append(run_path.resolve())
    return run_paths


def build_repeated_run_config(
    config_path: Path,
    output_dir: Path,
    output_config_path: Path,
    run_count: int,
    overwrite: bool,
) -> Path:
    if run_count < 1:
        raise ValueError("run_count must be >= 1")

    root = Path(__file__).resolve().parents[2]
    config = load_yaml_config(config_path)
    variants = require_field(config, "variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("Config field 'variants' must be a non-empty list.")

    repeat_root = ensure_output_dir(output_dir.resolve())
    new_variants: List[Dict[str, Any]] = []

    for item in variants:
        if "name" not in item:
            raise KeyError("Each variant must contain name field.")
        if "eval_json" not in item:
            raise KeyError(
                "Each variant must contain eval_json for repeat-run config generation."
            )

        variant_name = str(item["name"])
        source_eval_json = (root / str(item["eval_json"])).resolve()
        if not source_eval_json.exists():
            raise FileNotFoundError(f"Missing eval_json file: {source_eval_json}")

        run_paths = _build_variant_repeat_paths(
            root=root,
            output_dir=repeat_root,
            variant_name=variant_name,
            source_eval_json=source_eval_json,
            run_count=run_count,
            overwrite=overwrite,
        )

        updated = dict(item)
        updated["eval_json_runs"] = [_to_repo_relative(path, root) for path in run_paths]
        updated["eval_json"] = updated["eval_json_runs"][0]
        new_variants.append(updated)

    output_config = dict(config)
    output_config["variants"] = new_variants
    experiment = dict(output_config.get("experiment", {}))
    experiment["repeat_run_count"] = run_count
    output_config["experiment"] = experiment

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    output_config_path.write_text(
        yaml.safe_dump(
            output_config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return output_config_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build repeated task-v2 config by cloning each variant eval_json into "
            "run_{i}.json files and wiring eval_json_runs."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/task_v2.yaml",
        help="Base task-v2 config path.",
    )
    parser.add_argument(
        "--run-count",
        type=int,
        default=3,
        help="How many repeated run files to generate for each variant.",
    )
    parser.add_argument(
        "--output-dir",
        default="code/rag/outputs/repeat_runs/latest",
        help="Output directory for repeated eval_json files.",
    )
    parser.add_argument(
        "--output-config",
        default="configs/task_v2.repeated.generated.yaml",
        help="Generated config path with eval_json_runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing repeated run files if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[2]
    config_path = (root / args.config).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_config = (root / args.output_config).resolve()

    generated = build_repeated_run_config(
        config_path=config_path,
        output_dir=output_dir,
        output_config_path=output_config,
        run_count=int(args.run_count),
        overwrite=bool(args.overwrite),
    )
    print("repeat_config_built=true")
    print(f"repeat_run_count={int(args.run_count)}")
    print(f"output_dir={output_dir}")
    print(f"output_config={generated}")


if __name__ == "__main__":
    main()
