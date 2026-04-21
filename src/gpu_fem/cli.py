from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .presets import get_preset, list_gpu_presets, list_presets
from .workflow import run_gpu_fem


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        choices=["auto", "cuda", "cupy", "torch_cuda", "cpu"],
        default="auto",
        help="Requested backend/device. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="Maximum SIMP iterations when the preset does not override it.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of quality-check retries after the first attempt.",
    )
    parser.add_argument(
        "--controller",
        default="schedule",
        help="Parameter controller name passed to the workflow.",
    )
    parser.add_argument(
        "--router",
        default="rule",
        help="Router name for surrogate-enabled runs.",
    )
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        default=4096,
        help="Surrogate prediction batch size.",
    )
    parser.add_argument(
        "--use-large-net",
        action="store_true",
        help="Use the larger surrogate network variant.",
    )
    parser.add_argument(
        "--use-llm-eval",
        action="store_true",
        help="Enable the optional evaluator LLM path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-run workflow progress.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved run plan without executing it.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gpu_fem",
        description="Public command-line runner for the gpu_fem code release.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-presets",
        help="List available preset names.",
    )
    list_parser.add_argument(
        "--gpu-only",
        action="store_true",
        help="List only the GPU-scale presets.",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run one preset and write a report/artifacts directory.",
    )
    run_parser.add_argument("--preset", required=True, help="Preset name.")
    run_parser.add_argument(
        "--mode",
        choices=["surrogate", "fem"],
        default="surrogate",
        help="Run with surrogate assistance or FEM-only.",
    )
    run_parser.add_argument(
        "--output",
        required=True,
        help="Output directory for report.json and artifacts.",
    )
    _add_common_run_args(run_parser)

    suite_parser = subparsers.add_parser(
        "suite",
        help="Run a multi-case suite and aggregate summary files.",
    )
    suite_parser.add_argument(
        "--presets",
        nargs="*",
        help="Explicit preset names. Defaults to all GPU-scale presets.",
    )
    suite_parser.add_argument(
        "--modes",
        nargs="+",
        choices=["surrogate", "fem"],
        default=["surrogate", "fem"],
        help="Run modes to execute for each preset.",
    )
    suite_parser.add_argument(
        "--output-root",
        required=True,
        help="Root directory that will contain per-run folders plus suite summaries.",
    )
    suite_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue the suite when an individual run fails.",
    )
    _add_common_run_args(suite_parser)

    return parser


def _resolve_mode_router(mode: str, router: str) -> tuple[bool, str]:
    if mode == "fem":
        return False, "fem_only"
    return True, router


def _run_plan(args: argparse.Namespace, preset_name: str, mode: str, output_dir: Path) -> dict:
    use_surrogate, router = _resolve_mode_router(mode, args.router)
    spec = get_preset(preset_name)
    return {
        "preset": preset_name,
        "mode": mode,
        "output_dir": str(output_dir),
        "backend": args.backend,
        "max_iter": args.max_iter,
        "max_retries": args.max_retries,
        "controller": args.controller,
        "router": router,
        "use_surrogate": use_surrogate,
        "use_large_net": args.use_large_net,
        "gpu_batch_size": args.gpu_batch_size,
        "use_llm_eval": args.use_llm_eval,
        "nelx": spec.nelx,
        "nely": spec.nely,
        "nelz": spec.nelz,
        "volfrac": spec.volfrac,
    }


def _execute_run(args: argparse.Namespace, preset_name: str, mode: str, output_dir: Path) -> dict:
    use_surrogate, router = _resolve_mode_router(mode, args.router)
    spec = get_preset(preset_name)
    return run_gpu_fem(
        spec=spec,
        controller=args.controller,
        router=router,
        max_iter=args.max_iter,
        max_retries=args.max_retries,
        output_dir=str(output_dir),
        use_surrogate=use_surrogate,
        use_llm_eval=args.use_llm_eval,
        backend=args.backend,
        use_large_net=args.use_large_net,
        gpu_batch_size=args.gpu_batch_size,
        verbose=args.verbose,
    )


def _summary_row(report: dict, preset_name: str, mode: str, output_dir: Path) -> dict:
    solver_summary = report.get("solver_summary", {})
    surrogate_stats = report.get("surrogate_stats", {})
    evaluation = report.get("evaluation", {})
    return {
        "preset": preset_name,
        "mode": mode,
        "output_dir": str(output_dir),
        "gpu_backend": report.get("gpu_backend"),
        "requested_backend": report.get("requested_backend"),
        "passed": evaluation.get("passed"),
        "evaluation_summary": evaluation.get("summary"),
        "nelx": solver_summary.get("nelx"),
        "nely": solver_summary.get("nely"),
        "nelz": solver_summary.get("nelz"),
        "n_iter": solver_summary.get("n_iter"),
        "final_compliance": solver_summary.get("final_compliance"),
        "best_compliance": solver_summary.get("best_compliance"),
        "best_iteration": solver_summary.get("best_iteration"),
        "final_grayness": solver_summary.get("final_grayness"),
        "best_grayness": solver_summary.get("best_grayness"),
        "fem_calls": surrogate_stats.get("fem_calls"),
        "surrogate_calls": surrogate_stats.get("surrogate_calls"),
        "surrogate_used_fraction": surrogate_stats.get("surrogate_used_fraction"),
        "feature_mode": surrogate_stats.get("feature_mode"),
        "wall_time_sec": report.get("wall_time_sec"),
    }


def _write_suite_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"

    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    fieldnames = [
        "preset",
        "mode",
        "output_dir",
        "gpu_backend",
        "requested_backend",
        "passed",
        "evaluation_summary",
        "nelx",
        "nely",
        "nelz",
        "n_iter",
        "final_compliance",
        "best_compliance",
        "best_iteration",
        "final_grayness",
        "best_grayness",
        "fem_calls",
        "surrogate_calls",
        "surrogate_used_fraction",
        "feature_mode",
        "wall_time_sec",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cmd_list_presets(args: argparse.Namespace) -> int:
    presets = list_gpu_presets() if args.gpu_only else list_presets()
    for name in presets:
        print(name)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output)
    plan = _run_plan(args, args.preset, args.mode, output_dir)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    report = _execute_run(args, args.preset, args.mode, output_dir)
    print(json.dumps(_summary_row(report, args.preset, args.mode, output_dir), indent=2))
    return 0


def _cmd_suite(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    preset_names = args.presets if args.presets else list_gpu_presets()
    rows: list[dict] = []

    if args.dry_run:
        for mode in args.modes:
            for preset_name in preset_names:
                output_dir = output_root / mode / preset_name
                print(json.dumps(_run_plan(args, preset_name, mode, output_dir), indent=2))
        return 0

    for mode in args.modes:
        for preset_name in preset_names:
            output_dir = output_root / mode / preset_name
            try:
                report = _execute_run(args, preset_name, mode, output_dir)
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                rows.append(
                    {
                        "preset": preset_name,
                        "mode": mode,
                        "output_dir": str(output_dir),
                        "gpu_backend": None,
                        "requested_backend": args.backend,
                        "passed": False,
                        "evaluation_summary": f"run failed: {exc}",
                        "nelx": None,
                        "nely": None,
                        "nelz": None,
                        "n_iter": None,
                        "final_compliance": None,
                        "best_compliance": None,
                        "best_iteration": None,
                        "final_grayness": None,
                        "best_grayness": None,
                        "fem_calls": None,
                        "surrogate_calls": None,
                        "surrogate_used_fraction": None,
                        "feature_mode": None,
                        "wall_time_sec": None,
                    }
                )
                continue

            rows.append(_summary_row(report, preset_name, mode, output_dir))

    _write_suite_summary(output_root, rows)
    print(json.dumps(rows, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-presets":
        return _cmd_list_presets(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "suite":
        return _cmd_suite(args)

    parser.error(f"Unhandled command: {args.command}")
    return 2

