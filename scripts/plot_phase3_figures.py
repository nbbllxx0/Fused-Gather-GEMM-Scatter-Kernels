"""
plot_phase3_figures.py
----------------------
Generate CMAME paper figures from Phase 3 CSV/JSON data.

Figures produced:
  F3  scaling_fea.pdf      - Cold-start FEA solve wall time vs n_elem
  F4  scaling_simp.pdf     - SIMP-120 end-to-end wall time vs n_elem
  F5  speedup_bars.pdf     - Fused/FP64 and Fused/FP32 speedup bars per size
  F6  cg_residual.pdf      - Representative CG residual decay
  F7  simp_compliance.pdf  - Compliance and CG-iteration trajectories
  F8  external_bars.pdf    - Contextual PyTopo3D vs ours wall-time comparison
  F9  profiler_bars.pdf    - Synthetic hot-path gather/GEMM/scatter breakdown

All CSVs live under experiments/phase3/ and figures write to
papers/paper/figs/.
Run with no arguments: python scripts/plot_phase3_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "phase3"
OUT = ROOT / "figs"
OUT.mkdir(parents=True, exist_ok=True)
# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    }
)

COLORS = {
    "fp64": "#1f77b4",
    "fp32": "#ff7f0e",
    "fused": "#2ca02c",
    "bf16": "#d62728",
    "pytopo": "#7f7f7f",
}

PATH_LABELS = {
    "fp64": "FP64 three-stage baseline",
    "fp32": "FP32 three-stage baseline",
    "fused": "Fused FP32 kernel",
}

SIZE_LABELS = {
    "cantilever_gpu_medium": "64k",
    "cantilever_gpu_large": "216k",
    "cantilever_gpu_512k": "512k",
    "cantilever_gpu_1m": "1M",
}


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------
def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [WARN] missing: {path}")
        return []
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def merge_scaling_csvs(tags: list[str]) -> list[dict]:
    rows: list[dict] = []
    for tag in tags:
        rows.extend(load_csv(EXP / f"scaling_ladder_{tag}.csv"))
    return rows


def rows_by_mode_path(rows: list[dict]) -> dict[str, dict[str, list[tuple[int, float]]]]:
    out: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for r in rows:
        mode = r.get("mode", "?")
        path = r.get("path", "?")
        n = int(r.get("n_elem", 0))
        if mode.startswith("FEA"):
            t = float(r.get("mean_ms", "nan")) / 1000.0
        else:
            t = float(r.get("wall_s", "nan"))
        out.setdefault(mode, {}).setdefault(path, []).append((n, t))
    for mode in out:
        for path in out[mode]:
            out[mode][path].sort()
    return out


# -----------------------------------------------------------------------------
# F3 - cold-start FEA solve scaling
# -----------------------------------------------------------------------------
def fig3_fea_scaling(data: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    fea = data.get("FEA", {})
    if not fea:
        print("  [F3] no FEA rows found - skipping")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for path in ("fp64", "fp32", "fused"):
        pts = fea.get(path, [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        label = {
            "fp64": "FP64 three-stage baseline",
            "fp32": "FP32 three-stage baseline",
            "fused": "Fused FP32 kernel",
        }[path]
        ax.plot(xs, ys, "o-", label=label, color=COLORS[path])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Mean FEA solve wall time (s)")
    ax.set_title("Cold-start FEA solve scaling")
    ax.legend()
    fig.tight_layout()
    out = OUT / "F3_scaling_fea.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F3] {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# F4 - SIMP-120 end-to-end scaling
# -----------------------------------------------------------------------------
def fig4_simp_scaling(data: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    simp = next((v for k, v in data.items() if k.startswith("SIMP")), {})
    if not simp:
        print("  [F4] no SIMP rows found - skipping")
        return
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    for path in ("fp64", "fp32", "fused"):
        pts = simp.get(path, [])
        if not pts:
            continue
        xs, ys = zip(*pts)
        label = {
            "fp64": "FP64 three-stage baseline",
            "fp32": "FP32 three-stage baseline",
            "fused": "Fused FP32 kernel",
        }[path]
        ax.plot(xs, ys, "s-", label=label, color=COLORS[path])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("SIMP-120 end-to-end wall time (s)")
    ax.set_title("SIMP-120 end-to-end scaling")
    ax.legend()
    fig.tight_layout()
    out = OUT / "F4_scaling_simp.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F4] {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# F5 - speedup bars per size
# -----------------------------------------------------------------------------
def fig5_speedup_bars(data: dict[str, dict[str, list[tuple[int, float]]]]) -> None:
    simp = next((v for k, v in data.items() if k.startswith("SIMP")), {})
    if not simp or "fused" not in simp:
        print("  [F5] no SIMP rows - skipping")
        return
    fused = dict(simp["fused"])
    fp32 = dict(simp.get("fp32", []))
    fp64 = dict(simp.get("fp64", []))
    sizes = sorted(fused.keys())
    if not sizes:
        print("  [F5] no matching sizes")
        return

    x = np.arange(len(sizes))
    w = 0.35
    r32 = [fp32.get(n, np.nan) / fused[n] for n in sizes]
    r64 = [fp64.get(n, np.nan) / fused[n] for n in sizes]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar(x - w / 2, r64, w, label="FP64 / Fused", color=COLORS["fp64"])
    ax.bar(x + w / 2, r32, w, label="FP32 / Fused", color=COLORS["fp32"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n/1e6:.1f}M" if n >= 1e6 else f"{n/1e3:.0f}k" for n in sizes])
    ax.set_ylabel("SIMP-120 speedup")
    ax.set_title("Baseline-to-fused end-to-end speedup")
    ax.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.6)
    for xi, r in zip(x, r64):
        if np.isfinite(r):
            ax.text(xi - w / 2, r + 0.1, rf"{r:.1f}$\times$", ha="center", fontsize=8)
    for xi, r in zip(x, r32):
        if np.isfinite(r):
            ax.text(xi + w / 2, r + 0.1, rf"{r:.1f}$\times$", ha="center", fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = OUT / "F5_speedup_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F5] {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# F6 - CG residual decay
# -----------------------------------------------------------------------------
def fig6_cg_residual() -> None:
    path = EXP / "cg_residual_history.json"
    if not path.exists():
        print("  [F6] missing cg_residual_history.json - skipping")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    label_map = {
        "FP64 matfree": "FP64 three-stage baseline",
        "FP32 matfree": "FP32 three-stage baseline",
        "FUSED matfree": "Fused FP32 kernel",
    }
    for entry in data:
        label = label_map.get(entry.get("label", "path"), entry.get("label", "path"))
        res = entry.get("residuals", [])
        if not res:
            continue
        color = COLORS.get(entry.get("path", ""), None)
        ax.semilogy(np.arange(len(res)), res, label=label, color=color)
    title = data[0].get("problem_title", "CG residual decay") if data else "CG residual decay"
    ax.set_xlabel("CG iteration")
    ax.set_ylabel(r"$\|r_k\| / \|f\|$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    out = OUT / "F6_cg_residual.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F6] {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# F7 - SIMP compliance trajectory
# -----------------------------------------------------------------------------
def fig7_simp_compliance() -> None:
    jsons = list(EXP.glob("torsion_500k_*_history.json"))
    if not jsons:
        print("  [F7] no torsion history JSONs found - run benchmark_torsion.py first")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.0, 4.8), sharex=True)
    for j in sorted(jsons):
        path = j.stem.split("_")[-2]
        label = PATH_LABELS.get(path, path.upper())
        data = json.loads(j.read_text(encoding="utf-8"))
        hist = data.get("compliance_history", [])
        plog = data.get("params_log", [])
        it = np.arange(1, len(hist) + 1)
        ax1.plot(it, hist, label=label, color=COLORS.get(path, "k"))
        if plog:
            cg = [p.get("cg_iters", 0) for p in plog]
            ax2.plot(
                np.arange(1, len(cg) + 1),
                cg,
                label=label,
                color=COLORS.get(path, "k"),
                alpha=0.7,
            )
    ax1.set_ylabel("Compliance")
    ax1.set_title("Torsion SIMP-120 convergence (499,125 elements)")
    ax1.legend()
    ax1.set_yscale("log")
    ax2.set_xlabel("SIMP iteration")
    ax2.set_ylabel("CG iters")
    ax2.legend()
    fig.tight_layout()
    out = OUT / "F7_simp_compliance.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F7] {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# F8 - PyTopo3D external baseline comparison
# -----------------------------------------------------------------------------
def fig8_external_bars() -> None:
    path = EXP / "pytopo3d_vs_ours.csv"
    if not path.exists():
        print("  [F8] missing pytopo3d_vs_ours.csv - skipping")
        return

    rows: list[dict] = []
    for r in load_csv(path):
        code_val = r.get("code") or ""
        if code_val.startswith("#") or r.get("size_tag") is None:
            continue
        try:
            wall = float(str(r.get("wall_s", "")).lstrip("~"))
        except ValueError:
            continue
        rows.append({**r, "wall_s_num": wall})

    sizes = ["64k", "216k"]
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = np.arange(len(sizes))
    w = 0.35
    y_floor = 1.0
    pyt = [
        next((r["wall_s_num"] for r in rows if r["size_tag"] == s and "PyTopo3D" in r["code"]), np.nan)
        for s in sizes
    ]
    ours = [
        next((r["wall_s_num"] for r in rows if r["size_tag"] == s and "Ours" in r["code"]), np.nan)
        for s in sizes
    ]
    pyt_heights = [v - y_floor if np.isfinite(v) else np.nan for v in pyt]
    ours_heights = [v - y_floor if np.isfinite(v) else np.nan for v in ours]
    ax.bar(
        x - w / 2,
        pyt_heights,
        w,
        bottom=y_floor,
        label="PyTopo3D rerun (CPU/PyPardiso, 20-core)",
        color=COLORS["pytopo"],
    )
    ax.bar(
        x + w / 2,
        ours_heights,
        w,
        bottom=y_floor,
        label="Present work, fused FP32 (RTX 4090)",
        color=COLORS["fused"],
    )
    ax.set_yscale("log")
    finite_vals = [v for v in pyt + ours if np.isfinite(v)]
    if finite_vals:
        ax.set_ylim(y_floor, max(finite_vals) * 1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sizes)
    ax.set_ylabel("SIMP-120 wall time (s)")
    ax.set_title("Contextual wall-time comparison on the cantilever benchmark")
    ax.legend()
    for xi, (p, o) in enumerate(zip(pyt, ours)):
        if np.isfinite(p) and np.isfinite(o) and o > 0:
            ax.annotate(rf"{p / o:,.0f}$\times$", xy=(xi, max(p, o) * 1.4), ha="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    out = OUT / "F8_external_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F8] {out.relative_to(ROOT)}")


# -----------------------------------------------------------------------------
# F9 - synthetic hot-path breakdown
# -----------------------------------------------------------------------------
def fig9_profiler_bars() -> None:
    path = EXP / "profile_hotpath.csv"
    if not path.exists():
        print("  [F9] missing profile_hotpath.csv - skipping")
        return
    rows = load_csv(path)
    if not rows:
        print("  [F9] empty profile_hotpath.csv")
        return

    def f(row: dict, key: str) -> float:
        val = row.get(key) or ""
        try:
            return float(val)
        except ValueError:
            return float("nan")

    labels = [SIZE_LABELS.get(r["label"], r["label"]) for r in rows]
    gather = np.array([f(r, "t_gather_us") for r in rows])
    gemm64 = np.array([f(r, "t_gemm64_us") for r in rows])
    scat32 = np.array([f(r, "t_scatter_cur_us") for r in rows])
    full64 = np.array([f(r, "t_full64_us") for r in rows])
    fused32 = np.array([f(r, "t_fused_fp32_us") for r in rows])
    fusedbf = np.array([f(r, "t_fused_bf16_us") for r in rows])

    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w, gather, w, label="Gather", color="#c7c7c7")
    ax.bar(x - w, gemm64, w, bottom=gather, label="GEMM (FP64)", color=COLORS["fp64"])
    ax.bar(x - w, scat32, w, bottom=gather + gemm64, label="Scatter-add", color="#8c564b")
    ax.bar(x, fused32, w, label="Fused FP32 (single kernel)", color=COLORS["fused"])
    ax.bar(x + w, fusedbf, w, label="Fused BF16 (WMMA)", color=COLORS["bf16"])
    for xi, total in zip(x, full64):
        ax.hlines(total, xi - 1.35 * w, xi - 0.65 * w, colors="black", linewidth=1.6)
    ax.plot([], [], color="black", lw=1.6, label="FP64 full pipeline")

    for xi, (u, fu) in enumerate(zip(full64, fused32)):
        if np.isfinite(u) and np.isfinite(fu) and fu > 0:
            ax.annotate(
                rf"{u / fu:.1f}$\times$",
                xy=(xi, fu),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS["fused"],
            )
    for xi, (u, bf) in enumerate(zip(full64, fusedbf)):
        if np.isfinite(u) and np.isfinite(bf) and bf > 0:
            ax.annotate(
                rf"{u / bf:.1f}$\times$",
                xy=(xi + w, bf),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=COLORS["bf16"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time per matvec (\u00b5s)")
    ax.set_title("Synthetic hot-path breakdown")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    out = OUT / "F9_profiler_bars.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  [F9] {out.relative_to(ROOT)}")


# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("Generating Phase 3 paper figures...")
    rows = merge_scaling_csvs(["fea_mid", "fea_large", "fea_8m", "simp_mid", "simp_1m", "2m5m"])
    data = rows_by_mode_path(rows)
    print(f"  loaded {sum(len(v) for vv in data.values() for v in vv.values())} rows")
    fig3_fea_scaling(data)
    fig4_simp_scaling(data)
    fig5_speedup_bars(data)
    fig6_cg_residual()
    fig7_simp_compliance()
    fig8_external_bars()
    fig9_profiler_bars()
    print("Done.")


if __name__ == "__main__":
    main()

