"""
collect_numbers.py
------------------
Single audit table of every quantitative claim the accompanying paper
makes, each with the results file it comes from.

This exists because the worst kind of wrong number is an untraceable one.
The hazard is concrete: three files recording the same nominal 216k
configuration can disagree by a factor of nine -- 17.5 s, 39.6 s and 87.5 s for
the same fused path -- and if nothing connects a quoted number to the file
behind it, the disagreement is invisible to the authors and obvious to anyone
who downloads the data.  This table removes that gap: every quantitative
claim is recomputed here from the file it comes from.

Output: results/NUMBERS.md and results/NUMBERS.json.

Usage:
    python scripts/collect_numbers.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))

PATH_LABEL = {
    "three_stage_fp64": "three-stage FP64",
    "three_stage_fp32": "three-stage FP32",
    "three_stage_fp32_s64": "three-stage FP32 (FP64 scatter)",
    "fused_fp64": "fused FP64",
    "fused_fp32": "fused FP32",
    "fused_ai_fp64": "fused FP64 + analytic index",
    "fused_ai_fp32": "fused FP32 + analytic index",
    "node_fp64": "node-owned FP64",
    "node_fp32": "node-owned FP32",
    "ir_fused_fp32": "refined fused FP32",
    "ir_fused_ai_fp32": "refined fused FP32 + analytic index",
    "ir_node_fp32": "refined node-owned FP32",
}


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def rel(p):
    return os.path.relpath(p, _ROOT).replace("\\", "/")


def main():
    out = {"generated_from": [], "sections": {}}
    md = ["# Number audit",
          "",
          "Every quantitative claim the accompanying paper makes, with",
          "the results file it is computed from. Regenerate with",
          "`python scripts/collect_numbers.py`.",
          ""]

    # -- environment ------------------------------------------------------
    env = load(os.path.join(_ROOT, "results", "G0", "environment.json"))
    if env:
        out["sections"]["environment"] = env
        out["generated_from"].append(rel(os.path.join(
            _ROOT, "results", "G0", "environment.json")))
        md += ["## Environment", "",
               "| item | value |", "|---|---|"]
        for k, v in env.items():
            if isinstance(v, (str, int, float, bool)):
                md.append(f"| {k} | {v} |")
        md.append("")

    # -- operator parity --------------------------------------------------
    for name in ("operator_parity.json", "operator_parity_large.json"):
        p = os.path.join(_ROOT, "results", "G1", name)
        d = load(p)
        if not d:
            continue
        out["sections"].setdefault("parity", {})[name] = d
        out["generated_from"].append(rel(p))
    par = out["sections"].get("parity", {})
    if par:
        md += ["## Operator parity", "",
               "| mesh | path | rel L2 vs three-stage FP64 | bar | pass |",
               "|---|---|---|---|---|"]
        for _, d in sorted(par.items()):
            for tag, e in d.get("sizes", {}).items():
                for k, v in e.get("checks", {}).items():
                    if isinstance(v, dict) and "rel_l2" in v:
                        md.append(f"| {tag} | {PATH_LABEL.get(k, k)} | "
                                  f"{v['rel_l2']:.3e} | {v['bar']:.0e} | "
                                  f"{'yes' if v['pass'] else 'NO'} |")
        md.append("")
        md += ["### Determinism", "",
               "| mesh | path | bitwise identical over 5 repeats | max abs spread |",
               "|---|---|---|---|"]
        for _, d in sorted(par.items()):
            for tag, e in d.get("sizes", {}).items():
                for k, v in (e.get("checks", {}).get("determinism") or {}).items():
                    md.append(f"| {tag} | {PATH_LABEL.get(k, k)} | "
                              f"{'yes' if v['bitwise_identical'] else 'no'} | "
                              f"{v['max_abs_spread']:.3e} |")
        md.append("")

    # -- SIMP verification -------------------------------------------------
    p = os.path.join(_ROOT, "results", "G1", "simp_verification.json")
    d = load(p)
    if d:
        out["sections"]["simp_verification"] = d
        out["generated_from"].append(rel(p))
        md += ["## SIMP driver verification", "",
               f"- filter adjoint identity, relative error: "
               f"**{d['filter_adjoint_rel_err']:.3e}**",
               f"- fused adjoint kernel vs materialised three-stage, rel L2: "
               f"**{d.get('elem_compliance_fused_vs_three_stage_rel_l2', float('nan')):.3e}**",
               f"- finite-difference sensitivity check, worst relative error "
               f"over {len(d['fd_check']['rows'])} probed elements: "
               f"**{d['fd_check']['worst_rel_err']:.3e}** (h = "
               f"{d['fd_check']['h']:g})",
               f"- achieved physical volume fraction, max deviation from the "
               f"prescribed value: **{d['volume_max_abs_err']:.2e}**",
               f"- cross-path compliance spread: "
               f"**{d['cross_path_compliance_spread']:.3e}**", ""]

    # -- traffic model -----------------------------------------------------
    try:
        from gpu_fem.cuda_operators import traffic_terms, PATH_ARRAYS
        md += ["## Traffic model (logical and compulsory bounds)", "",
               "| path | vectors | K_e | E_e | index | logical B/elem | "
               "compulsory B/elem | FLOP/elem | I_logical | I_compulsory |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        tt = {}
        for pth in PATH_ARRAYS:
            t = traffic_terms(pth)
            tt[pth] = t
            md.append(
                f"| {PATH_LABEL.get(pth, pth)} | {t['vectors']} | {t['KE']} | "
                f"{t['E_e']} | {t['index']} | {t['logical_B_per_elem']} | "
                f"{t['compulsory_B_per_elem']} | {t['FLOP_per_elem']} | "
                f"{t['FLOP_per_elem']/t['logical_B_per_elem']:.2f} | "
                f"{t['FLOP_per_elem']/t['compulsory_B_per_elem']:.2f} |")
        md.append("")
        out["sections"]["traffic_model"] = tt
    except Exception as ex:                                     # noqa: BLE001
        md += [f"(traffic model unavailable: {ex!r})", ""]

    # -- operator microbenchmark ------------------------------------------
    p = os.path.join(_ROOT, "results", "G3", "operator_benchmark.json")
    d = load(p)
    if d:
        out["sections"]["operator_benchmark"] = d
        out["generated_from"].append(rel(p))
        rows = [r for r in d["rows"] if "us_median" in r]
        md += ["## Operator microbenchmark", "",
               f"Protocol: {d.get('timing_protocol', '')}", "",
               "| mesh | path | us/application (median) | IQR % | range % | "
               "speedup vs three-stage FP64 |", "|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['size']} | {PATH_LABEL.get(r['path'], r['path'])} "
                      f"| {r['us_median']:.1f} | {r.get('iqr_pct', float('nan')):.1f} "
                      f"| {r.get('spread_pct', float('nan')):.1f} | "
                      f"{r.get('speedup_vs_three_stage_fp64', float('nan')):.2f} |")
        md.append("")

    # -- end-to-end ladder -------------------------------------------------
    for tag in ("s1",):
        p = os.path.join(_ROOT, "results", "G6", f"ladder_{tag}.json")
        d = load(p)
        if not d:
            continue
        out["sections"]["ladder"] = {"tag": tag, "rows": d}
        out["generated_from"].append(rel(p))
        # Median wall times come from the repeats file; a single run is not a
        # timing measurement and the audit trail must quote the same number
        # the manuscript prints.
        preps = os.path.join(_ROOT, "results", "G6", "ladder_rep.json")
        reps = load(preps) or []
        if reps:
            out["generated_from"].append(rel(preps))
        for r in d:
            w = sorted(x["wall_s"] for x in reps
                       if x["size"] == r["size"] and x["path"] == r["path"])
            if w:
                r["wall_s"] = (w[len(w) // 2] if len(w) % 2 else
                               0.5 * (w[len(w) // 2 - 1] + w[len(w) // 2]))
                r["n_reps"] = len(w)
        good = [r for r in d if r.get("final_compliance") is not None
                and not r.get("invalid") and not r.get("oom")]
        bad = [r for r in d if r.get("invalid")]
        # Count, do not assert. This sentence read "Every row terminated on
        # the declared outer-convergence criterion" while the table beneath it
        # carried three rows marked `guard` -- untrue of the one
        # document whose entire purpose is to be the audit trail. Deriving it
        # from the rows means it cannot drift from them again.
        n_conv = sum(1 for r in good if r.get("outer_converged"))
        n_guard = len(good) - n_conv
        conv_line = (
            f"{n_conv} of {len(good)} valid rows terminated on the declared "
            "outer-convergence criterion"
            + (f"; the other {n_guard} stopped on the iteration guard, "
               "are marked `guard` in the converged column, and are not "
               "converged optimizations" if n_guard else "")
            + ". Every linear solve in every valid row reached the 1e-5 "
              "relative residual tolerance, measured on the true residual "
              "||b-Ax||/||b||."
            + (f" A further {len(bad)} run(s) are invalid: their first "
               "equilibrium solve could not reach the tolerance before the "
               "iteration cap, and they are listed separately below with the "
               "residual each one attained." if bad else ""))
        md += [f"## End-to-end optimization runs (`{tag}`)", "",
               conv_line, "",
               "| mesh | path | outer iters | converged | compliance | "
               "V_phys | grayness | CG total | max resid | wall s | "
               "solve share | peak GiB |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(good, key=lambda r: (r.get("n_elem", 0),
                                             r.get("path", ""))):
            md.append(
                f"| {r['size']} | {PATH_LABEL.get(r['path'], r['path'])} | "
                f"{r['outer_iters']} | "
                f"{'yes' if r.get('outer_converged') else r.get('stop_reason')} | "
                f"{r['final_compliance']:.6f} | {r['final_vol_phys']:.4f} | "
                f"{r['final_grayness']:.2e} | {r['total_cg_iters']} | "
                f"{r['max_rel_resid']:.2e} | {r['wall_s']:.1f} | "
                f"{r.get('linear_solve_share', float('nan')):.2f} | "
                f"{(r.get('memory') or r).get('peak_device_used_GiB', float('nan')):.2f} |")
        md.append("")

        if bad:
            md += ["### Runs invalidated by the fail-closed rule", "",
                   "These are not slow results; they are non-results. Each "
                   "exhausted the 20,000-iteration cap on its first "
                   "equilibrium solve without reaching 1e-5 on the true "
                   "residual, so no design iteration was ever taken.", "",
                   "| mesh | path | CG iters | residual attained | wall s |",
                   "|---|---|---|---|---|"]
            for r in sorted(bad, key=lambda r: (r.get("size", ""),
                                                r.get("path", ""))):
                m = re.search(r"achieved relative residual ([0-9.eE+-]+) "
                              r"after (\d+) iterations", r.get("reason", ""))
                res = f"{float(m.group(1)):.3e}" if m else "--"
                its = m.group(2) if m else "--"
                md.append(f"| {r['size']} | "
                          f"{PATH_LABEL.get(r['path'], r['path'])} | {its} | "
                          f"{res} | {r.get('wall_s', float('nan')):.1f} |")
            md.append("")
            out["sections"]["ladder_invalid"] = bad

        # Decomposed effects. Every pair must be two runs that both reached
        # the tolerance and both converged the design; a ratio taken across a
        # convergence failure measures nothing, which is why precision enters
        # only through refinement.
        by = {(r["size"], r["path"]): r for r in good}
        sizes = sorted({r["size"] for r in good},
                       key=lambda s: min(r["n_elem"] for r in good
                                         if r["size"] == s))
        effects = [("fusion at matched FP64", "three_stage_fp64",
                    "fused_fp64"),
                   ("mixed precision, by refinement", "fused_fp64",
                    "ir_fused_fp32"),
                   ("analytic indexing, inside refinement", "ir_fused_fp32",
                    "ir_fused_ai_fp32"),
                   ("node ownership, inside refinement", "ir_fused_ai_fp32",
                    "ir_node_fp32"),
                   ("all four, compounded", "three_stage_fp64",
                    "ir_node_fp32")]
        md += ["### Decomposed end-to-end effect (wall-time ratio)", "",
               "Medians over repeated complete optimizations.", "",
               "| effect | " + " | ".join(sizes) + " |",
               "|---" * (len(sizes) + 1) + "|"]
        eff_out = {}
        for lab, a, b in effects:
            cells = []
            for s in sizes:
                ra, rb = by.get((s, a)), by.get((s, b))
                if (ra and rb and rb["wall_s"] > 0
                        and ra.get("outer_converged")
                        and rb.get("outer_converged")):
                    v = ra["wall_s"] / rb["wall_s"]
                    cells.append(f"{v:.2f}x")
                    eff_out.setdefault(lab, {})[s] = v
                else:
                    cells.append("--")
            md.append(f"| {lab} | " + " | ".join(cells) + " |")
        md.append("")
        out["sections"]["decomposed_effects"] = eff_out
        break

    # -- cold start --------------------------------------------------------
    p = os.path.join(_ROOT, "results", "G6", "coldstart_ladder.json")
    d = load(p)
    if d:
        out["sections"]["coldstart"] = d
        out["generated_from"].append(rel(p))
        md += ["## Converged cold-start FEA ladder", "",
               "Iterations and achieved residual for every point.", "",
               "| mesh | path | CG iters | achieved rel. residual | at cap | "
               "time to tolerance s | us per application |",
               "|---|---|---|---|---|---|---|"]
        for r in d:
            if "cg_iters" not in r:
                continue
            md.append(f"| {r['size']} | {PATH_LABEL.get(r['path'], r['path'])} "
                      f"| {r['cg_iters']} | {r['rel_resid']:.3e} | "
                      f"{'YES' if r.get('at_cap') else 'no'} | "
                      f"{r['wall_to_tolerance_s']:.2f} | "
                      f"{r['us_per_matvec']:.1f} |")
        md.append("")

    # -- mesh comparability ------------------------------------------------
    p = os.path.join(_ROOT, "results", "G6", "ladder_comparability.json")
    d = load(p)
    if d:
        out["sections"]["comparability"] = d
        out["generated_from"].append(rel(p))
        md += ["## Mesh-comparability series", "",
               "Fixed physical filter radius and patch area, run on both the "
               "double-precision control and the refined path. No "
               "mesh-refinement claim is drawn from this series: no mesh "
               "satisfies the design criterion within the guard.", "",
               "| mesh | path | elements | rmin (elem) | filter nbrs | "
               "compliance | V_phys | outer iters | design converged |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(d, key=lambda r: (r.get("n_elem", 0),
                                          r.get("path", ""))):
            md.append(f"| {r['size']} | "
                      f"{PATH_LABEL.get(r['path'], r['path'])} | "
                      f"{r['n_elem']} | {r.get('rmin_final', float('nan')):.2f}"
                      f" | {r.get('filter_neighbours_per_elem','--')} | "
                      f"{r['final_compliance']:.10f} | "
                      f"{r['final_vol_phys']:.6f} | {r['outer_iters']} | "
                      f"{'yes' if r.get('outer_converged') else 'no'} |")
        md.append("")

    # -- condition numbers -------------------------------------------------
    p = os.path.join(_ROOT, "results", "G6", "kappa_estimation.json")
    d = load(p)
    if d:
        out["sections"]["kappa"] = d
        out["generated_from"].append(rel(p))
        md += ["## Condition-number bounds", "",
               "Ritz values of a Lanczos run lie inside the spectrum, so the "
               "ratio bounds kappa from below. The power-iteration column is "
               "an independent check that the run resolved the top of the "
               "spectrum.", "",
               "| mesh | penal | theta_max | lambda_max (power) | ratio | "
               "theta_min | kappa >= | eps_bf16*kappa >= | steps |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in d:
            md.append(f"| {r['size']} | {r['penal']:g} | "
                      f"{r['theta_max']:.6f} | {r['lam_max_power']:.6f} | "
                      f"{r['theta_max_over_power']:.4f} | "
                      f"{r['theta_min']:.4e} | "
                      f"{r['kappa_lower_bound']:.4e} | "
                      f"{r['eps_bf16_kappa_lb']:.4e} | "
                      f"{r['lanczos_steps']} |")
        md.append("")

    # -- BF16 precision boundary -------------------------------------------
    p = os.path.join(_ROOT, "results", "G6", "bf16_study.json")
    d = load(p)
    if d:
        out["sections"]["bf16"] = d
        out["generated_from"].append(rel(p))
        md += ["## BF16 precision boundary", "",
               "Reference is an FP64 solve verified to converge. Note the two "
               "error columns disagree by orders of magnitude on the same "
               "solve: the residual is what the acceptance rule tests.", "",
               "| mesh | solver | CG iters | true rel. residual | compliance | "
               "rel. error in c |", "|---|---|---|---|---|---|"]
        for r in d:
            if "reference" not in r:
                continue
            ref = r["reference"]
            md.append(f"| {r['size']} | FP64 reference ({ref['solver']}) | "
                      f"{ref['cg_iters']} | {ref['rel_resid']:.3e} | "
                      f"{ref['compliance']:.6f} | -- |")
            for key, lab in (("plain_bf16", "plain BF16 CG"),
                             ("bf16_ir_0.001", "BF16 refined, inner 1e-3"),
                             ("bf16_ir_1e-05", "BF16 refined, inner 1e-5")):
                v = r.get(key)
                if not v:
                    continue
                md.append(f"| | {lab} | {v['cg_iters']} | "
                          f"{v['rel_resid']:.3e} | {v['compliance']:.6f} | "
                          f"{v['compliance_rel_err']:.4f} |")
        md.append("")

    # -- floor sweep -------------------------------------------------------
    for p in sorted(glob.glob(os.path.join(_ROOT, "results", "G6",
                                           "floor_sweep_*.json"))):
        d = load(p)
        if not d:
            continue
        out["sections"]["floor_sweep"] = d
        out["generated_from"].append(rel(p))
        md += [f"## Stiffness-floor sweep ({d['size']})", "",
               "| Emin/E0 | outer iters | mean CG/solve | max CG | "
               "compliance | V_phys | grayness | binary mismatch vs ref | "
               "min projected density | min element stiffness |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in sorted(d["runs"].items(), key=lambda kv: float(kv[0])):
            if v.get("invalid"):
                md.append(f"| {k} | INVALID: {v.get('reason','')[:60]} | | | "
                          f"| | | | | |")
                continue
            md.append(
                f"| {k} | {v['outer_iters']} | {v['mean_cg_per_solve']:.1f} | "
                f"{v['max_cg_single_solve']} | {v['final_compliance']:.6f} | "
                f"{v['final_vol_phys']:.4f} | {v['final_grayness']:.2e} | "
                f"{v.get('binary_mismatch_vs_ref', float('nan')):.5f} | "
                f"{v['rho_phys_min']:.3e} | {v['E_min_achieved']:.3e} |")
        md.append("")
        fd = d.get("fd_check", {})
        if fd:
            md += ["Finite-difference sensitivity spot check per floor:", ""]
            for k, v in sorted(fd.items(), key=lambda kv: float(kv[0])):
                if v.get("ok"):
                    md.append(f"- Emin/E0 = {k}: worst relative error "
                              f"{v['worst_rel_err']:.3e}")
            md.append("")

    # -- DRAM measurement status ------------------------------------------
    p = os.path.join(_ROOT, "results", "G3", "dram_measured.json")
    d = load(p)
    if d:
        out["sections"]["dram"] = {"blocked": d.get("blocked"),
                                   "reason": d.get("reason")}
        out["generated_from"].append(rel(p))
        md += ["## Measured DRAM traffic", ""]
        if d.get("blocked"):
            md += [f"**NOT COLLECTED.** {d.get('reason','')}", "",
                   "The paper therefore reports the analytic logical and "
                   "compulsory bounds and states that measured DRAM counters "
                   "are unavailable on this host. No modelled quantity is "
                   "described as a measured bandwidth.", ""]

    os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
    with open(os.path.join(_ROOT, "results", "NUMBERS.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(md))
    with open(os.path.join(_ROOT, "results", "NUMBERS.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"written: results/NUMBERS.md  ({len(md)} lines)")
    print(f"         results/NUMBERS.json")
    print(f"sources: {len(out['generated_from'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
