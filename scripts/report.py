"""Turn the audit JSON into the tables the paper reports.

Reads outputs/audit_<regressor>.json and prints ERROR for RQ1, GAP and LEAK
for RQ2, and the ZOOM results for RQ3, each next to what Section III says
should accompany it: the reference predictors beside the errors, the cell
sizes and rates beside the gaps, and the bootstrap intervals beside both.

    python scripts/report.py outputs/audit_gb.json
"""

import argparse
import json


def interval(record):
    if not record:
        return ""
    low, high = record.get("low"), record.get("high")
    if low is None or high is None:
        return ""
    return f"  [{low:+.3f}, {high:+.3f}]"


def show_error(results):
    print("=" * 72)
    print("ERROR (RQ1): mean absolute error, student-level")
    print("=" * 72)

    references = results.get("references", {})
    print("reference predictors")
    for name, label in (("B1", "training mean"),
                        ("B2", "per-activity mean"),
                        ("B3", "per-session mean (ceiling)")):
        record = references.get(name)
        if isinstance(record, dict):
            print(f"  {name} {label:<28} {record['mean_error']:.4f}")
    if isinstance(references.get("B4"), float):
        print(f"  B4 one expert vs other three  {references['B4']:.4f}")

    print("\nvariants")
    for variant in ("RAW", "TARE-S", "TARE-FULL"):
        record = results["variants"].get(variant)
        if not record:
            continue
        error = record["error"]
        line = (f"  {variant:<11} mean {error['mean_error']:.4f}   "
                f"max {error['e_max']:.4f}   spread {error['e_spread']:.4f}")
        if "mean_delta" in error:
            line += f"   gain over RAW {error['mean_delta']:+.4f}"
        print(line)
        by_activity = record.get("error_by_activity", {})
        if by_activity:
            print("              per activity  " + "  ".join(
                f"{a}={v:.4f}" for a, v in sorted(by_activity.items())
            ))

    inference = results.get("inference", {})
    if inference:
        print("\ninference on the mean gain")
        for variant in ("TARE-S", "TARE-FULL"):
            boot = inference["bootstrap"].get(f"{variant}:mean_delta", {})
            sign = inference["sign_flip"].get(variant, {})
            equiv = inference["equivalence"].get(variant, {})
            if not boot:
                continue
            print(f"  {variant:<11} {boot.get('point', float('nan')):+.4f}"
                  f"{interval(boot)}   "
                  f"sign-flip p={sign.get('p_value', float('nan')):.5f}"
                  f"   unchanged={equiv.get('unchanged')}")


def show_gap(results):
    print()
    print("=" * 72)
    print("GAP (RQ2): equalized odds across apparent-scale groups")
    print("=" * 72)

    for proxy in ("P_iod", "P_sh"):
        if proxy not in results.get("groups", {}):
            continue
        groups = results["groups"][proxy]
        sizes = {}
        for student, group in groups.items():
            sizes[group] = sizes.get(group, 0) + 1
        print(f"\nproxy {proxy}: tertiles hold "
              + ", ".join(f"g{g}={n}" for g, n in sorted(sizes.items()))
              + " students")

        for variant in ("RAW", "TARE-S", "TARE-FULL"):
            record = results["variants"][variant].get(proxy, {})
            if not record:
                continue
            reg = record["eo_reg"]
            binary = record["eo_bin"]["2.5"]
            line = (f"  {variant:<11} EO_reg {reg['gap']:.4f}"
                    f"{interval(reg.get('interval'))}"
                    f"   EO_bin(2.5) {binary['gap']:.4f}"
                    f"{interval(binary.get('interval'))}")
            print(line)
            print(f"              rho {record['rho']:+.3f}"
                  f"{interval(record.get('rho_interval'))}", end="")
            if "delta_eo_reg" in record:
                print(f"   dEO_reg {record['delta_eo_reg']:+.4f}"
                      f"   dEO_bin {record['delta_eo_bin']:+.4f}"
                      f"   drho {record['delta_rho']:+.3f}")
            else:
                print()

            excluded = reg.get("excluded_cells", [])
            if excluded:
                print(f"              cells excluded for holding under three "
                      f"students: {excluded}")

        raw = results["variants"]["RAW"].get(proxy, {})
        if raw:
            print("  cell sizes (label class, group) -> frames/students")
            for cell, size in sorted(raw["eo_reg"]["cell_sizes"].items()):
                print(f"              {cell}  {size['frames']:>5} frames  "
                      f"{size['students']} students")

        print("  sensitivity thresholds")
        for variant in ("RAW", "TARE-S", "TARE-FULL"):
            record = results["variants"][variant].get(proxy, {})
            if not record:
                continue
            print(f"    {variant:<11} " + "  ".join(
                f"tau={t}: {record['eo_bin'][t]['gap']:.4f}"
                for t in sorted(record["eo_bin"])
            ))


def show_leak(results):
    print()
    print("=" * 72)
    print("LEAK (RQ2): can a model recover apparent scale from the features")
    print("=" * 72)
    print("Descriptive by III-C6e, not used for any test.")
    for proxy in ("P_iod", "P_sh"):
        print(f"\nproxy {proxy}")
        for variant in ("RAW", "TARE-S", "TARE-FULL"):
            record = results["variants"][variant].get(proxy, {})
            leak = record.get("leak")
            if not leak:
                continue
            intervals = leak.get("intervals", {})
            print(f"  {variant:<11} R2 {leak['r2']:+.3f}"
                  f"{interval(intervals.get('r2'))}"
                  f"   Xi {leak['xi']:.3f}{interval(intervals.get('xi'))}")


def show_zoom(results):
    keypoint = results.get("zoom_keypoint")
    if keypoint:
        print()
        print("=" * 72)
        print("ZOOM, keypoint level: the unit test of Remark 3")
        print("=" * 72)
        print("Proposition 1 makes the TARE features identical before and "
              "after (17), so\nthe drift is zero in exact arithmetic. What "
              "survives in float is a feature\ndifference near machine "
              "epsilon, which a boosted tree can turn into a discrete\njump "
              "when it lands on the wrong side of a split. A residual of "
              "about 1e-4 is\nthat effect; anything near the RAW figure "
              "would mean the invariance is broken.")
        for variant in ("RAW", "TARE-S", "TARE-FULL"):
            table = keypoint.get(variant, {})
            if not table:
                continue
            worst = max(v["drift"] for v in table.values())
            note = ""
            if variant != "RAW":
                note = "  float noise" if worst < 1e-3 else "  INVARIANCE BROKEN"
            print(f"  {variant:<11} largest drift over the grid "
                  f"{worst:.6f}{note}")

    image = results.get("zoom_image")
    if image:
        print()
        print("=" * 72)
        print("ZOOM, image level (RQ3): drift after re-extracting landmarks")
        print("=" * 72)
        for variant in ("RAW", "TARE-S", "TARE-FULL"):
            table = image.get(variant, {})
            for setting in sorted(table):
                value = table[setting]
                print(f"  {variant:<11} {setting:<24} drift {value['drift']:.4f}"
                      f"   flip {value['flip_rate']:.4f}")

    dose = results.get("dose_response")
    if dose:
        print()
        print("dose-response, Gamma(c) = gamma_0 + gamma_1 ln c")
        for variant, fit in dose.get("fit", {}).items():
            print(f"  {variant:<11} slope {fit['slope']:+.4f}"
                  f"{interval(fit.get('slope_interval'))}")

    cohort = results.get("perturbed_cohort")
    if cohort:
        print()
        print("independent per-student zoom, distribution over replicates")
        for variant, record in cohort.items():
            spread = record["e_spread"]
            print(f"  {variant:<11} e_spread mean {spread['mean']:.4f}  "
                  f"[{spread['q05']:.4f}, {spread['q95']:.4f}]  "
                  f"n={spread['n_replicates']}")
            for proxy in ("P_iod", "P_sh"):
                if proxy in record:
                    rho = record[proxy]["rho"]
                    print(f"              {proxy} rho mean {rho['mean']:+.3f}  "
                          f"[{rho['q05']:+.3f}, {rho['q95']:+.3f}]")


def show_holm(results):
    inference = results.get("inference", {})
    holm = inference.get("holm")
    if not holm:
        return
    print()
    print("=" * 72)
    print("Holm correction inside the primary family of ten comparisons")
    print("=" * 72)
    for name, record in sorted(holm.items(), key=lambda kv: kv[1]["rank"]):
        mark = "reject" if record["rejected"] else "      "
        print(f"  {record['rank']:>2}. {name:<34} p={record['p_value']:.5f}"
              f"  <= {record['threshold']:.5f}  {mark}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="outputs/audit_gb.json")
    args = parser.parse_args()

    try:
        with open(args.path) as handle:
            results = json.load(handle)
    except FileNotFoundError:
        raise SystemExit(f"{args.path} not found. Run run_audit.py first.")

    run = results.get("run", {})
    print(f"regressor {run.get('regressor')}   "
          f"{run.get('n_students')} students   "
          f"{run.get('n_frames')} frames   "
          f"complete={run.get('complete')}")

    diagnostics = results.get("diagnostics", {})
    detection = diagnostics.get("detection_rate_by_activity", {})
    if detection:
        print("detection rate by activity (III-B1 reports 98.7% lecture, "
              "78.4% robotics)")
        for activity, rate in sorted(detection.items()):
            print(f"  activity {activity}: {100 * rate:.1f}%")
    shares = diagnostics.get("scale_branch_shares", {})
    if shares:
        print(f"branch of (4) that set L_i: shoulder "
              f"{100 * shares.get('shoulder', 0):.1f}%, "
              f"eye {100 * shares.get('eye', 0):.1f}%")
    missing = diagnostics.get("students_without_baseline", [])
    if missing:
        print(f"students with no H2 baseline, falling back to TARE-S: {missing}")

    show_error(results)
    show_gap(results)
    show_leak(results)
    show_zoom(results)
    show_holm(results)


if __name__ == "__main__":
    main()
