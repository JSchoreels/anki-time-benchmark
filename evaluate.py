import argparse
import json
import math
import pathlib
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import scipy  # type: ignore


DEFAULT_METHODS = [
    "CONST",
    "USER_MEDIAN",
    "GRADE_MEDIAN_4",
    "GRADE_MEDIAN_8",
    "FSRS7_R_LINEAR",
    "FSRS7_R_GRADE_INTERACT",
    "FSRS7_DSR_GRADE_NN",
]


def sigdig(value: float, CI: float) -> Tuple[str, str]:
    def num_lead_zeros(x: float) -> float:
        return math.inf if x == 0 else -math.floor(math.log10(abs(x))) - 1

    n_lead_zeros_CI = num_lead_zeros(CI)
    CI_sigdigs = 2
    decimals = int(n_lead_zeros_CI + CI_sigdigs)
    rounded_CI = round(CI, decimals)
    rounded_value = round(value, decimals)
    if n_lead_zeros_CI > num_lead_zeros(rounded_CI):
        d = max(decimals - 1, 0)
        return str(f"{round(value, d):.{d}f}"), str(f"{round(CI, d):.{d}f}")
    return str(f"{rounded_value:.{max(decimals, 0)}f}"), str(
        f"{rounded_CI:.{max(decimals, 0)}f}"
    )


def confidence_interval(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0

    identifiers = np.arange(len(values))
    dict_x_w = {
        int(i): (float(v), float(w))
        for i, (v, w) in enumerate(zip(values, weights))
    }

    def weighted_mean_bootstrap(z, axis):
        data = np.vectorize(dict_x_w.get)(z)
        return np.average(data[0], weights=data[1], axis=axis)

    try:
        ci_99 = scipy.stats.bootstrap(
            (identifiers,),
            statistic=weighted_mean_bootstrap,
            confidence_level=0.99,
            axis=0,
            method="BCa",
            random_state=42,
        )
        low = float(ci_99.confidence_interval.low)
        high = float(ci_99.confidence_interval.high)
        return max((high - low) / 2, 0.0)
    except Exception:
        return 0.0


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def discover_methods(result_dir: pathlib.Path, suffix: Optional[str]) -> List[str]:
    stems = [f.stem for f in sorted(result_dir.glob("*.jsonl"))]
    if suffix:
        tag = f"_{suffix}"
        methods = [s[:-len(tag)] for s in stems if s.endswith(tag)]
        return sorted(set(methods))
    return stems


def resolve_result_file(method: str, result_dir: pathlib.Path, suffix: Optional[str]) -> Optional[pathlib.Path]:
    candidates: List[pathlib.Path] = []
    if suffix:
        candidates.append(result_dir / f"{method}_{suffix}.jsonl")
    candidates.append(result_dir / f"{method}.jsonl")

    for fp in candidates:
        if fp.exists():
            return fp
    return None


def load_method_user_metrics(
    method: str, result_dir: pathlib.Path, suffix: Optional[str]
) -> Dict[str, Dict]:
    """
    Returns per-user metrics for one method:
      {
        user_id: {"MAE": float, "RMSE": float, "MAPE": float, "size": int},
        ...
      }
    """
    fp = resolve_result_file(method, result_dir, suffix)
    if fp is None:
        return {}

    with fp.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    out: Dict[str, Dict] = {}
    for r in rows:
        user = r.get("user")
        m = r.get("metrics", {})
        if user is None:
            continue
        if "MAE" not in m or "RMSE" not in m or "MAPE" not in m:
            continue
        if m["MAE"] is None or m["RMSE"] is None or m["MAPE"] is None:
            continue

        out[str(user)] = {
            "MAE": float(m["MAE"]),
            "RMSE": float(m["RMSE"]),
            "MAPE": float(m["MAPE"]),
            "size": int(r.get("size", 0)),
        }
    return out


def _metric_mean_ci(
    metrics_list: List[Dict], metric: str, weight_by: str
) -> Tuple[Optional[float], Optional[float]]:
    vals = np.array([x[metric] for x in metrics_list], dtype=float)

    if weight_by == "reviews":
        w = np.array([x["size"] for x in metrics_list], dtype=float)
        if np.all(w == 0):
            w = np.ones_like(w)
    else:
        w = np.ones(len(metrics_list), dtype=float)

    mask = ~np.isnan(vals)
    vals = vals[mask]
    w = w[mask]

    if len(vals) == 0:
        return None, None

    mean = weighted_mean(vals, w)
    ci = confidence_interval(vals, w)
    return mean, ci


def fmt_mean_ci(mean: Optional[float], ci: Optional[float], suffix: str = "") -> str:
    if mean is None or ci is None:
        return "N/A"
    m, c = sigdig(mean, ci)
    if suffix == "s":
        return f"{m}±{c} s"
    if suffix == "%":
        return f"{m}%±{c}%"
    raise Exception("Unknown suffix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="./result", help="Directory with *.jsonl result files")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional base method names (e.g., CONST FSRS7_R_LINEAR). "
            "If omitted, uses --default-methods or auto-discovery."
        ),
    )
    parser.add_argument(
        "--default-methods",
        action="store_true",
        help="Use built-in default method list instead of auto-discovery.",
    )
    parser.add_argument(
        "--suffix",
        default="NO_FIRST_REVIEWS",
        help=(
            "Result filename suffix used by benchmark outputs, e.g. "
            "NO_FIRST_REVIEWS or WITH_FIRST_REVIEWS. "
            "Set empty string to disable suffix matching."
        ),
    )
    parser.add_argument(
        "--weight-by",
        choices=["reviews", "users"],
        default="reviews",
        help="Aggregate per-user metrics weighted by number of reviews or equally by users",
    )
    args = parser.parse_args()

    result_dir = pathlib.Path(args.result_dir)
    if not result_dir.exists():
        print(f"Result directory not found: {result_dir}")
        return

    suffix = args.suffix.strip() or None

    if args.methods:
        methods = args.methods
    elif args.default_methods:
        methods = DEFAULT_METHODS
    else:
        methods = discover_methods(result_dir, suffix=suffix)

    if not methods:
        print("No result files found.")
        return

    method_user_data: Dict[str, Dict[str, Dict]] = {}
    for method in methods:
        d = load_method_user_metrics(method, result_dir, suffix=suffix)
        if d:
            method_user_data[method] = d

    if not method_user_data:
        print("No valid method files with MAE/RMSE/MAPE found.")
        return

    user_sets: List[Set[str]] = [set(d.keys()) for d in method_user_data.values()]
    common_users = set.intersection(*user_sets) if user_sets else set()

    if not common_users:
        print("No common users found across selected method files.")
        return

    suffix_label = suffix if suffix else "(none)"
    print(f"Suffix: {suffix_label}")
    print(f"Aggregation weight: {args.weight_by} (99% CI, bootstrap BCa)")
    print(f"Methods compared: {len(method_user_data)}")
    print(f"Common users in all method files: {len(common_users)}")

    ref_method = next(iter(method_user_data.keys()))
    common_reviews = sum(method_user_data[ref_method][u]["size"] for u in common_users)
    print(f"Common reviews (from shared users): {common_reviews}\n")

    rows = []
    for method, user_map in method_user_data.items():
        metrics_list = [user_map[u] for u in common_users]

        mae_mean, mae_ci = _metric_mean_ci(metrics_list, "MAE", args.weight_by)
        rmse_mean, rmse_ci = _metric_mean_ci(metrics_list, "RMSE", args.weight_by)
        mape_mean, mape_ci = _metric_mean_ci(metrics_list, "MAPE", args.weight_by)

        rows.append(
            (
                method,
                float("inf") if rmse_mean is None else rmse_mean,
                fmt_mean_ci(rmse_mean, rmse_ci, "s"),
                fmt_mean_ci(mae_mean, mae_ci, "s"),
                fmt_mean_ci(mape_mean, mape_ci, "%"),
            )
        )

    rows.sort(key=lambda x: x[1])

    print("| Method | RMSE | MAE | MAPE |")
    print("| --- | --- | --- | --- |")
    for method, _, rmse, mae, mape in rows:
        print(f"| {method} | {rmse} | {mae} | {mape} |")


if __name__ == "__main__":
    main()
