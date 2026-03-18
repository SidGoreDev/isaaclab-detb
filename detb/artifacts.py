"""Artifact writers for DETB reports and plots."""

from __future__ import annotations

from html import escape
from pathlib import Path

from detb.io import read_csv, read_json
from detb.models import AggregateMetric, RequirementRecord, RunManifest


def _line_plot_svg(points: list[tuple[float, float]], title: str, x_label: str, y_label: str) -> str:
    width = 720
    height = 320
    margin = 40
    chart_width = width - margin * 2
    chart_height = height - margin * 2

    if not points:
        polyline = ""
    else:
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if x_max == x_min:
            x_max += 1.0
        if y_max == y_min:
            y_max += 1.0
        coords = []
        for x_value, y_value in points:
            x_pos = margin + ((x_value - x_min) / (x_max - x_min)) * chart_width
            y_pos = height - margin - ((y_value - y_min) / (y_max - y_min)) * chart_height
            coords.append(f"{x_pos:.1f},{y_pos:.1f}")
        polyline = " ".join(coords)

    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">\n  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n  <text x=\"{width / 2:.0f}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-family=\"Segoe UI\">{escape(title)}</text>\n  <line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#222\"/>\n  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{height - margin}\" stroke=\"#222\"/>\n  <polyline fill=\"none\" stroke=\"#1d4ed8\" stroke-width=\"3\" points=\"{polyline}\"/>\n  <text x=\"{width / 2:.0f}\" y=\"{height - 8}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"Segoe UI\">{escape(x_label)}</text>\n  <text x=\"14\" y=\"{height / 2:.0f}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"Segoe UI\" transform=\"rotate(-90 14 {height / 2:.0f})\">{escape(y_label)}</text>\n</svg>\n"""


def _bar_plot_svg(labels: list[str], values: list[float], title: str, y_label: str) -> str:
    width = 720
    height = 320
    margin = 40
    chart_height = height - margin * 2
    bar_width = (width - margin * 2) / max(len(labels), 1)
    max_value = max(values) if values else 1.0
    if max_value == 0:
        max_value = 1.0

    bars = []
    for index, label in enumerate(labels):
        value = values[index]
        scaled = (value / max_value) * chart_height
        x_pos = margin + index * bar_width + 12
        y_pos = height - margin - scaled
        bars.append(
            f'<rect x="{x_pos:.1f}" y="{y_pos:.1f}" width="{bar_width - 24:.1f}" height="{scaled:.1f}" fill="#059669"/>'
            f'<text x="{x_pos + (bar_width - 24) / 2:.1f}" y="{height - 18}" text-anchor="middle" font-size="11" font-family="Segoe UI">{escape(label)}</text>'
        )

    return f"""<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\">\n  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>\n  <text x=\"{width / 2:.0f}\" y=\"24\" text-anchor=\"middle\" font-size=\"18\" font-family=\"Segoe UI\">{escape(title)}</text>\n  <line x1=\"{margin}\" y1=\"{height - margin}\" x2=\"{width - margin}\" y2=\"{height - margin}\" stroke=\"#222\"/>\n  <line x1=\"{margin}\" y1=\"{margin}\" x2=\"{margin}\" y2=\"{height - margin}\" stroke=\"#222\"/>\n  {"".join(bars)}\n  <text x=\"14\" y=\"{height / 2:.0f}\" text-anchor=\"middle\" font-size=\"13\" font-family=\"Segoe UI\" transform=\"rotate(-90 14 {height / 2:.0f})\">{escape(y_label)}</text>\n</svg>\n"""


def write_training_curve(run_dir: Path, reward_curve: list[tuple[int, float]]) -> Path:
    """Write a simple SVG plot for reward progression."""
    path = run_dir / "training_curve.svg"
    path.write_text(_line_plot_svg(reward_curve, "Training Reward Curve", "Step", "Reward"), encoding="utf-8")
    return path


def write_success_plot(run_dir: Path, aggregates: list[AggregateMetric]) -> Path:
    """Write a compact aggregate overview plot."""
    metrics = {item.metric_name: item.mean for item in aggregates}
    labels = ["success", "distance", "energy"]
    values = [
        metrics.get("task_success_rate", 0.0),
        metrics.get("distance_m", 0.0),
        metrics.get("energy_proxy", 0.0),
    ]
    path = run_dir / "aggregate_overview.svg"
    path.write_text(_bar_plot_svg(labels, values, "Aggregate Overview", "Value"), encoding="utf-8")
    return path


def write_markdown_summary(
    run_dir: Path,
    manifest: RunManifest,
    aggregates: list[AggregateMetric],
    note: str,
) -> Path:
    """Write a reviewable run summary."""
    lines = [
        f"# DETB Run Summary: {manifest.run_id}",
        "",
        f"- Command: `{manifest.command}`",
        f"- Backend: `{manifest.backend}`",
        f"- Robot: `{manifest.robot_variant}`",
        f"- Terrain: `{manifest.terrain_name}`",
        f"- Sensor: `{manifest.sensor_profile}`",
        f"- Fault: `{manifest.fault_profile}`",
        f"- Seeds: `{manifest.seeds}`",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Mean | CI Low | CI High | N | Seed Count |",
        "|--------|------|--------|---------|---|------------|",
    ]
    for item in aggregates:
        lines.append(
            f"| {item.metric_name} | {item.mean:.3f} | {item.ci_low:.3f} | {item.ci_high:.3f} | {item.n} | {item.seed_count} |"
        )
    lines.extend(["", "## Notes", "", note, ""])

    path = run_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_requirements_markdown(run_dir: Path, requirements: list[RequirementRecord]) -> Path:
    """Write the candidate requirements review packet."""
    lines = [
        "# DETB Candidate Requirements",
        "",
        "| ID | Status | Statement | Evidence |",
        "|----|--------|-----------|----------|",
    ]
    for item in requirements:
        lines.append(
            f"| {item.req_id} | {item.status} | {item.statement} | {item.source_metric} {item.confidence_interval} |"
        )
    path = run_dir / "candidate_requirements.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_playback_summary(run_dir: Path, manifest: RunManifest, playback: dict, note: str) -> Path:
    """Write a reviewable playback diagnostics summary."""
    diagnostics = playback.get("diagnostics", {})
    initial_position = diagnostics.get("initial_position_m", ["?", "?", "?"])
    final_position = diagnostics.get("final_position_m", ["?", "?", "?"])
    video_files = playback.get("video_files", [])
    lines = [
        f"# DETB Playback Summary: {manifest.run_id}",
        "",
        f"- Command: `{manifest.command}`",
        f"- Backend: `{manifest.backend}`",
        f"- Task: `{playback.get('task_registry_id', manifest.task_registry_id)}`",
        f"- Robot: `{manifest.robot_variant}`",
        f"- Verdict: `{diagnostics.get('verdict', 'unknown')}`",
        f"- Motion expected from command: `{diagnostics.get('command_motion_expected', False)}`",
        f"- Net displacement (m): `{diagnostics.get('net_displacement_m', 0.0)}`",
        f"- Path length (m): `{diagnostics.get('path_length_m', 0.0)}`",
        f"- Mean planar speed (m/s): `{diagnostics.get('mean_planar_speed_mps', 0.0)}`",
        f"- Mean commanded planar speed (m/s): `{diagnostics.get('mean_command_planar_speed_mps', 0.0)}`",
        f"- Initial position (m): `{initial_position}`",
        f"- Final position (m): `{final_position}`",
        f"- Minimum height (m): `{diagnostics.get('min_height_m', 0.0)}`",
        f"- Steps completed: `{diagnostics.get('steps_completed', 0)}`",
        f"- Video files: `{len(video_files)}`",
        "",
        "## Notes",
        "",
        note,
        "",
    ]
    path = run_dir / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def rebuild_summary(run_dir: Path) -> Path:
    """Rebuild summary markdown from stored aggregate metrics."""
    manifest = RunManifest(**read_json(run_dir / "run_manifest.json"))
    aggregate_rows = read_csv(run_dir / "aggregate_metrics.csv")
    aggregates = [
        AggregateMetric(
            metric_name=row["metric_name"],
            aggregation_scope=row["aggregation_scope"],
            mean=float(row["mean"]),
            median=float(row["median"]),
            stddev=float(row["stddev"]),
            ci_low=float(row["ci_low"]),
            ci_high=float(row["ci_high"]),
            n=int(row["n"]),
            seed_count=int(row["seed_count"]),
        )
        for row in aggregate_rows
    ]
    return write_markdown_summary(run_dir, manifest, aggregates, "Rebuilt from stored aggregate metrics.")
