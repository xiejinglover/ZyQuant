from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Mapping, Protocol

import pandas as pd


class ReportPlugin(Protocol):
    def write(
        self,
        path: str | Path,
        metrics: Mapping[str, Any],
        frames: Mapping[str, pd.DataFrame],
        manifest: Mapping[str, Any],
    ) -> Path: ...


def write_html_report(
    path: str | Path,
    metrics: Mapping[str, Any],
    nav: pd.DataFrame,
    attribution: pd.DataFrame,
) -> Path:
    destination = Path(path)
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(name))}</th><td>{html.escape(str(value))}</td></tr>"
        for name, value in metrics.items()
    )
    nav_html = nav.tail(100).to_html(index=False, border=0) if not nav.empty else "<p>No NAV rows.</p>"
    attribution_html = (
        attribution.tail(200).to_html(index=False, border=0)
        if not attribution.empty else "<p>No attribution rows.</p>"
    )
    destination.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>ZyQuant Report</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:1200px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.4rem;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}</style>
</head><body><h1>ZyQuant 回测报告</h1><h2>指标</h2><table>{metric_rows}</table>
<h2>NAV（最近 100 行）</h2>{nav_html}<h2>归因（最近 200 行）</h2>{attribution_html}
</body></html>""", encoding="utf-8")
    return destination
