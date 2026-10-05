"""Standalone HTML reports for TokenMon snapshot commands."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from tokenmon.analyzer import (TimelineLane, analyze_agent_model_windows,
                               filter_by_window, summarize_spans, WINDOW_DURATIONS)
from tokenmon.models import GenerationSpan, SessionTimeline, TimelineEvent, format_session_duration


COLORS = {
    "stream": "#3b82f6",
    "assistant": "#60a5fa",
    "reasoning": "#a855f7",
    "tool": "#22c55e",
    "user": "#06b6d4",
    "turn": "#94a3b8",
}


def _local_time(timestamp: float, seconds: bool = True) -> str:
    fmt = "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).astimezone().strftime(fmt)


def _number(value: float | None, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.1f}{suffix}"


def _page(title: str, subtitle: str, body: str) -> str:
    safe_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>
:root {{ color-scheme: light dark; --bg:#f8fafc; --panel:#fff; --text:#172033; --muted:#64748b; --line:#dbe3ef; --accent:#2563eb; }}
@media (prefers-color-scheme:dark) {{ :root {{ --bg:#0b1220; --panel:#111b2e; --text:#e5edf8; --muted:#94a3b8; --line:#26334a; --accent:#60a5fa; }} }}
* {{ box-sizing:border-box }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }}
main {{ max-width:1440px; margin:auto; padding:32px }} h1 {{ margin:0; font-size:28px }} h2 {{ margin:28px 0 12px; font-size:18px }} .subtitle,.muted {{ color:var(--muted) }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px }} .card,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; box-shadow:0 5px 18px #0000000d }}
.card {{ padding:16px }} .card b {{ display:block; font-size:24px; margin-top:4px }} .panel {{ padding:16px; overflow:auto }}
table {{ width:100%; border-collapse:collapse; white-space:nowrap }} th,td {{ padding:9px 12px; border-bottom:1px solid var(--line); text-align:left }} th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; cursor:pointer }} tr:last-child td {{ border-bottom:0 }}
.pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:color-mix(in srgb,var(--accent) 14%,transparent); color:var(--accent) }}
.event {{ border-left:4px solid var(--event-color,var(--line)); padding:10px 12px; margin:8px 0; background:color-mix(in srgb,var(--event-color,var(--line)) 8%,var(--panel)); border-radius:0 8px 8px 0 }}
.event-head {{ display:flex; gap:12px; flex-wrap:wrap; color:var(--muted); font-size:12px }} .event p {{ margin:5px 0 0 }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; margin:14px 0 }} .swatch {{ width:10px; height:10px; border-radius:2px; display:inline-block; margin-right:5px }}
.chart {{ min-width:900px }} svg {{ width:100%; height:auto; display:block }} .axis {{ fill:var(--muted); font-size:11px }} .label {{ fill:var(--text); font-size:12px }} .track-label {{ fill:var(--muted); font-size:10px }} .gridline {{ stroke:var(--line); stroke-width:1 }}
@media print {{ body {{ background:#fff }} main {{ max-width:none; padding:12px }} .card,.panel {{ box-shadow:none }} }}
@media (max-width:700px) {{ main {{ padding:18px }} h1 {{ font-size:23px }} }}
</style>
</head>
<body><main>
<h1>{safe_title}</h1><div class="subtitle">{escape(subtitle)}</div>
{body}
</main>
<script>
document.querySelectorAll('th[data-sort]').forEach(th=>th.addEventListener('click',()=>{{const table=th.closest('table'),body=table.tBodies[0],i=[...th.parentNode.children].indexOf(th),rows=[...body.rows],asc=th.dataset.dir!=='asc';rows.sort((a,b)=>{{let x=a.cells[i].dataset.value??a.cells[i].textContent,y=b.cells[i].dataset.value??b.cells[i].textContent;return (Number(x)-Number(y)||x.localeCompare(y))*(asc?1:-1)}});rows.forEach(r=>body.appendChild(r));th.dataset.dir=asc?'asc':'desc'}}));
</script></body></html>
"""


def _table(headers: list[str], rows: list[list[tuple[str, str | float | int | None]]]) -> str:
    head = "".join(f'<th data-sort="1">{escape(header)}</th>' for header in headers)
    body_rows = []
    for row in rows:
        cells = []
        for display, raw in row:
            value = "" if raw is None else str(raw)
            cells.append(f'<td data-value="{escape(value, quote=True)}">{escape(display)}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="panel"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody></table></div>'


def render_stats_report(
    adapter_names: list[str], spans: list[GenerationSpan], timelines: list[SessionTimeline],
    windows: list[str], now: float, tasks: int,
) -> str:
    cards = []
    for window in windows:
        windowed = filter_by_window(spans, WINDOW_DURATIONS[window], now)
        summary = summarize_spans(windowed, window, "all")
        cards.append(
            f'<div class="card"><span class="muted">{escape(window)}</span>'
            f'<b>{_number(summary.weighted_tps, " TPS")}</b>'
            f'<span>{summary.valid_spans}/{summary.total_spans} outputs · {summary.total_tokens:,} tokens</span></div>'
        )
    sections = ['<h2>Overview</h2><div class="grid">' + "".join(cards) + "</div>"]
    for (agent, model), summaries in analyze_agent_model_windows(spans, windows, now).items():
        rows = []
        for summary in summaries:
            rows.append([
                (summary.window_name, summary.window_name),
                (f"{summary.valid_spans}/{summary.total_spans}", summary.valid_spans),
                (f"{summary.total_tokens:,}", summary.total_tokens),
                (format_session_duration(summary.total_duration), summary.total_duration),
                (_number(summary.weighted_tps), summary.weighted_tps),
                (_number(summary.median_tps), summary.median_tps),
            ])
        sections.append(f'<h2>{escape(model)} <span class="pill">{escape(agent)}</span></h2>' +
                        _table(["Window", "Outputs", "Tokens", "Time", "TPS weighted", "Median"], rows))
    recent = [span for span in spans if span.tps is not None][-10:]
    if recent:
        rows = [[(_local_time(s.ended_at), s.ended_at), (s.agent, s.agent), (s.model, s.model),
                 (f"{s.tokens:,}", s.tokens), (format_session_duration(s.duration), s.duration),
                 (_number(s.tps), s.tps)] for s in reversed(recent)]
        sections.append('<h2>Recent model output</h2>' + _table(
            ["Completed", "Agent", "Model", "Tokens", "Duration", "TPS"], rows))
    subtitle = f"Agents: {', '.join(adapter_names)} · {len(spans)} streams · up to {tasks} sessions · generated {_local_time(now)}"
    return _page("TokenMon statistics", subtitle, "".join(sections))


def render_sessions_report(timelines: list[SessionTimeline], now: float) -> str:
    rows = [[
        (t.session_id, t.session_id), (t.agent, t.agent), (t.model, t.model),
        (str(t.user_messages), t.user_messages), (str(t.assistant_messages), t.assistant_messages),
        (str(t.tool_calls), t.tool_calls), (f"{t.total_tokens:,}", t.total_tokens),
        (format_session_duration(t.session_duration), t.session_duration), (t.status(now), t.idle_time(now)),
    ] for t in timelines]
    body = '<h2>Sessions</h2>' + _table(
        ["Session", "Agent", "Model", "Prompts", "Responses", "Tools", "Tokens", "Duration", "Status"], rows)
    return _page("TokenMon sessions", f"{len(timelines)} sessions · generated {_local_time(now)}", body)


def _event_color(kind: str) -> str:
    mapping = {"user_message": "user", "assistant_message": "assistant", "reasoning": "reasoning",
               "tool_call": "tool", "tool_output": "tool", "turn_start": "turn", "turn_end": "turn"}
    return COLORS.get(mapping.get(kind, "turn"), COLORS["turn"])


def render_logs_report(timelines: list[SessionTimeline], now: float) -> str:
    sections = []
    for timeline in timelines:
        events = []
        for event in timeline.events:
            meta = [event.kind.replace("_", " "), _local_time(event.timestamp)]
            if event.tokens is not None:
                meta.append(f"{event.tokens:,} tokens")
            if event.duration is not None:
                meta.append(format_session_duration(event.duration))
            events.append(
                f'<div class="event" style="--event-color:{_event_color(event.kind)}">'
                f'<div class="event-head">{"<span>" + "</span><span>".join(escape(value) for value in meta) + "</span>"}</div>'
                f'<p>{escape(event.summary)}</p></div>'
            )
        sections.append(
            f'<h2>{escape(timeline.session_id)} <span class="pill">{escape(timeline.agent)}</span></h2>'
            f'<div class="muted">{escape(timeline.model)} · {timeline.total_tokens:,} tokens · {escape(timeline.status(now))}</div>'
            + "".join(events)
        )
    return _page("TokenMon event logs", f"{len(timelines)} session(s) · generated {_local_time(now)}", "".join(sections))


def render_timeline_report(lanes: list[TimelineLane], now: float) -> str:
    if not lanes:
        return _page("TokenMon timeline", f"Generated {_local_time(now)}", "<p>No session activity found.</p>")
    chart_start = min(lane.started_at for lane in lanes)
    chart_end = max(lane.ended_at for lane in lanes)
    duration = max(1.0, chart_end - chart_start)
    width, left, plot_width = 1200, 185, 850
    tracks = [("Model", {"stream", "assistant"}), ("Reasoning", {"reasoning"}),
              ("Tools", {"tool"}), ("User", {"user"}), ("Turns", {"turn"})]
    lane_height = 122
    height = 58 + lane_height * len(lanes) + 35
    svg: list[str] = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Session activity timeline">']
    for index in range(5):
        x = left + plot_width * index / 4
        at = chart_start + duration * index / 4
        svg.append(f'<line class="gridline" x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{height - 24}"/>')
        svg.append(f'<text class="axis" x="{x:.1f}" y="18" text-anchor="middle">{escape(_local_time(at)[11:])}</text>')
    for lane_index, lane in enumerate(lanes):
        base = 45 + lane_index * lane_height
        svg.append(f'<rect x="0" y="{base - 12}" width="{width}" height="{lane_height - 4}" rx="8" fill="var(--panel)"/>')
        svg.append(f'<text class="label" x="8" y="{base + 4}">{escape(lane.session.session_id[:20])}</text>')
        svg.append(f'<text class="track-label" x="8" y="{base + 18}">{escape(lane.session.agent)} · {escape(lane.session.model[:22])}</text>')
        for track_index, (track_name, kinds) in enumerate(tracks):
            y = base + track_index * 17
            svg.append(f'<text class="track-label" x="{left - 8}" y="{y + 5}" text-anchor="end">{track_name}</text>')
            svg.append(f'<line class="gridline" x1="{left}" y1="{y + 3}" x2="{left + plot_width}" y2="{y + 3}" opacity=".45"/>')
            for interval in lane.intervals:
                if interval.kind not in kinds:
                    continue
                x = left + (interval.started_at - chart_start) / duration * plot_width
                interval_width = interval.duration / duration * plot_width
                color = COLORS[interval.kind]
                title = escape(interval.summary or (
                    f"{interval.kind}: {interval.tokens or 0} tokens at {_number(interval.tps, ' TPS')}"), quote=False)
                if interval_width >= 1.5:
                    svg.append(f'<rect x="{x:.2f}" y="{y - 3}" width="{max(2.0, interval_width):.2f}" height="12" rx="2" fill="{color}"><title>{title}</title></rect>')
                else:
                    svg.append(f'<circle cx="{x:.2f}" cy="{y + 3}" r="3" fill="{color}"><title>{title}</title></circle>')
        details = f"{format_session_duration(lane.duration)} · model {format_session_duration(lane.streaming_duration)}"
        if lane.weighted_tps is not None:
            details += f" · {lane.weighted_tps:.1f} TPS"
        svg.append(f'<text class="track-label" x="{left + plot_width + 12}" y="{base + 5}">{escape(details)}</text>')
    svg.append("</svg>")
    legend = '<div class="legend">' + "".join(
        f'<span><i class="swatch" style="background:{color}"></i>{escape(name.title())}</span>'
        for name, color in COLORS.items()) + "</div>"
    body = legend + '<div class="panel chart">' + "".join(svg) + "</div>"
    subtitle = f"{len(lanes)} session(s), longest first · {_local_time(chart_start)} — {_local_time(chart_end)} · generated {_local_time(now)}"
    return _page("TokenMon activity timeline", subtitle, body)
