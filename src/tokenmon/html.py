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


TIMELINE_SCRIPT = r"""
const chart = document.getElementById('timeline-chart');
if (chart) {
  const overview = document.getElementById('timeline-overview');
  const selection = document.getElementById('timeline-selection');
  const zoom = document.getElementById('timeline-zoom');
  const epoch = Number(chart.dataset.start), duration = Number(chart.dataset.duration);
  const initial = [Number(chart.dataset.focusStart), Number(chart.dataset.focusEnd)];
  const minimum = Math.min(1, 1 / duration);
  let start = 0, end = 1;
  const ns = 'http://www.w3.org/2000/svg';
  const format = value => new Date((epoch + value * duration) * 1000).toLocaleString();
  function range(a, b) {
    const span = Math.max(minimum, Math.min(1, b - a));
    start = Math.max(0, Math.min(1 - span, a)); end = start + span;
    chart.querySelectorAll('[data-event]').forEach(el => {
      const pos = Number(el.dataset.pos), finish = Number(el.dataset.end ?? pos);
      el.style.display = finish < start || pos > end ? 'none' : '';
      el.setAttribute('x', el.hasAttribute('data-end') ? 220 + (pos - start) / span * 960 : Math.min(1178, 220 + (pos - start) / span * 960));
      if (el.hasAttribute('data-end')) el.setAttribute('width', Math.max(3, (finish - pos) / span * 960));
    });
    const axis = document.getElementById('timeline-axis');
    axis.replaceChildren();
    const steps = [1,2,5,10,15,30,60,120,300,600,900,1800,3600,7200,14400,43200,86400];
    const seconds = span * duration;
    const step = steps.find(s => s >= seconds / 6) || Math.ceil(seconds / 6 / 86400) * 86400;
    const begin = epoch + start * duration, finish = epoch + end * duration;
    for (let at = Math.ceil(begin / step) * step; at <= finish; at += step) {
      const x = 220 + (at - begin) / seconds * 960;
      const line = document.createElementNS(ns, 'line');
      for (const [key, value] of Object.entries({x1:x,x2:x,y1:28,y2:Number(chart.dataset.height)-8,class:'gridline'})) line.setAttribute(key, value);
      const label = document.createElementNS(ns, 'text');
      for (const [key, value] of Object.entries({x,y:18,class:'axis','text-anchor':'middle'})) label.setAttribute(key, value);
      label.textContent = new Date(at * 1000).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit', ...(step < 60 ? {second:'2-digit'} : {})});
      axis.append(line, label);
    }
    selection.setAttribute('x', 220 + start * 960);
    selection.setAttribute('width', Math.max(2, span * 960));
    document.getElementById('visible-range').textContent = `${format(start)} — ${format(end)}`;
    document.getElementById('zoom-value').textContent = `${(1 / span).toFixed(1)}×`;
    zoom.value = Math.log2(1 / span);
  }
  zoom.addEventListener('input', () => { const span = 1 / 2 ** Number(zoom.value), center = (start + end) / 2; range(center - span / 2, center + span / 2); });
  document.getElementById('timeline-full').addEventListener('click', () => range(0, 1));
  document.getElementById('timeline-latest').addEventListener('click', () => range(...initial));
  document.getElementById('timeline-earlier').addEventListener('click', () => range(start - (end - start) / 2, end - (end - start) / 2));
  document.getElementById('timeline-later').addEventListener('click', () => range(start + (end - start) / 2, end + (end - start) / 2));
  const position = event => { const point = overview.createSVGPoint(); point.x=event.clientX; point.y=event.clientY; return Math.max(0, Math.min(1, (point.matrixTransform(overview.getScreenCTM().inverse()).x - 220) / 960)); };
  let anchor = null;
  overview.addEventListener('pointerdown', event => { anchor = position(event); overview.setPointerCapture(event.pointerId); });
  overview.addEventListener('pointermove', event => { if (anchor !== null && Math.abs(position(event) - anchor) > minimum) range(Math.min(anchor, position(event)), Math.max(anchor, position(event))); });
  overview.addEventListener('pointerup', event => { if (anchor === null) return; const point=position(event); if(Math.abs(point-anchor)<minimum) { const span=end-start; range(point-span/2,point+span/2); } else range(Math.min(anchor,point),Math.max(anchor,point)); anchor=null; });
  overview.addEventListener('pointercancel', () => { anchor=null; });
  range(...initial);
}
"""


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
.chart {{ overflow:auto }} .chart svg {{ display:block; width:100%; height:auto }} .axis {{ fill:var(--muted); font-size:11px }} .label {{ fill:var(--text); font-size:12px }} .track-label {{ fill:var(--muted); font-size:10px }} .gridline {{ stroke:var(--line); stroke-width:1 }}
.chart-controls {{ display:flex; align-items:center; gap:12px; margin:12px 0 }} .chart-controls input {{ width:180px }}
.chart-controls {{ flex-wrap:wrap }} button {{ padding:6px 10px; border:1px solid var(--line); border-radius:6px; background:var(--panel); color:var(--text); cursor:pointer }}
#timeline-overview {{ width:100%; display:block; touch-action:none; cursor:crosshair }}
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
{TIMELINE_SCRIPT}
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
    width, left, plot_width = 1200, 220, 960
    tracks = [("Model", {"stream", "assistant"}), ("Reasoning", {"reasoning"}),
              ("Tools", {"tool"}), ("User", {"user"}), ("Turns", {"turn"})]
    lane_tracks = [[(name, kinds) for name, kinds in tracks
                    if any(interval.kind in kinds for interval in lane.intervals)]
                   for lane in lanes]
    activity = sorted((interval for lane in lanes for interval in lane.intervals),
                      key=lambda interval: interval.started_at)
    cluster_start = activity[0].started_at if activity else chart_start
    last_end = cluster_start
    for interval in activity:
        if interval.started_at - last_end > 120:
            cluster_start = interval.started_at
        last_end = max(last_end, interval.ended_at)
    focus_end = min(chart_end, last_end + 15) if activity else chart_end
    focus_start = max(chart_start, min(cluster_start - 15, focus_end - 60), focus_end - 900)
    focus = ((focus_start - chart_start) / duration, (focus_end - chart_start) / duration)
    height = 34 + sum(40 + max(1, len(active)) * 20 for active in lane_tracks) + 8
    svg: list[str] = [f'<svg id="timeline-chart" data-height="{height}" data-start="{chart_start}" data-duration="{duration}" data-focus-start="{focus[0]}" data-focus-end="{focus[1]}" viewBox="0 0 {width} {height}" role="img" aria-label="Session activity timeline">',
                      f'<defs><clipPath id="timeline-plot"><rect x="{left}" y="28" width="{plot_width}" height="{height - 28}"/></clipPath></defs>',
                      '<g id="timeline-axis">']
    for index in range(5):
        x = left + plot_width * index / 4
        at = chart_start + duration * index / 4
        svg.append(f'<line class="gridline" data-pos="{index / 4}" data-attr="x1" data-x2="same" x1="{x:.1f}" y1="28" x2="{x:.1f}" y2="{height - 8}"/>')
        svg.append(f'<text class="axis" data-pos="{index / 4}" x="{x:.1f}" y="18" text-anchor="middle">{escape(_local_time(at)[11:])}</text>')
    svg.append('</g>')
    base = 34
    for lane_index, lane in enumerate(lanes):
        svg.append(f'<text class="label" x="8" y="{base + 12}">{escape(lane.session.session_id[:26])}<title>{escape(lane.session.session_id)}</title></text>')
        svg.append(f'<text class="track-label" x="8" y="{base + 26}">{escape(lane.session.agent)} · {escape(lane.session.model[:25])}</text>')
        for track_index, (track_name, kinds) in enumerate(lane_tracks[lane_index]):
            y = base + 38 + track_index * 20
            svg.append(f'<text class="track-label" x="{left - 8}" y="{y + 5}" text-anchor="end">{track_name}</text>')
            svg.append(f'<line class="gridline" data-pos="0" data-attr="x1" data-x2="end" x1="{left}" y1="{y + 3}" x2="{left + plot_width}" y2="{y + 3}" opacity=".45"/>')
            for interval in lane.intervals:
                if interval.kind not in kinds:
                    continue
                x = left + (interval.started_at - chart_start) / duration * plot_width
                interval_width = interval.duration / duration * plot_width
                color = COLORS[interval.kind]
                title = escape((interval.summary or interval.kind) +
                               f" · {_local_time(interval.started_at)} · {interval.duration:.3f}s" +
                               (f" · {interval.tokens:,} tokens · {_number(interval.tps, ' TPS')}"
                                if interval.tokens is not None else ""), quote=False)
                pos = (interval.started_at - chart_start) / duration
                end = (interval.ended_at - chart_start) / duration
                if interval.duration > 0:
                    svg.append(f'<rect clip-path="url(#timeline-plot)" data-event="1" data-pos="{pos}" data-end="{end}" x="{x:.2f}" y="{y - 3}" width="{max(3.0, interval_width):.2f}" height="12" rx="2" fill="{color}"><title>{title}</title></rect>')
                else:
                    svg.append(f'<rect clip-path="url(#timeline-plot)" data-event="1" data-pos="{pos}" x="{x:.2f}" y="{y - 2}" width="2" height="10" fill="{color}"><title>{title} · recorded timestamp only</title></rect>')
        details = f"{format_session_duration(lane.duration)} · model {format_session_duration(lane.streaming_duration)}"
        if lane.weighted_tps is not None:
            details += f" · {lane.weighted_tps:.1f} TPS"
        svg.append(f'<text class="track-label" x="{left}" y="{base + 12}">{escape(details)}</text>')
        if not lane_tracks[lane_index]:
            svg.append(f'<text class="track-label" x="{left}" y="{base + 43}">No recorded activity</text>')
        base += 40 + max(1, len(lane_tracks[lane_index])) * 20
    svg.append("</svg>")
    legend = '<div class="legend">' + "".join(
        f'<span><i class="swatch" style="background:{color}"></i>{escape(name.title())}</span>'
        for name, color in COLORS.items()) + "</div>"
    controls = '<div class="chart-controls"><button id="timeline-full">Full session</button><button id="timeline-latest">Latest activity</button><button id="timeline-earlier" aria-label="Earlier time range">← Earlier</button><button id="timeline-later" aria-label="Later time range">Later →</button><label for="timeline-zoom">Time zoom</label><input id="timeline-zoom" type="range" min="0" max="12" step="0.1" value="0"><output id="zoom-value">1×</output></div>'
    overview = ['<div class="panel"><div class="muted">Full history · drag a range to inspect, or click to move the detail view</div>',
                '<svg id="timeline-overview" viewBox="0 0 1200 58" role="img" aria-label="Select a time range from the full history">',
                f'<text class="axis" x="220" y="15">{escape(_local_time(chart_start))}</text>',
                f'<text class="axis" x="1180" y="15" text-anchor="end">{escape(_local_time(chart_end))}</text>',
                '<rect x="220" y="22" width="960" height="24" fill="var(--line)"/>']
    for interval in activity:
        x = left + (interval.started_at - chart_start) / duration * plot_width
        bar_width = max(2, interval.duration / duration * plot_width)
        overview.append(f'<rect x="{min(x, 1178):.2f}" y="28" width="{bar_width:.2f}" height="12" fill="{COLORS[interval.kind]}"/>')
    overview.append('<rect id="timeline-selection" x="220" y="22" width="960" height="24" fill="#3b82f6" fill-opacity=".15" stroke="#60a5fa" stroke-width="2" pointer-events="none"/></svg></div>')
    body = legend + "".join(overview) + controls + '<div id="visible-range" class="muted" aria-live="polite"></div><div class="panel chart">' + "".join(svg) + "</div>"
    body += '<p class="muted">Opens on the latest activity cluster (up to 15 minutes). Bars show recorded duration (minimum display width: 3 px). Thin markers show timestamps without duration. Gaps mean no recorded activity; they may include waiting or unobserved work. Hover for exact times. Empty tracks are hidden.</p>'
    subtitle = f"{len(lanes)} session(s), longest first · {_local_time(chart_start)} — {_local_time(chart_end)} · generated {_local_time(now)}"
    return _page("TokenMon activity timeline", subtitle, body)
