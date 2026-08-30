"""
NeuroResearch Observability Dashboard — Streamlit

Shows:
  • Per-session trace timelines (which agent ran, how long, any errors)
  • Critique score trends across sessions
  • Iteration efficiency distribution
  • Eval results viewer (if eval reports exist)

Run:
    streamlit run src/dashboard.py

Requires:
    pip install streamlit plotly
"""

import json
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="NeuroResearch — Observability",
    page_icon="🧠",
    layout="wide",
)

# ── Load DB ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _setup():
    """Initialize DB (creates tables if missing)."""
    os.environ.setdefault("HF_API_TOKEN", "placeholder")
    os.environ.setdefault("JWT_SECRET",   "placeholder")
    from src.database import init_db
    init_db()

_setup()

from src.database import get_all_trace_sessions, get_trace_events, get_conn


def load_sessions():
    return get_all_trace_sessions(limit=200)


def load_events(session_id: str):
    return get_trace_events(session_id)


def load_eval_reports() -> list[dict]:
    """Load all versioned eval JSON reports from data/eval_results/."""
    try:
        from src.config.settings import get_settings
        base = Path(get_settings().EVAL_OUTPUT_DIR)
    except Exception:
        base = Path("data/eval_results")
    reports = []
    if base.exists():
        for f in sorted(base.glob("*.json"), reverse=True):
            try:
                with open(f) as fp:
                    data = json.load(fp)
                data["_filename"] = f.name
                reports.append(data)
            except Exception:
                pass
    return reports


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("🧠 NeuroResearch")
st.sidebar.caption("Observability Dashboard")
page = st.sidebar.radio(
    "View",
    ["Session Traces", "Score Trends", "Eval Results"],
    index=0,
)

# ── Page: Session Traces ──────────────────────────────────────────────────────

if page == "Session Traces":
    st.title("Session Trace Timeline")
    sessions = load_sessions()

    if not sessions:
        st.info("No sessions with trace data yet. Run some research first.")
        st.stop()

    # Session selector
    options = {
        f"{s['topic'][:55]}  [{s['created_at'][:10]}]  score={s['critique_score']:.2f}": s["id"]
        for s in sessions
    }
    selected_label = st.selectbox("Select session", list(options.keys()))
    session_id     = options[selected_label]

    # Session summary
    sess = next(s for s in sessions if s["id"] == session_id)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critique Score", f"{sess['critique_score']:.2f}")
    c2.metric("Iterations",     sess["iterations"])
    c3.metric("Total Latency",  f"{(sess['total_latency_ms'] or 0)/1000:.1f}s")
    c4.metric("Errors",         sess["error_count"])

    # Trace events
    events = load_events(session_id)
    if not events:
        st.warning("No trace events recorded for this session. "
                   "Events are captured from the next research run onwards.")
        st.stop()

    st.subheader("Agent Timeline")
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd

        df = pd.DataFrame(events)
        df["start"] = pd.to_datetime(df["created_at"])
        df["end"]   = df["start"] + pd.to_timedelta(df["latency_ms"], unit="ms")
        df["label"] = df.apply(
            lambda r: f"{r['node_name']} iter={r['iteration']}  {r['latency_ms']:.0f}ms"
                      + (" ⚠" if r["error"] else ""),
            axis=1,
        )

        color_map = {"research": "#582CFF", "write": "#8B5CF6", "critique": "#00F2FF"}
        df["color"] = df["node_name"].map(color_map).fillna("#888")

        fig = px.timeline(
            df, x_start="start", x_end="end", y="node_name",
            color="node_name",
            color_discrete_map=color_map,
            hover_name="label",
            title="Agent execution timeline",
        )
        fig.update_layout(
            paper_bgcolor="#0A0A0F",
            plot_bgcolor="#111118",
            font_color="#FFFFFF",
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        # Fallback: table view
        st.dataframe(
            [{"node": e["node_name"], "iter": e["iteration"],
              "latency_ms": f"{e['latency_ms']:.0f}",
              "error": e["error"] or "—"} for e in events],
            use_container_width=True,
        )

    # Raw events table
    with st.expander("Raw trace events"):
        st.json(events)


# ── Page: Score Trends ────────────────────────────────────────────────────────

elif page == "Score Trends":
    st.title("Score & Latency Trends")
    sessions = load_sessions()

    if not sessions:
        st.info("No sessions yet.")
        st.stop()

    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(sessions)
        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")
        df["total_s"] = (df["total_latency_ms"].fillna(0) / 1000).round(1)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                df, x="created_at", y="critique_score",
                size="iterations", color="iterations",
                hover_name="topic",
                title="Critique score over time",
                range_y=[0, 1],
                color_continuous_scale="Viridis",
            )
            fig.update_layout(paper_bgcolor="#0A0A0F", plot_bgcolor="#111118",
                              font_color="#FFF")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                df, x="created_at", y="total_s",
                hover_name="topic",
                title="Total latency per session (seconds)",
                color="critique_score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 1],
            )
            fig.update_layout(paper_bgcolor="#0A0A0F", plot_bgcolor="#111118",
                              font_color="#FFF")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Iteration distribution")
        fig = px.histogram(df, x="iterations", title="Iterations to convergence",
                           nbins=6, color_discrete_sequence=["#582CFF"])
        fig.update_layout(paper_bgcolor="#0A0A0F", plot_bgcolor="#111118", font_color="#FFF")
        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.warning("pip install plotly pandas  for charts")
        st.table([{"topic": s["topic"][:50], "score": s["critique_score"],
                   "iters": s["iterations"]} for s in sessions])


# ── Page: Eval Results ────────────────────────────────────────────────────────

elif page == "Eval Results":
    st.title("Eval Benchmark Results")
    reports = load_eval_reports()

    if not reports:
        st.info("No eval reports yet. Run:  python -m src.evals.harness")
        st.stop()

    # Report selector
    report_labels = [r["_filename"] for r in reports]
    selected_file = st.selectbox("Eval report", report_labels)
    report        = next(r for r in reports if r["_filename"] == selected_file)
    meta          = report.get("meta", {})
    results       = report.get("results", [])

    # Meta summary
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Model",      meta.get("model_id", "—"))
    m2.metric("Passed",     f"{meta.get('passed',0)}/{meta.get('total',0)}")
    m3.metric("Avg Overall",f"{meta.get('avg_overall', 0):.3f}")
    m4.metric("Ret Precision", f"{meta.get('avg_retrieval_precision', 0):.3f}")
    m5.metric("Ret Recall",    f"{meta.get('avg_retrieval_recall', 0):.3f}")

    # Results table
    st.subheader("Per-topic breakdown")
    try:
        import plotly.express as px
        import pandas as pd

        df = pd.DataFrame(results)
        df["pass_icon"] = df["passed"].map({True: "✅", False: "❌"})

        # Breakdown columns
        breakdown_cols = []
        if results and results[0].get("breakdown"):
            breakdown_cols = [k for k, v in results[0]["breakdown"].items() if v is not None]

        display_cols = ["pass_icon", "topic", "difficulty", "overall",
                        "critique_score", "iterations", "elapsed_seconds"] + breakdown_cols
        available    = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available].round(3), use_container_width=True)

        # Chart: overall by difficulty
        if "difficulty" in df.columns:
            fig = px.box(df, x="difficulty", y="overall",
                         color="difficulty", title="Overall score by difficulty",
                         category_orders={"difficulty": ["easy","medium","hard","adversarial"]})
            fig.update_layout(paper_bgcolor="#0A0A0F", plot_bgcolor="#111118", font_color="#FFF")
            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        for r in results:
            st.write(r)

    with st.expander("Raw JSON"):
        st.json(report)
