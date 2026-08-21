"""Shared visual treatment for the Streamlit dashboard."""

from __future__ import annotations


def apply_style() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        :root {
          --ink: #17211d;
          --muted: #64716b;
          --line: #dce3df;
          --paper: #f7f9f7;
          --green: #176b52;
          --blue: #2b5f8a;
          --amber: #9a6418;
          --red: #a33b35;
        }
        .stApp { background: var(--paper); color: var(--ink); }
        [data-testid="stSidebar"] { background: #eef2ef; border-right: 1px solid var(--line); }
        [data-testid="stMainBlockContainer"] { padding-top: 2rem; max-width: 1600px; }
        h1, h2, h3 { color: var(--ink); letter-spacing: 0 !important; }
        h1 { font-size: 1.75rem !important; font-weight: 680 !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        .brand { font-weight: 760; font-size: 1.05rem; color: var(--green); padding-top: .35rem; }
        .eyebrow { color: var(--green); font-size: .72rem; font-weight: 700; text-transform: uppercase; }
        .section-label { color: var(--muted); font-size: .76rem; font-weight: 650; margin: .15rem 0 .35rem; }
        .answer { border-left: 3px solid var(--green); padding: .25rem 0 .25rem 1rem; font-size: 1.02rem; line-height: 1.72; }
        .empty-state { border: 1px dashed #bdc9c2; padding: 1.4rem; color: var(--muted); background: #fbfcfb; }
        .trace-step { border-left: 2px solid #afbeb6; padding: .15rem 0 .75rem .9rem; margin-left: .3rem; }
        .trace-step strong { color: var(--green); }
        [data-testid="stMetric"] { background: transparent; border-bottom: 2px solid var(--line); padding: .55rem 0 .7rem; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 4px; background: #fff; }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); }
        .stButton > button { border-radius: 4px; min-height: 2.4rem; }
        .stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"] { border-radius: 4px; }
        div[data-testid="stStatusWidget"] { border-radius: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, section: str) -> None:
    import streamlit as st

    st.markdown(f'<div class="eyebrow">{section}</div>', unsafe_allow_html=True)
    st.title(title)
