import streamlit as st

PALETTE = ["#7aa2ff", "#ffb86b", "#7be0b0", "#d59bff", "#ff7a90", "#f5e663"]
GRID = "rgba(158,193,255,0.12)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;500;600&display=swap');
.stApp {
background-color: #3A4261;
background-image:
radial-gradient(1px 1px at 23px 41px, rgba(255,255,255,0.9), transparent),
radial-gradient(1.5px 1.5px at 97px 133px, rgba(207,216,255,0.9), transparent),
radial-gradient(1px 1px at 181px 77px, rgba(255,233,196,0.8), transparent),
radial-gradient(2px 2px at 47px 203px, rgba(255,255,255,0.7), transparent),
radial-gradient(ellipse at 12% 0%, rgba(53,79,190,0.45), transparent 55%),
radial-gradient(ellipse at 92% 12%, rgba(110,52,170,0.35), transparent 50%);
background-size: 211px 173px, 307px 251px, 389px 283px, 521px 397px, 100% 100%, 100% 100%;
background-repeat: repeat, repeat, repeat, repeat, no-repeat, no-repeat;
background-attachment: fixed;
font-family: 'Inter', sans-serif;
}
header[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
background: linear-gradient(180deg, #0c1230, #090e24);
border-right: 1px solid rgba(122,162,255,0.18);
}
h1, h2, h3 { font-family: 'Orbitron', sans-serif; letter-spacing: 0.03em; }
h1 {
background: linear-gradient(90deg, #9ec1ff, #d59bff);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}
div[data-testid="stMetric"] {
background: rgba(16,23,51,0.65);
border: 1px solid rgba(122,162,255,0.22);
border-radius: 14px;
padding: 14px 16px;
}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * { color: #dbe4ff !important; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span { color: #eef2ff; }
div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] *, .stApp small { color: #cfdaff !important; opacity: 1 !important; }
div[data-testid="stWidgetLabel"] *, section[data-testid="stSidebar"] * { color: #eef2ff; }
section[data-testid="stSidebar"] a span { color: #eef2ff !important; }
div[data-testid="stMetricValue"] { font-family: 'Orbitron', sans-serif; }
div[data-testid="stDataFrame"] {
border: 1px solid rgba(122,162,255,0.2);
border-radius: 12px;
overflow: hidden;
}
div[data-testid="stAlert"] { border-radius: 12px; }
.stButton > button {
border-radius: 999px;
border: 1px solid #7aa2ff;
background: linear-gradient(90deg, #2c4cc5, #6b3fd1);
color: #ffffff;
}
.hero {
position: relative;
overflow: hidden;
border-radius: 20px;
padding: 36px 40px;
margin-bottom: 22px;
background: linear-gradient(120deg, rgba(20,30,80,0.88), rgba(60,30,110,0.78));
border: 1px solid rgba(122,162,255,0.25);
box-shadow: 0 10px 40px rgba(0,0,40,0.5);
}
.hero-kicker { font-size: 0.72rem; letter-spacing: 0.28em; color: #dbe4ff; }
.hero-title { font-family: 'Orbitron', sans-serif; font-size: 2rem; font-weight: 700; color: #ffffff; margin: 10px 0 8px 0; line-height: 1.25; max-width: 620px; }
.hero-sub { color: #eef2ff; max-width: 600px; font-size: 1rem; }
.hero-star {
position: absolute; right: 60px; top: 50%; width: 120px; height: 120px; margin-top: -60px;
border-radius: 50%;
background: radial-gradient(circle at 40% 40%, #fff7d6, #ffd36b 55%, #ff9a3c);
box-shadow: 0 0 60px 20px rgba(255,190,90,0.45);
}
.hero-planet {
position: absolute; right: 190px; top: 50%; width: 26px; height: 26px; margin-top: -13px;
border-radius: 50%; background: #05060f;
animation: transit 7s ease-in-out infinite alternate;
}
@keyframes transit { from { transform: translateX(0px); } to { transform: translateX(165px); } }
@media (max-width: 800px) { .hero-star, .hero-planet { display: none; } }
.step {
background: 3A4261;
border: 1px solid rgba(122,162,255,0.18);
border-radius: 14px;
padding: 16px 18px;
color: #eef2ff;
min-height: 110px;
}
.step-title { font-family: 'Orbitron', sans-serif; color: #b8d0ff; margin-bottom: 6px; }
.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp div[data-testid="stMarkdownContainer"],
.stApp div[data-testid="stMarkdownContainer"] *, div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] *, .stApp small { color: #ffffff !important; opacity: 1 !important; }
section[data-testid="stSidebar"], section[data-testid="stSidebar"] *,
div[data-testid="stSidebarNav"] a, div[data-testid="stSidebarNav"] a span,
[data-testid="stSidebarNavLink"], [data-testid="stSidebarNavLink"] span { color: #ffffff !important; opacity: 1 !important; }
h1 { -webkit-text-fill-color: transparent !important; }
</style>
"""


def apply_style():
    """Inject the astronomy theme (call right after st.set_page_config)."""
    st.markdown(CSS, unsafe_allow_html=True)


def hero(kicker, title, subtitle):
    html = ('<div class="hero"><div class="hero-star"></div><div class="hero-planet"></div>'
            f'<div class="hero-kicker">{kicker}</div>'
            f'<div class="hero-title">{title}</div>'
            f'<div class="hero-sub">{subtitle}</div></div>')
    st.markdown(html, unsafe_allow_html=True)


def step_card(number, title, text):
    st.markdown(f'<div class="step"><div class="step-title">{number} | {title}</div>'
                f'<div>{text}</div></div>', unsafe_allow_html=True)


def style_fig(fig):
    """Dark space look for a Plotly figure."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(12,18,45,0.55)", font=dict(color="#f2f5ff"),
                      colorway=PALETTE, legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def show_chart(fig, key):
    st.plotly_chart(style_fig(fig), key=key, theme=None)