"""
PropFirmX — AI Debate Trading Terminal
Gemini + Claude ทำงานร่วมกันแบบ Debate Pattern พร้อม Market Scanner
"""

import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pydantic import BaseModel

from google import genai
from google.genai import types
import anthropic

# ============================================================
# ⚙️ CONFIG
# ============================================================

st.set_page_config(
    page_title="PropFirmX — AI Debate Terminal",
    layout="wide",
    page_icon="◆",
    initial_sidebar_state="expanded"
)

def get_secret(key: str) -> str:
    """อ่าน key จาก Streamlit secrets ก่อน ถ้าไม่มีลอง env var (สำหรับ local + cloud)"""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")

# ============================================================
# 📊 SCHEMAS & WATCHLISTS
# ============================================================

class GeminiOpinion(BaseModel):
    market_sentiment: str       # มุมมองตลาดโดยรวม
    initial_signal: str         # BUY / SELL / HOLD (มุมมองแรก)
    key_observation: str        # สิ่งที่สังเกตเห็นจากข้อมูล
    confidence: str             # สูง / กลาง / ต่ำ

class ClaudeVerdict(BaseModel):
    agrees_with_gemini: bool
    final_signal: str           # BUY / SELL / HOLD
    risk_level: str             # ต่ำ / กลาง / สูง
    support_zone: str
    resistance_zone: str
    challenge_notes: str        # สิ่งที่ Claude ท้าทาย/แก้ไขจาก Gemini
    final_reasoning: str        # เหตุผลสรุปสุดท้าย

WATCHLISTS = {
    "🔥 US Tech Giants": ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"],
    "🇹🇭 Thai SET50 (Top)": ["PTT.BK", "AOT.BK", "CPALL.BK", "ADVANC.BK", "DELTA.BK"],
    "🪙 Crypto Majors": ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"],
    "💱 Forex Pairs": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"]
}

# ============================================================
# 🔄 SESSION STATE
# ============================================================

defaults = {
    "active_ticker": "AAPL",
    "timeframe": "6M (รายวัน)",
    "debate_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

tf_mapping = {
    "1D (1 นาที)": {"period": "1d", "interval": "1m", "tv": "1"},
    "1W (15 นาที)": {"period": "7d", "interval": "15m", "tv": "15"},
    "1M (รายวัน)": {"period": "1mo", "interval": "1d", "tv": "D"},
    "6M (รายวัน)": {"period": "6mo", "interval": "1d", "tv": "D"},
    "1Y (รายสัปดาห์)": {"period": "1y", "interval": "1wk", "tv": "W"},
}
current_tf = tf_mapping[st.session_state.timeframe]

# ============================================================
# 🎨 THEME — Dark Terminal
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #0a0c10;
    --panel: #12151c;
    --panel-2: #181c25;
    --border: #232834;
    --text: #e8eaed;
    --text-dim: #7b8494;
    --gemini: #4fb3a9;
    --claude: #d97757;
    --verdict: #c9a86a;
    --green: #4ade80;
    --red: #f87171;
}

html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
.stApp { background: var(--bg); color: var(--text); }
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }

/* Header band */
.term-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 22px; background: linear-gradient(90deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--border); border-radius: 12px; margin-bottom: 18px;
}
.term-header .brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; color: var(--text); }
.term-header .brand span { color: var(--verdict); }
.term-header .tag { color: var(--text-dim); font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }

/* AI vs AI banner */
.vs-banner {
    display: flex; align-items: center; gap: 0; border-radius: 12px; overflow: hidden;
    border: 1px solid var(--border); margin-bottom: 20px;
}
.vs-side { flex: 1; padding: 16px 20px; position: relative; }
.vs-gemini { background: linear-gradient(135deg, rgba(79,179,169,0.12), rgba(79,179,169,0.03)); border-right: 1px solid var(--border); }
.vs-claude { background: linear-gradient(135deg, rgba(217,119,87,0.03), rgba(217,119,87,0.12)); }
.vs-label { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 0.95rem; }
.vs-gemini .vs-label { color: var(--gemini); }
.vs-claude .vs-label { color: var(--claude); }
.vs-sub { color: var(--text-dim); font-size: 0.72rem; margin-top: 2px; }
.vs-divider {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--verdict);
    padding: 0 4px; font-size: 1.1rem;
}

/* Card */
.card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 14px;
}
.card-eyebrow {
    font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-dim);
    margin-bottom: 8px; font-weight: 500;
}
.card-title { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.05rem; margin-bottom: 10px; }

/* Verdict box */
.verdict-box {
    background: linear-gradient(135deg, rgba(201,168,106,0.10), rgba(201,168,106,0.02));
    border: 1px solid rgba(201,168,106,0.35); border-radius: 12px; padding: 20px;
}
.verdict-signal { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.8rem; letter-spacing: 0.02em; }
.signal-buy { color: var(--green); }
.signal-sell { color: var(--red); }
.signal-hold { color: var(--verdict); }

.pill {
    display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
    font-weight: 500; border: 1px solid var(--border); color: var(--text-dim);
}
.pill-agree { color: var(--green); border-color: rgba(74,222,128,0.4); background: rgba(74,222,128,0.08); }
.pill-disagree { color: var(--red); border-color: rgba(248,113,113,0.4); background: rgba(248,113,113,0.08); }
.divider-thin { border-top: 1px solid var(--border); margin: 14px 0; }

/* Overrides */
.stButton > button {
    border-radius: 8px; border: 1px solid var(--border); font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; transition: all 0.15s ease;
}
.stButton > button:hover { border-color: var(--verdict); color: var(--verdict); }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.stTextInput input, .stSelectbox > div > div { background: var(--panel-2) !important; border-color: var(--border) !important; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }
.stDataFrame { border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🛠️ DATA HELPERS
# ============================================================

@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period: str, interval: str):
    dat = yf.Ticker(ticker)
    df = dat.history(period=period, interval=interval)
    if not df.empty and isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def compute_indicators(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 20:
        return None
        
    close = df["Close"]
    volume = df["Volume"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    rsi = float((100 - (100 / (1 + rs))).iloc[-1])

    ma20 = float(close.rolling(20).mean().iloc[-1])
    std20 = float(close.rolling(20).std().iloc[-1])
    bb_upper = ma20 + std20 * 2
    bb_lower = ma20 - std20 * 2

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])

    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_now = float(volume.iloc[-1])
    vol_ratio = round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = round((price - prev) / prev * 100, 2)
    
    # เพิ่มการส่งข้อมูล 5 แท่งล่าสุดให้ AI เห็นเทรนด์
    recent_prices = close.tail(5).round(2).tolist()

    return {
        "price": round(price, 2),
        "change_pct": change_pct,
        "rsi": round(rsi, 2),
        "ma20": round(ma20, 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "macd_hist": round(macd_hist, 4),
        "vol_ratio": vol_ratio,
        "recent_prices": recent_prices
    }

# ============================================================
# 🤖 AI LOGIC
# ============================================================

def gemini_first_opinion(ticker: str, ind: dict) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    คุณคือนักวิเคราะห์ตลาดที่มองภาพกว้างและจับ sentiment ตลาดได้ไว
    วิเคราะห์หุ้น {ticker} จากข้อมูลทางเทคนิคนี้:
    - ราคาปัจจุบัน: ${ind['price']} ({ind['change_pct']:+}% วันนี้)
    - ราคา 5 แท่งล่าสุด (เทรนด์): {ind['recent_prices']}
    - RSI(14): {ind['rsi']}
    - MA20: ${ind['ma20']}
    - Bollinger Bands: บน ${ind['bb_upper']} / ล่าง ${ind['bb_lower']}
    - MACD Histogram: {ind['macd_hist']}
    - Volume Ratio: {ind['vol_ratio']}x ของค่าเฉลี่ย 20 วัน

    ให้ความเห็นเบื้องต้นแบบนักวิเคราะห์ที่มองโอกาสและความเสี่ยงในตลาด
    นี่เป็นความเห็น 'รอบแรก' เท่านั้น จะมีนักวิเคราะห์สายคุมความเสี่ยงมาตรวจสอบคุณอีกที
    """
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiOpinion,
            temperature=0.4,
            system_instruction="ตอบเป็นภาษาไทย 100% กระชับ คมคาย ไม่ใช้ภาษาอังกฤษปนยกเว้นคำศัพท์เฉพาะทางการเงิน"
        )
    )
    return json.loads(response.text)

def claude_challenge_and_verdict(ticker: str, ind: dict, gemini_opinion: dict) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""คุณคือนักวิเคราะห์ความเสี่ยงระดับสูง (Risk Manager) ที่ตรวจทานความเห็นของนักวิเคราะห์คนอื่นอย่างเข้มงวด

ข้อมูลทางเทคนิคของหุ้น {ticker}:
- ราคาปัจจุบัน: ${ind['price']} ({ind['change_pct']:+}% วันนี้)
- ราคา 5 แท่งล่าสุด (เทรนด์): {ind['recent_prices']}
- RSI(14): {ind['rsi']}
- MA20: ${ind['ma20']}
- Bollinger: บน ${ind['bb_upper']} / ล่าง ${ind['bb_lower']}
- MACD Histogram: {ind['macd_hist']}
- Volume Ratio: {ind['vol_ratio']}x

ความเห็นรอบแรกจากนักวิเคราะห์สายเทรนด์ (Gemini):
- มุมมองตลาด: {gemini_opinion['market_sentiment']}
- สัญญาณเบื้องต้น: {gemini_opinion['initial_signal']}
- สิ่งที่สังเกตเห็น: {gemini_opinion['key_observation']}
- ความมั่นใจ: {gemini_opinion['confidence']}

หน้าที่ของคุณ:
1. ตรวจสอบว่าความเห็นนี้สมเหตุสมผลกับข้อมูลหรือไม่ ท้าทายจุดที่อ่อน หรือจุดที่มองข้ามความเสี่ยง
2. ให้สัญญาณสุดท้าย BUY/SELL/HOLD ที่อาจเหมือนหรือต่างจาก Gemini ก็ได้
3. ระบุระดับความเสี่ยง แนวรับ-แนวต้านหลัก
4. อธิบายเหตุผลสรุปสุดท้ายอย่างตรงไปตรงมา อ้างอิงตัวเลขทางเทคนิค

ตอบเป็นภาษาไทยกระชับ ชัดเจน ไม่เกรงใจหากต้องแย้งความเห็นแรก"""

    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        system="คุณตอบกลับเป็น JSON ที่ตรงกับ schema เท่านั้น ห้ามมีข้อความอื่น",
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "name": "submit_verdict",
            "description": "ส่งคำตัดสินสุดท้ายหลังตรวจทาน",
            "input_schema": ClaudeVerdict.model_json_schema()
        }],
        tool_choice={"type": "tool", "name": "submit_verdict"}
    )

    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("Claude ไม่ได้ส่งคำตัดสินกลับมา")

def run_debate(ticker: str, ind: dict) -> dict:
    gemini_op = gemini_first_opinion(ticker, ind)
    claude_v = claude_challenge_and_verdict(ticker, ind, gemini_op)
    return {"gemini": gemini_op, "claude": claude_v, "indicators": ind, "ticker": ticker}

# ============================================================
# 📌 SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("<h3 style='color:#c9a86a;'>◆ PropFirmX</h3>", unsafe_allow_html=True)
    st.caption("AI Debate Terminal — Gemini × Claude")
    st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)

    # เชื่อมต่อกับ Ticker ใน Session State
    def update_ticker():
        st.session_state.debate_result = None

    search_ticker = st.text_input("Ticker (หุ้นที่ต้องการวิเคราะห์)", 
                                  value=st.session_state.active_ticker, 
                                  on_change=update_ticker).upper()
    
    if search_ticker and search_ticker != st.session_state.active_ticker:
        st.session_state.active_ticker = search_ticker
        st.session_state.debate_result = None

    selected_tf = st.selectbox("Timeframe", list(tf_mapping.keys()), 
                               index=list(tf_mapping.keys()).index(st.session_state.timeframe))
    if selected_tf != st.session_state.timeframe:
        st.session_state.timeframe = selected_tf
        st.session_state.debate_result = None
        st.rerun()

    st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)

    if not GEMINI_API_KEY or not ANTHROPIC_API_KEY:
        st.warning("⚠️ ยังไม่ได้ตั้งค่า API Key ครบ")
    else:
        st.success("✅ API Keys พร้อมใช้งาน")

    run_clicked = st.button("▶ เริ่ม AI Debate", type="primary", use_container_width=True, 
                             disabled=not (GEMINI_API_KEY and ANTHROPIC_API_KEY))

# ============================================================
# 📌 MAIN LAYOUT (TABS)
# ============================================================

tab_terminal, tab_scanner = st.tabs(["📊 Terminal & Debate", "📡 Market Scanner (ค้นหาหุ้น)"])

# ------------------------------------------------------------
# 🎯 TAB 1: TERMINAL & DEBATE
# ------------------------------------------------------------
with tab_terminal:
    ticker = st.session_state.active_ticker

    st.markdown(f"""
    <div class="term-header">
        <div>
            <div class="brand">PROP<span>FIRM</span>X</div>
            <div class="tag">AI Debate Trading Terminal</div>
        </div>
        <div class="tag">{datetime.now().strftime('%Y-%m-%d %H:%M')} · {ticker}</div>
    </div>
    """, unsafe_allow_html=True)

    # ดึงข้อมูล
    df = fetch_price_data(ticker, current_tf["period"], current_tf["interval"])
    
    if df.empty:
        st.error(f"❌ ไม่พบข้อมูลของ {ticker} ในช่วงเวลานี้ (อาจพิมพ์ชื่อผิด หรือตลาดเพิ่งเปิด/ปิด)")
    else:
        ind = compute_indicators(df)
        if not ind:
            st.warning("⚠️ ข้อมูลย้อนหลังไม่เพียงพอสำหรับคำนวณ Indicator (ต้องการอย่างน้อย 20 แท่ง)")
        else:
            # CHART
            tradingview_html = f"""
            <div class="tradingview-widget-container" style="height:380px;width:100%">
              <div id="tv_chart" style="height:100%;width:100%"></div>
              <script src="https://s3.tradingview.com/tv.js"></script>
              <script>
              new TradingView.widget({{
                "autosize": true, "symbol": "{ticker.replace('.BK', '')}", "interval": "{current_tf['tv']}",
                "timezone": "Asia/Bangkok", "theme": "dark", "style": "1", "locale": "th",
                "enable_publishing": false, "hide_side_toolbar": false,
                "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
                "container_id": "tv_chart", "backgroundColor": "#0a0c10", "gridColor": "#232834"
              }});
              </script>
            </div>
            """
            components.html(tradingview_html, height=390)

            # METRICS
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("ราคาปัจจุบัน", f"${ind['price']:,.2f}", f"{ind['change_pct']:+.2f}%")
            
            # เปลี่ยนสี RSI ถ้าร้อนแรง
            rsi_color = "normal"
            if ind['rsi'] > 70: rsi_color = "inverse" # แดง
            elif ind['rsi'] < 30: rsi_color = "normal" # เขียว
            m2.metric("RSI(14)", f"{ind['rsi']:.1f}", delta="Overbought" if ind['rsi']>70 else "Oversold" if ind['rsi']<30 else "", delta_color=rsi_color)
            
            m3.metric("MA20", f"${ind['ma20']:,.2f}")
            m4.metric("Volume", f"{ind['vol_ratio']:.2f}x", "เทียบค่าเฉลี่ย 20 วัน", delta_color="off")

            st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)

            # AI DEBATE SECTION
            st.markdown("""
            <div class="vs-banner">
                <div class="vs-side vs-gemini">
                    <div class="vs-label">◷ GEMINI</div>
                    <div class="vs-sub">นักวิเคราะห์ตลาด (Market Sentiment)</div>
                </div>
                <div class="vs-divider">⟷</div>
                <div class="vs-side vs-claude">
                    <div class="vs-label">◈ CLAUDE</div>
                    <div class="vs-sub">ผู้คุมความเสี่ยง (Risk & Validation)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if run_clicked:
                with st.spinner("🤖 AI กำลังถกเถียงและวิเคราะห์... (อาจใช้เวลา 10-20 วินาที)"):
                    try:
                        result = run_debate(ticker, ind)
                        st.session_state.debate_result = result
                    except Exception as e:
                        st.error(f"❌ เกิดข้อผิดพลาดระหว่าง AI Debate: {e}")

            result = st.session_state.debate_result

            if result and result.get("ticker") == ticker: # เช็คให้ชัวร์ว่าเป็นของตัวปัจจุบัน
                col_g, col_c = st.columns(2)

                with col_g:
                    g = result["gemini"]
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-eyebrow" style="color:#4fb3a9;">GEMINI · ความเห็นแรก</div>
                        <div class="card-title">{g['initial_signal']}</div>
                        <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">มุมมองตลาด</p>
                        <p style="font-size:0.9rem;">{g['market_sentiment']}</p>
                        <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px; margin-top:12px;">สิ่งที่สังเกตเห็นจากกราฟ</p>
                        <p style="font-size:0.9rem;">{g['key_observation']}</p>
                        <div class="divider-thin"></div>
                        <span class="pill">ความมั่นใจ: {g['confidence']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                with col_c:
                    c = result["claude"]
                    agree_class = "pill-agree" if c["agrees_with_gemini"] else "pill-disagree"
                    agree_text = "เห็นด้วยกับ Gemini" if c["agrees_with_gemini"] else "ไม่เห็นด้วยกับ Gemini"
                    st.markdown(f"""
                    <div class="card">
                        <div class="card-eyebrow" style="color:#d97757;">CLAUDE · ตรวจสอบ & ท้าทาย</div>
                        <div class="card-title">{c['final_signal']}</div>
                        <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">ข้อโต้แย้ง / จุดที่ต้องระวัง</p>
                        <p style="font-size:0.9rem;">{c['challenge_notes']}</p>
                        <div class="divider-thin"></div>
                        <span class="pill {agree_class}">{agree_text}</span>
                        <span class="pill">ระดับความเสี่ยง: {c['risk_level']}</span>
                    </div>
                    """, unsafe_allow_html=True)

                signal_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(
                    c["final_signal"].upper(), "signal-hold"
                )

                st.markdown(f"""
                <div class="verdict-box">
                    <div class="card-eyebrow" style="color:#c9a86a;">⚖ คำตัดสินและแผนการเทรดสุดท้าย</div>
                    <div class="verdict-signal {signal_class}">{c['final_signal']}</div>
                    <div class="divider-thin"></div>
                    <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">ระดับราคาที่ต้องจับตา</p>
                    <p style="font-size:0.95rem;">🛡 แนวรับ: <b>{c['support_zone']}</b> &nbsp;&nbsp;|&nbsp;&nbsp; 🚀 แนวต้าน: <b>{c['resistance_zone']}</b></p>
                    <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px; margin-top:14px;">เหตุผลสรุป</p>
                    <p style="font-size:0.95rem;">{c['final_reasoning']}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("💡 กดปุ่ม **▶ เริ่ม AI Debate** ที่แถบด้านซ้าย เพื่อให้ AI เริ่มทำงาน")


# ------------------------------------------------------------
# 🎯 TAB 2: MARKET SCANNER
# ------------------------------------------------------------
with tab_scanner:
    st.markdown("### 📡 Market Scanner (รายวัน)")
    st.markdown("<p style='color:var(--text-dim); font-size:0.9rem;'>สแกนหาหุ้นที่มีสัญญาณทางเทคนิคน่าสนใจ เพื่อนำไปวิเคราะห์ต่อในหน้า Terminal</p>", unsafe_allow_html=True)
    
    col_w1, col_w2 = st.columns([1, 2])
    with col_w1:
        selected_list = st.selectbox("เลือกกลุ่มหุ้นที่ต้องการสแกน:", list(WATCHLISTS.keys()))
    
    if st.button("🔍 เริ่มสแกนกลุ่มนี้"):
        tickers_to_scan = WATCHLISTS[selected_list]
        scan_results = []
        
        progress_text = "กำลังดึงข้อมูลและคำนวณ Indicator..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, t in enumerate(tickers_to_scan):
            try:
                # ดึงข้อมูล 1 เดือนเพื่อคำนวณ Indicator แบบไวๆ
                sdf = fetch_price_data(t, "1mo", "1d")
                sind = compute_indicators(sdf)
                if sind:
                    # แปลงค่าเพื่อความสวยงามในตาราง
                    signal = "🟢 สวย" if sind['rsi'] < 40 and sind['macd_hist'] > 0 else "🔴 ระวัง" if sind['rsi'] > 70 else "⚪ ทรงตัว"
                    
                    scan_results.append({
                        "Ticker": t,
                        "Price": f"${sind['price']:.2f}",
                        "% Change": f"{sind['change_pct']:+.2f}%",
                        "RSI (14)": sind['rsi'],
                        "MACD Hist": sind['macd_hist'],
                        "Vol Ratio": f"{sind['vol_ratio']}x",
                        "Signal": signal
                    })
            except Exception:
                pass
            
            my_bar.progress((i + 1) / len(tickers_to_scan), text=progress_text)
            
        my_bar.empty()
        
        if scan_results:
            df_scan = pd.DataFrame(scan_results)
            
            # การแสดงผลแบบสีในตาราง
            def color_change(val):
                if isinstance(val, str) and '%' in val:
                    color = '#4ade80' if '+' in val else '#f87171'
                    return f'color: {color}'
                return ''
                
            styled_df = df_scan.style.map(color_change).background_gradient(subset=['RSI (14)'], cmap='RdYlGn_r', vmin=20, vmax=80)
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
            st.info("💡 นำชื่อ Ticker จากตารางด้านบน ไปใส่ในกล่องค้นหาด้านซ้ายเพื่อเปิด AI Debate ได้เลย")
        else:
            st.warning("ไม่สามารถสแกนข้อมูลได้ในขณะนี้")

st.markdown("<div style='text-align:center; color:#7b8494; font-size:0.75rem; margin-top:30px;'>"
            "PropFirmX Terminal © 2024 · ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน"
            "</div>", unsafe_allow_html=True)