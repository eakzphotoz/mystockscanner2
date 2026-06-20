"""
PropFirmX — AI Debate Trading Terminal
Gemini + Claude ทำงานร่วมกันแบบ Debate Pattern:
  1) Gemini วิเคราะห์ก่อน (มุมมองตลาด/sentiment)
  2) Claude ท้าทาย ตรวจสอบ และสรุปคำแนะนำสุดท้าย

รันแบบ local:
    pip install -r requirements.txt
    streamlit run app.py
    ใส่ key ใน .env (ดู .env.example)

Deploy บน Streamlit Cloud:
    ใส่ key ใน Settings > Secrets แทน .env
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
# 📊 SCHEMAS
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
    action_summary: str         # สรุปสั้น 1-2 บรรทัด: ทำอะไร เพราะอะไร (ภาษาคนทั่วไปเข้าใจง่าย)
    entry_price: str            # ราคา/ช่วงราคาที่ควรเข้าซื้อ (ถ้า HOLD/SELL ให้ระบุ "-" หรือเงื่อนไข)
    stop_loss: str              # ราคาที่ควรตัดขาดทุน
    take_profit: str            # ราคาเป้าหมายที่ควรพิจารณาขายทำกำไร
    position_sizing_note: str   # คำแนะนำสั้นๆเรื่องสัดส่วนการลงทุน/ความเสี่ยงที่รับได้


# ============================================================
# 📋 รายชื่อหุ้นสำหรับ Scanner
# ============================================================

US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
             "AMD", "INTC", "NFLX", "AVGO", "ORCL", "CRM", "ADBE", "QCOM"]

SET_STOCKS = ["PTT.BK", "KBANK.BK", "SCB.BK", "AOT.BK", "CPALL.BK",
              "ADVANC.BK", "GULF.BK", "BBL.BK", "MINT.BK", "CPN.BK",
              "DELTA.BK", "BDMS.BK", "TRUE.BK", "TOP.BK", "IVL.BK"]

SCAN_VOL_MULTIPLIER = 2.0


# ============================================================
# 🔄 SESSION STATE
# ============================================================

defaults = {
    "active_ticker": "AAPL",
    "timeframe": "6M (รายวัน)",
    "debate_result": None,
    "debate_running": False,
    "scan_results": [],
    "scan_running": False,
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
# 🎨 THEME — Dark Terminal, ส้มทองแดง+เขียวมรกตคู่ AI
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
.verdict-signal {
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.8rem;
    letter-spacing: 0.02em;
}
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

/* Streamlit element overrides */
.stButton > button {
    border-radius: 8px; border: 1px solid var(--border); font-family: 'Space Grotesk', sans-serif;
    font-weight: 600; transition: all 0.15s ease;
}
.stButton > button:hover { border-color: var(--verdict); color: var(--verdict); }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--border); }
.stTextInput input, .stSelectbox > div > div { background: var(--panel-2) !important; border-color: var(--border) !important; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; }

/* Scanner */
.scan-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 10px;
}
.scan-badge {
    display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.68rem;
    font-weight: 600; margin-right: 4px; margin-bottom: 4px;
}
.badge-buy { background: rgba(74,222,128,0.12); color: var(--green); border: 1px solid rgba(74,222,128,0.3); }
.badge-sell { background: rgba(248,113,113,0.12); color: var(--red); border: 1px solid rgba(248,113,113,0.3); }
.badge-neutral { background: rgba(123,132,148,0.12); color: var(--text-dim); border: 1px solid var(--border); }
.badge-vol { background: rgba(201,168,106,0.12); color: var(--verdict); border: 1px solid rgba(201,168,106,0.3); }

[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; }

/* Quick action banner */
.action-banner {
    border-radius: 12px; padding: 16px 22px; margin-bottom: 16px;
    display: flex; align-items: center; gap: 16px; border: 1px solid;
}
.action-banner.buy { background: linear-gradient(90deg, rgba(74,222,128,0.14), rgba(74,222,128,0.03)); border-color: rgba(74,222,128,0.4); }
.action-banner.sell { background: linear-gradient(90deg, rgba(248,113,113,0.14), rgba(248,113,113,0.03)); border-color: rgba(248,113,113,0.4); }
.action-banner.hold { background: linear-gradient(90deg, rgba(201,168,106,0.14), rgba(201,168,106,0.03)); border-color: rgba(201,168,106,0.4); }
.action-banner .icon { font-size: 1.8rem; }
.action-banner .label { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.1rem; }
.action-banner .desc { color: var(--text-dim); font-size: 0.88rem; margin-top: 2px; }

/* Action plan grid */
.plan-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
.plan-cell {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px;
    padding: 12px 14px;
}
.plan-cell .plan-label { font-size: 0.7rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.plan-cell .plan-value { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1rem; }
.plan-cell.entry .plan-value { color: var(--gemini); }
.plan-cell.stop .plan-value { color: var(--red); }
.plan-cell.target .plan-value { color: var(--green); }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 🛠️ DATA HELPERS
# ============================================================

@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period: str, interval: str):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def compute_indicators(df: pd.DataFrame) -> dict:
    close = df["Close"]
    volume = df["Volume"]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
    loss = -delta.where(delta < 0, 0).ewm(alpha=1/14).mean()
    rsi = float((100 - (100 / (1 + (gain / loss)))).iloc[-1])

    ma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else float(close.iloc[-1])
    std20 = float(close.rolling(20).std().iloc[-1]) if len(close) >= 20 else 0.0
    bb_upper = ma20 + std20 * 2
    bb_lower = ma20 - std20 * 2

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    macd_hist = float((macd_line - signal_line).iloc[-1])

    vol_avg = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.iloc[-1])
    vol_now = float(volume.iloc[-1])
    vol_ratio = round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0

    price = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    change_pct = round((price - prev) / prev * 100, 2)

    return {
        "price": round(price, 2),
        "change_pct": change_pct,
        "rsi": round(rsi, 2),
        "ma20": round(ma20, 2),
        "bb_upper": round(bb_upper, 2),
        "bb_lower": round(bb_lower, 2),
        "macd_hist": round(macd_hist, 4),
        "vol_ratio": vol_ratio,
    }


def scan_one_ticker(ticker: str) -> dict | None:
    """สแกนหุ้น 1 ตัวด้วยทุกเงื่อนไข: RSI, MACD Cross, Bollinger Breakout, Volume Spike"""
    try:
        df = yf.download(ticker, period="3mo", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna()
        if df.empty or len(df) < 35:
            return None

        close = df["Close"]
        volume = df["Volume"]

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14).mean()
        rsi = float((100 - (100 / (1 + (gain / loss)))).iloc[-1])

        # MACD Cross
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        macd_now = float((macd_line - signal_line).iloc[-1])
        macd_prev = float((macd_line - signal_line).iloc[-2])
        macd_bullish_cross = macd_prev < 0 and macd_now > 0
        macd_bearish_cross = macd_prev > 0 and macd_now < 0

        # Bollinger Band
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = float((ma20 + std20 * 2).iloc[-1])
        bb_lower = float((ma20 - std20 * 2).iloc[-1])

        # Volume spike
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_now = float(volume.iloc[-1])
        vol_ratio = round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0
        vol_spike = vol_ratio >= SCAN_VOL_MULTIPLIER

        price = float(close.iloc[-1])
        prev_price = float(close.iloc[-2])
        change_pct = round((price - prev_price) / prev_price * 100, 2)

        # สร้างสัญญาณ
        signals = []
        if rsi < 30:
            signals.append(("RSI Oversold", "buy"))
        elif rsi > 70:
            signals.append(("RSI Overbought", "sell"))
        if macd_bullish_cross:
            signals.append(("MACD Bullish Cross", "buy"))
        elif macd_bearish_cross:
            signals.append(("MACD Bearish Cross", "sell"))
        if price > bb_upper:
            signals.append(("BB Breakout บน", "buy" if vol_spike else "neutral"))
        elif price < bb_lower:
            signals.append(("BB Breakout ล่าง", "sell" if vol_spike else "neutral"))
        if vol_spike:
            signals.append((f"Volume x{vol_ratio}", "vol"))

        if not signals:
            return None  # ไม่มีสัญญาณน่าสนใจ ไม่ต้องแสดง

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "change_pct": change_pct,
            "rsi": round(rsi, 2),
            "vol_ratio": vol_ratio,
            "signals": signals,
            "signal_count": len(signals),
        }
    except Exception:
        return None


def scan_market(tickers: list[str]) -> list[dict]:
    """สแกนหุ้นทั้งลิสต์ คืนเฉพาะตัวที่มีสัญญาณ เรียงตามจำนวนสัญญาณมากไปน้อย"""
    results = []
    for t in tickers:
        r = scan_one_ticker(t)
        if r:
            results.append(r)
    results.sort(key=lambda x: x["signal_count"], reverse=True)
    return results


# ============================================================
# 🤖 STEP 1 — GEMINI: ความเห็นแรก
# ============================================================

def gemini_first_opinion(ticker: str, ind: dict) -> dict:
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    คุณคือนักวิเคราะห์ตลาดที่มองภาพกว้างและจับ sentiment ตลาดได้ไว
    วิเคราะห์หุ้น {ticker} จากข้อมูลทางเทคนิคนี้:
    - ราคา: ${ind['price']} ({ind['change_pct']:+}% วันนี้)
    - RSI(14): {ind['rsi']}
    - MA20: ${ind['ma20']}
    - Bollinger: บน ${ind['bb_upper']} / ล่าง ${ind['bb_lower']}
    - MACD Histogram: {ind['macd_hist']}
    - Volume Ratio: {ind['vol_ratio']}x ของค่าเฉลี่ย

    ให้ความเห็นเบื้องต้นแบบนักวิเคราะห์ที่มองโอกาสและความเสี่ยงในตลาด
    นี่เป็นความเห็น 'รอบแรก' เท่านั้น จะมีนักวิเคราะห์อีกคนมาท้าทายความเห็นนี้ต่อ
    """
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GeminiOpinion,
            temperature=0.4,
            system_instruction="ตอบเป็นภาษาไทย 100% กระชับ คมคาย ไม่ใช้ภาษาอังกฤษปนยกเว้นคำศัพท์เฉพาะทางการเงิน"
        )
    )
    return json.loads(response.text)


# ============================================================
# 🤖 STEP 2 — CLAUDE: ท้าทาย + สรุปคำตัดสินสุดท้าย
# ============================================================

def claude_challenge_and_verdict(ticker: str, ind: dict, gemini_opinion: dict) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""คุณคือนักวิเคราะห์ความเสี่ยงระดับสูงที่ตรวจทานความเห็นของนักวิเคราะห์คนอื่นอย่างเข้มงวด

ข้อมูลทางเทคนิคของหุ้น {ticker}:
- ราคา: ${ind['price']} ({ind['change_pct']:+}% วันนี้)
- RSI(14): {ind['rsi']}
- MA20: ${ind['ma20']}
- Bollinger: บน ${ind['bb_upper']} / ล่าง ${ind['bb_lower']}
- MACD Histogram: {ind['macd_hist']}
- Volume Ratio: {ind['vol_ratio']}x

ความเห็นรอบแรกจากนักวิเคราะห์อีกคน (Gemini):
- มุมมองตลาด: {gemini_opinion['market_sentiment']}
- สัญญาณเบื้องต้น: {gemini_opinion['initial_signal']}
- สิ่งที่สังเกตเห็น: {gemini_opinion['key_observation']}
- ความมั่นใจ: {gemini_opinion['confidence']}

หน้าที่ของคุณ:
1. ตรวจสอบว่าความเห็นนี้สมเหตุสมผลกับข้อมูลทางเทคนิคหรือไม่ ท้าทายจุดที่อ่อนหรือมองข้ามไป
2. ให้สัญญาณสุดท้าย BUY/SELL/HOLD ที่อาจเหมือนหรือต่างจาก Gemini ก็ได้
3. ระบุระดับความเสี่ยง แนวรับ-แนวต้าน
4. อธิบายเหตุผลสรุปสุดท้ายอย่างตรงไปตรงมา
5. สรุป Action Plan ที่นำไปใช้ได้จริง:
   - action_summary: สรุป 1-2 บรรทัดสั้นๆว่าควรทำอะไรและเพราะอะไร ให้คนทั่วไปอ่านแล้วเข้าใจทันที
   - entry_price: ราคาหรือช่วงราคาที่เหมาะเข้าซื้อ (ถ้าแนะนำ HOLD/SELL ให้ใส่เงื่อนไขที่จะกลับมาซื้อ หรือ "-" ถ้าไม่เกี่ยวข้อง)
   - stop_loss: ราคาที่ควรตัดขาดทุนถ้าผิดทาง
   - take_profit: ราคาเป้าหมายที่ควรพิจารณาขายทำกำไร
   - position_sizing_note: คำแนะนำสั้นๆเรื่องสัดส่วนการลงทุนหรือการบริหารความเสี่ยง (เช่น ไม่ควรเกินกี่% ของพอร์ต)

สำคัญ: challenge_notes และ final_reasoning ให้เขียนกระชับ ไม่เกิน 3-4 บรรทัดต่อ field เพื่อให้ตอบ JSON ได้ครบทุก field

ตอบเป็นภาษาไทยกระชับ ชัดเจน ไม่เกรงใจหากต้องแย้งความเห็นแรก"""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system="คุณตอบกลับเป็น JSON ที่ตรงกับ schema เท่านั้น ห้ามมีข้อความอื่นนอก JSON ตอบให้ครบทุก field แบบกระชับ ไม่ต้องอธิบายยาวเกินจำเป็น",
        messages=[{"role": "user", "content": prompt}],
        tools=[{
            "name": "submit_verdict",
            "description": "ส่งคำตัดสินสุดท้ายหลังตรวจทานความเห็นของ Gemini",
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

    search_ticker = st.text_input("Ticker", value=st.session_state.active_ticker).upper()
    if search_ticker != st.session_state.active_ticker:
        st.session_state.active_ticker = search_ticker
        st.session_state.debate_result = None

    selected_tf = st.selectbox("Timeframe", list(tf_mapping.keys()),
                                index=list(tf_mapping.keys()).index(st.session_state.timeframe))
    if selected_tf != st.session_state.timeframe:
        st.session_state.timeframe = selected_tf
        st.rerun()

    st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)

    if not GEMINI_API_KEY or not ANTHROPIC_API_KEY:
        st.warning("⚠️ ยังไม่ได้ตั้งค่า API Key ครบ\n\nดู .env.example หรือ Streamlit Secrets")
    else:
        st.success("✅ API Keys พร้อมใช้งาน")

    run_clicked = st.button("▶ เริ่ม AI Debate", type="primary", use_container_width=True,
                             disabled=not (GEMINI_API_KEY and ANTHROPIC_API_KEY))

    st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)

    st.markdown("**🔍 Scanner**")
    scan_market_choice = st.selectbox("ตลาด", ["ทั้งสองตลาด", "หุ้นไทย (SET)", "หุ้นสหรัฐ (US)"])
    scan_clicked = st.button("🔍 สแกนหุ้น", use_container_width=True)


# ============================================================
# 📌 HEADER
# ============================================================

st.markdown(f"""
<div class="term-header">
    <div>
        <div class="brand">PROP<span>FIRM</span>X</div>
        <div class="tag">AI Debate Trading Terminal</div>
    </div>
    <div class="tag">{datetime.now().strftime('%Y-%m-%d %H:%M')} · {st.session_state.active_ticker}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 📌 SCANNER EXECUTION & RESULTS
# ============================================================

if scan_clicked:
    if scan_market_choice == "หุ้นไทย (SET)":
        tickers_to_scan = SET_STOCKS
    elif scan_market_choice == "หุ้นสหรัฐ (US)":
        tickers_to_scan = US_STOCKS
    else:
        tickers_to_scan = SET_STOCKS + US_STOCKS

    with st.spinner(f"🔍 กำลังสแกน {len(tickers_to_scan)} หุ้น..."):
        st.session_state.scan_results = scan_market(tickers_to_scan)

if st.session_state.scan_results:
    st.markdown(f"""
    <div class="scan-header">
        <div class="card-title" style="margin-bottom:0;">📋 ผลสแกน — พบ {len(st.session_state.scan_results)} หุ้นที่มีสัญญาณ</div>
    </div>
    """, unsafe_allow_html=True)

    for r in st.session_state.scan_results:
        badge_html = ""
        for label, kind in r["signals"]:
            cls = {"buy": "badge-buy", "sell": "badge-sell", "neutral": "badge-neutral", "vol": "badge-vol"}[kind]
            badge_html += f'<span class="scan-badge {cls}">{label}</span>'

        change_color = "var(--green)" if r["change_pct"] >= 0 else "var(--red)"

        col_info, col_badges, col_btn = st.columns([2, 5, 1.3])
        with col_info:
            st.markdown(f"""
            <div style="padding-top:6px;">
                <span style="font-family:'Space Grotesk',sans-serif; font-weight:600; font-size:1rem;">{r['ticker']}</span>
                <span style="color:{change_color}; font-size:0.85rem; margin-left:8px;">${r['price']} ({r['change_pct']:+.2f}%)</span>
            </div>
            """, unsafe_allow_html=True)
        with col_badges:
            st.markdown(f"<div style='padding-top:6px;'>{badge_html}</div>", unsafe_allow_html=True)
        with col_btn:
            if st.button("วิเคราะห์ →", key=f"select_{r['ticker']}", use_container_width=True):
                st.session_state.active_ticker = r["ticker"]
                st.session_state.debate_result = None
                st.rerun()

    st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)
else:
    st.caption("💡 กด **🔍 สแกนหุ้น** ที่แถบด้านซ้ายเพื่อค้นหาหุ้นที่มีสัญญาณ RSI / MACD / Bollinger / Volume น่าสนใจ")
    st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)


# ============================================================
# 📌 CHART
# ============================================================

ticker = st.session_state.active_ticker

tradingview_html = f"""
<div class="tradingview-widget-container" style="height:380px;width:100%">
  <div id="tv_chart" style="height:100%;width:100%"></div>
  <script src="https://s3.tradingview.com/tv.js"></script>
  <script>
  new TradingView.widget({{
    "autosize": true, "symbol": "{ticker}", "interval": "{current_tf['tv']}",
    "timezone": "Asia/Bangkok", "theme": "dark", "style": "1", "locale": "th",
    "enable_publishing": false, "hide_side_toolbar": false,
    "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
    "container_id": "tv_chart", "backgroundColor": "#0a0c10", "gridColor": "#232834"
  }});
  </script>
</div>
"""
components.html(tradingview_html, height=390)

try:
    df = fetch_price_data(ticker, current_tf["period"], current_tf["interval"])
    ind = compute_indicators(df)
except Exception as e:
    st.error(f"ดึงข้อมูลไม่ได้: {e}")
    ind = {"price": 0, "change_pct": 0, "rsi": 50, "ma20": 0, "bb_upper": 0, "bb_lower": 0, "macd_hist": 0, "vol_ratio": 1}

m1, m2, m3, m4 = st.columns(4)
m1.metric("ราคา", f"${ind['price']:,.2f}", f"{ind['change_pct']:+.2f}%")
m2.metric("RSI(14)", f"{ind['rsi']:.1f}")
m3.metric("MA20", f"${ind['ma20']:,.2f}")
m4.metric("Volume", f"{ind['vol_ratio']:.2f}x")

st.markdown("<div class='divider-thin'></div>", unsafe_allow_html=True)


# ============================================================
# 📌 AI DEBATE EXECUTION
# ============================================================

st.markdown("""
<div class="vs-banner">
    <div class="vs-side vs-gemini">
        <div class="vs-label">◷ GEMINI</div>
        <div class="vs-sub">ความเห็นรอบแรก · มุมมองตลาด</div>
    </div>
    <div class="vs-divider">⟷</div>
    <div class="vs-side vs-claude">
        <div class="vs-label">◈ CLAUDE</div>
        <div class="vs-sub">ท้าทาย · ตรวจสอบ · ตัดสิน</div>
    </div>
</div>
""", unsafe_allow_html=True)

if run_clicked:
    with st.spinner("Gemini กำลังวิเคราะห์รอบแรก..."):
        try:
            result = run_debate(ticker, ind)
            st.session_state.debate_result = result
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาดระหว่าง AI Debate: {e}")

result = st.session_state.debate_result

if result:
    c = result["claude"]
    signal_upper = c.get("final_signal", "HOLD").upper()
    banner_class = "buy" if signal_upper == "BUY" else "sell" if signal_upper == "SELL" else "hold"
    icon = "🟢" if banner_class == "buy" else "🔴" if banner_class == "sell" else "🟡"
    action_label = {"buy": "ซื้อ", "sell": "ขาย", "hold": "ถือต่อ"}[banner_class]

    st.markdown(f"""
    <div class="action-banner {banner_class}">
        <div class="icon">{icon}</div>
        <div>
            <div class="label">คำแนะนำ: {action_label} ({c.get('final_signal', '-')})</div>
            <div class="desc">{c.get('action_summary', '')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_g, col_c = st.columns(2)

    with col_g:
        g = result["gemini"]
        st.markdown(f"""
        <div class="card">
            <div class="card-eyebrow" style="color:#4fb3a9;">GEMINI · ความเห็นแรก</div>
            <div class="card-title">{g.get('initial_signal', '-')}</div>
            <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">มุมมองตลาด</p>
            <p style="font-size:0.9rem;">{g.get('market_sentiment', '-')}</p>
            <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px; margin-top:12px;">สิ่งที่สังเกตเห็น</p>
            <p style="font-size:0.9rem;">{g.get('key_observation', '-')}</p>
            <div class="divider-thin"></div>
            <span class="pill">ความมั่นใจ: {g.get('confidence', '-')}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        agree_class = "pill-agree" if c.get("agrees_with_gemini") else "pill-disagree"
        agree_text = "เห็นด้วยกับ Gemini" if c.get("agrees_with_gemini") else "ไม่เห็นด้วยกับ Gemini"
        st.markdown(f"""
        <div class="card">
            <div class="card-eyebrow" style="color:#d97757;">CLAUDE · ท้าทาย & ตัดสิน</div>
            <div class="card-title">{c.get('final_signal', '-')}</div>
            <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">ท้าทายความเห็นแรก</p>
            <p style="font-size:0.9rem;">{c.get('challenge_notes', '-')}</p>
            <div class="divider-thin"></div>
            <span class="pill {agree_class}">{agree_text}</span>
            <span class="pill">เสี่ยง: {c.get('risk_level', '-')}</span>
        </div>
        """, unsafe_allow_html=True)

    signal_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(
        c.get("final_signal", "HOLD").upper(), "signal-hold"
    )

    st.markdown(f"""
    <div class="verdict-box">
        <div class="card-eyebrow" style="color:#c9a86a;">⚖ คำตัดสินสุดท้าย</div>
        <div class="verdict-signal {signal_class}">{c.get('final_signal', '-')}</div>
        <div class="divider-thin"></div>
        <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">แนวรับ / แนวต้าน</p>
        <p style="font-size:0.95rem;">🛡 {c.get('support_zone', '-')} &nbsp;&nbsp;|&nbsp;&nbsp; 🚀 {c.get('resistance_zone', '-')}</p>
        <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px; margin-top:14px;">เหตุผลสรุปสุดท้าย</p>
        <p style="font-size:0.95rem;">{c.get('final_reasoning', '-')}</p>
        <div class="plan-grid">
            <div class="plan-cell entry">
                <div class="plan-label">จุดเข้าซื้อ</div>
                <div class="plan-value">{c.get('entry_price', '-')}</div>
            </div>
            <div class="plan-cell stop">
                <div class="plan-label">Stop Loss</div>
                <div class="plan-value">{c.get('stop_loss', '-')}</div>
            </div>
            <div class="plan-cell target">
                <div class="plan-label">Take Profit</div>
                <div class="plan-value">{c.get('take_profit', '-')}</div>
            </div>
        </div>
        <div class="divider-thin"></div>
        <p style="color:#7b8494; font-size:0.85rem; margin-bottom:6px;">💼 การบริหารความเสี่ยง</p>
        <p style="font-size:0.9rem;">{c.get('position_sizing_note', '-')}</p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("กด **▶ เริ่ม AI Debate** ที่แถบด้านซ้ายเพื่อให้ Gemini และ Claude ช่วยกันวิเคราะห์หุ้นนี้")

st.markdown("<div style='text-align:center; color:#7b8494; font-size:0.75rem; margin-top:24px;'>"
            "ข้อมูลนี้ใช้เพื่อการศึกษาเท่านั้น ไม่ใช่คำแนะนำการลงทุน"
            "</div>", unsafe_allow_html=True)