import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
import io
import json
import re
import random
import sqlite3
import os
import time
from datetime import datetime
from pydantic import BaseModel

from google import genai 
from google.genai import types
try:
    import anthropic
except ImportError:
    pass  # รองรับเผื่อบางสภาพแวดล้อมไม่มีไลบรารี anthropic

import journal  # 📓 โมดูลใหม่: Trade Journal + Win-Rate (แยกไฟล์ ไม่กระทบโค้ดเดิม)

# --- ⚙️ การตั้งค่าหน้าเว็บ (Premium Dark Theme) ---
st.set_page_config(
    page_title="PropFirmX - AI Debate & Shared Portfolio Terminal", 
    layout="wide", 
    page_icon="◆",
    initial_sidebar_state="expanded"
)

# ดึง API Key จาก Secrets / Env
def get_secret(key: str) -> str:
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key, "")

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
ANTHROPIC_API_KEY = get_secret("ANTHROPIC_API_KEY")

# ==========================================
# 🗄️ DATABASE SETUP (Persistent Storage)
# ==========================================
DB_FILE = 'family_portfolio.db'

# รายชื่อตารางที่อนุญาตให้ใช้งานเท่านั้น (ป้องกัน SQL Injection ผ่านชื่อตาราง
# แม้ปัจจุบัน table_name จะมาจากค่าคงที่ในโค้ดเท่านั้น แต่กันไว้เผื่ออนาคตมีการรับชื่อตารางจาก UI/input)
ALLOWED_PORTFOLIO_TABLES = {'port_us', 'port_th', 'port_crypto'}

def _validate_table_name(table_name):
    if table_name not in ALLOWED_PORTFOLIO_TABLES:
        raise ValueError(f"ชื่อตารางไม่ได้รับอนุญาต: {table_name}")

def init_db():
    """สร้างตารางฐานข้อมูล SQLite สำหรับเก็บพอร์ตร่วมกัน หากยังไม่มี"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    tables = {
        'port_us': [("AAPL", 15.0, 172.50), ("NVDA", 25.0, 110.00)],
        'port_th': [("PTT.BK", 1000.0, 32.50), ("CPALL.BK", 500.0, 57.00)],
        'port_crypto': [("BTC-USD", 0.05, 61500.00), ("ETH-USD", 0.50, 3100.00)]
    }
    
    for table_name, default_data in tables.items():
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                Ticker TEXT PRIMARY KEY,
                Shares REAL,
                AvgCost REAL
            )
        ''')
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        if cursor.fetchone()[0] == 0:
            cursor.executemany(f'INSERT INTO {table_name} (Ticker, Shares, AvgCost) VALUES (?, ?, ?)', default_data)
            
    conn.commit()
    conn.close()

def load_portfolio(table_name):
    """อ่านข้อมูลพอร์ตจาก SQLite เป็น DataFrame"""
    try:
        _validate_table_name(table_name)
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database Load Error ({table_name}): {e}")
        return pd.DataFrame(columns=["Ticker", "Shares", "AvgCost"])

def save_portfolio(table_name, df):
    """
    บันทึก DataFrame ลงตาราง SQLite ด้วยวิธี Upsert (อัปเดต/เพิ่ม/ลบเฉพาะแถวที่เปลี่ยนแปลงจริง)
    แทนการ DROP/REPLACE ตารางทั้งก้อนแบบเดิม เพื่อลดความเสี่ยงข้อมูลหายเวลามีคนแก้พอร์ต
    พร้อมกันจากหลายอุปกรณ์ (เช่น คู่รักเปิดพร้อมกันคนละมือถือ)

    หมายเหตุข้อจำกัด: วิธีนี้ลดความเสี่ยงจากการล้างตารางทั้งก้อน แต่ถ้าสองคนแก้ "แถวเดียวกัน"
    พร้อมกันเป๊ะๆ ระบบยังเป็นแบบ Last-Write-Wins อยู่ดี — ถ้าต้องการแก้ปัญหานี้ทั้งหมดต้องเพิ่มระบบ
    version/timestamp column + optimistic locking ซึ่งซับซ้อนกว่านี้
    """
    try:
        _validate_table_name(table_name)
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        clean_df = df.dropna(subset=["Ticker"]).copy()
        clean_df["Ticker"] = clean_df["Ticker"].astype(str).str.strip()
        clean_df = clean_df[clean_df["Ticker"] != ""]

        current_tickers = set(clean_df["Ticker"].tolist())

        cursor.execute(f'SELECT Ticker FROM {table_name}')
        existing_tickers = {row[0] for row in cursor.fetchall()}

        tickers_to_delete = existing_tickers - current_tickers
        if tickers_to_delete:
            cursor.executemany(
                f'DELETE FROM {table_name} WHERE Ticker = ?',
                [(t,) for t in tickers_to_delete]
            )

        upsert_rows = list(
            clean_df[["Ticker", "Shares", "AvgCost"]].itertuples(index=False, name=None)
        )
        if upsert_rows:
            cursor.executemany(
                f'INSERT OR REPLACE INTO {table_name} (Ticker, Shares, AvgCost) VALUES (?, ?, ?)',
                upsert_rows
            )

        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database Save Error ({table_name}): {e}")

# เริ่มต้นสร้าง Database (รันครั้งแรก)
if not os.path.exists(DB_FILE):
    init_db()
else:
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.close()
    except sqlite3.Error as e:
        print(f"DB file corrupted or unreadable, recreating: {e}")
        init_db()

journal.init_journal_db()  # 📓 สร้างตาราง trade_journal ถ้ายังไม่มี (ปลอดภัย ใช้ IF NOT EXISTS)

# ==========================================
# 📊 SCHEMAS & MODELS FOR DEBATE
# ==========================================
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
    action_summary: str         # สรุปสั้น 1-2 บรรทัดเข้าใจง่าย
    entry_price: str            # ราคา/ช่วงราคาที่ควรเข้าซื้อ
    stop_loss: str              # จุดตัดขาดทุน
    take_profit: str            # เป้ากำไร
    position_sizing_note: str   # คำแนะนำเรื่องการบริหารความเสี่ยง

class PortfolioAnalysisResult(BaseModel):
    diversification_score: str
    highest_risk_asset: str
    portfolio_pnl_summary: str
    strategic_advice: str

# --- 🔄 ระบบจำข้อมูลและสถานะเว็บ ---
if 'active_ticker' not in st.session_state:
    st.session_state.active_ticker = "AAPL"
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = []
if 'ai_debate_result' not in st.session_state:
    st.session_state.ai_debate_result = None
if 'ai_portfolio_analysis' not in st.session_state:
    st.session_state.ai_portfolio_analysis = None
if 'timeframe' not in st.session_state:
    st.session_state.timeframe = "6M (รายวัน)"
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# โหลดข้อมูลพอร์ตจากฐานข้อมูลเมื่อเริ่มเปิดหน้าเว็บ (เพื่อการแชร์ข้ามอุปกรณ์)
if 'port_us' not in st.session_state:
    st.session_state.port_us = load_portfolio('port_us')
if 'port_th' not in st.session_state:
    st.session_state.port_th = load_portfolio('port_th')
if 'port_crypto' not in st.session_state:
    st.session_state.port_crypto = load_portfolio('port_crypto')

# รายชื่อแถบวิ่งด้านบนสุด
TAPE_SYMBOLS = ["NASDAQ:AAPL", "NASDAQ:MSFT", "NASDAQ:NVDA", "NASDAQ:AMZN", "FX:EURUSD", "BITSTAMP:BTCUSD", "CMCMARKETS:GOLD"]

tf_mapping = {
    "1D (1 นาที)": {"period": "1d", "interval": "1m", "tv": "1"},
    "1W (15 นาที)": {"period": "7d", "interval": "15m", "tv": "15"},
    "1M (รายวัน)": {"period": "1mo", "interval": "1d", "tv": "D"},
    "6M (รายวัน)": {"period": "6mo", "interval": "1d", "tv": "D"},
    "1Y (รายสัปดาห์)": {"period": "1y", "interval": "1wk", "tv": "W"}
}
current_tf = tf_mapping[st.session_state.timeframe]

# --- 🎨 การฉีด CSS เพื่อคุมธีมสี Dark ระดับพรีเมียม (Hybrid AI Style) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg: #0a0c10;
        --panel: #111622;
        --panel-2: #181c25;
        --border: #1e293b;
        --text: #f1f5f9;
        --gemini: #4fb3a9;
        --claude: #d97757;
        --verdict: #c9a86a;
        --green: #10b981;
        --red: #ef4444;
    }
    
    html, body, [class*="css"] { font-family: 'JetBrains Mono', monospace; }
    .stApp { background-color: var(--bg); color: var(--text); }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
    
    .prop-card { background-color: var(--panel); border: 1px solid var(--border); padding: 16px; border-radius: 12px; margin-bottom: 15px; }
    .chat-bubble-user { background-color: #1e293b; border-radius: 10px; padding: 10px; margin-bottom: 8px; border-left: 4px solid #38bdf8; text-align: left; }
    .chat-bubble-ai { background-color: #161b26; border-radius: 10px; padding: 10px; margin-bottom: 8px; border-left: 4px solid var(--verdict); text-align: left; }
    
    /* VS Banner */
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
    .vs-sub { color: #7b8494; font-size: 0.72rem; margin-top: 2px; }
    .vs-divider { font-family: 'Space Grotesk', sans-serif; font-weight: 700; color: var(--verdict); padding: 0 10px; font-size: 1.1rem; }

    /* Verdict Box & Action Plan */
    .verdict-box {
        background: linear-gradient(135deg, rgba(201,168,106,0.10), rgba(201,168,106,0.02));
        border: 1px solid rgba(201,168,106,0.35); border-radius: 12px; padding: 20px; margin-top: 15px;
    }
    .verdict-signal { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.8rem; letter-spacing: 0.02em; }
    .signal-buy { color: var(--green); }
    .signal-sell { color: var(--red); }
    .signal-hold { color: var(--verdict); }
    
    .pill { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 500; border: 1px solid var(--border); color: #7b8494; margin-right: 5px;}
    .pill-agree { color: var(--green); border-color: rgba(16,185,129,0.4); background: rgba(16,185,129,0.08); }
    .pill-disagree { color: var(--red); border-color: rgba(239,68,68,0.4); background: rgba(239,68,68,0.08); }

    .plan-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 14px; }
    .plan-cell { background: var(--panel-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px; }
    .plan-cell .plan-label { font-size: 0.7rem; color: #7b8494; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
    .plan-cell .plan-value { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1rem; }
    .plan-cell.entry .plan-value { color: var(--gemini); }
    .plan-cell.stop .plan-value { color: var(--red); }
    .plan-cell.target .plan-value { color: var(--green); }
    
    .divider-thin { border-top: 1px solid var(--border); margin: 14px 0; }
    
    /* Scanner Badges */
    .scan-badge { display: inline-block; padding: 2px 9px; border-radius: 6px; font-size: 0.68rem; font-weight: 600; margin-right: 4px; margin-bottom: 4px; }
    .badge-buy { background: rgba(16,185,129,0.12); color: var(--green); border: 1px solid rgba(16,185,129,0.3); }
    .badge-sell { background: rgba(239,68,68,0.12); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
    .badge-neutral { background: rgba(123,132,148,0.12); color: #7b8494; border: 1px solid var(--border); }
    .badge-vol { background: rgba(201,168,106,0.12); color: var(--verdict); border: 1px solid rgba(201,168,106,0.3); }

    .strategy-tag { display: inline-block; padding: 3px 11px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; margin-right: 5px; margin-bottom: 4px; }
    .tag-reversal-short { background: rgba(16,185,129,0.22); color: #6ee7a0; border: 1px solid rgba(16,185,129,0.55); }
    .tag-reversal-medium { background: rgba(16,185,129,0.14); color: var(--green); border: 1.5px solid rgba(16,185,129,0.45); }
    .tag-takeprofit-short { background: rgba(217,119,87,0.22); color: #f0a085; border: 1px solid rgba(217,119,87,0.55); }
    .tag-takeprofit-medium { background: rgba(217,119,87,0.14); color: var(--claude); border: 1.5px solid rgba(217,119,87,0.45); }
</style>
""", unsafe_allow_html=True)

# --- 🔄 1. ระบบ GENERATE TICKER TAPE ---
tape_json_list = [{"proName": sym, "title": sym.split(":")[-1]} for sym in TAPE_SYMBOLS]
tape_json_string = json.dumps(tape_json_list)

ticker_tape_html = f"""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js" async>
  {{
  "symbols": {tape_json_string},
  "showSymbolLogo": true,
  "isTransparent": true,
  "displayMode": "adaptive",
  "colorTheme": "dark",
  "locale": "th"
}}
  </script>
</div>
"""
components.html(ticker_tape_html, height=50)

# --- 🛠️ Helper Functions ---
def fetch_data_with_header(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response: 
        return response.read()

@st.cache_data(ttl=86400) # Cache 1 วันสำหรับรายชื่อหุ้น
def load_market_tickers(market):
    try:
        if market == "S&P 500":
            csv_bytes = fetch_data_with_header('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv')
            return pd.read_csv(io.BytesIO(csv_bytes))['Symbol'].tolist()
        elif "NASDAQ" in market:
            html_bytes = fetch_data_with_header('https://en.wikipedia.org/wiki/Nasdaq-100')
            tables = pd.read_html(io.StringIO(html_bytes.decode('utf-8')))
            for df in tables:
                if 'Ticker' in df.columns or 'Symbol' in df.columns:
                    return df['Ticker' if 'Ticker' in df.columns else 'Symbol'].tolist()
        elif market == "Penny Stocks (ต่ำกว่า $5)":
            return ["SNDL", "RIOT", "GRWG", "PLTR", "LCID", "SOFI", "NKLA", "DNA", "RIG", "MULN", "BBIG", "XELA"]
        elif market == "SET100 (หุ้นไทย)":
            tickers = [
                "ADVANC", "AOT", "AWC", "BANPU", "BBL", "BCP", "BDMS", "BEM", "BGRIM", "BH",
                "BTS", "CBG", "CENTEL", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EA", "EGCO",
                "GLOBAL", "GPSC", "GULF", "HMPRO", "INTUCH", "IRPC", "IVL", "KBANK", "KCE", "KTB",
                "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH",
                "SAWAD", "SCB", "SCC", "SCGP", "TASCO", "TIDLOR", "TISCO", "TOP", "TRUE", "TTB",
                "TU", "WHA", "AMATA", "AP", "BCH", "BJC", "BLA", "CHG", "CK", "CKP",
                "COM7", "DOHOME", "ERW", "ESSO", "GFPT", "GUNKUL", "ICHI", "JMART", "JMT", "KKP",
                "MAJOR", "MEGA", "MFC", "MOSHI", "ORI", "PLANB", "PR9", "PSL", "QH", "RS",
                "SABUY", "SAPPE", "SIRI", "SJWD", "SPALI", "SPRC", "STA", "STGT", "STECON", "SUPER",
                "TFG", "THANI", "THG", "TKN", "TOA", "TVO", "VGI", "WICE", "ITC", "SISB"
            ]
            return [t + ".BK" for t in tickers]
        elif market == "SET50 (หุ้นไทย)":
            tickers = ["ADVANC", "AOT", "AWC", "BANPU", "BBL", "BDMS", "BEM", "BGRIM", "BH", "BTS", "CBG", "CENTEL", "COM7", "CPALL", "CPF", "CPN", "CRC", "DELTA", "EA", "EGCO", "GLOBAL", "GPSC", "GULF", "HMPRO", "INTUCH", "IRPC", "IVL", "JMART", "JMT", "KBANK", "KCE", "KTB", "KTC", "LH", "MINT", "MTC", "OR", "OSP", "PTT", "PTTEP", "PTTGC", "RATCH", "SAWAD", "SCB", "SCC", "SCGP", "TIDLOR", "TISCO", "TOP", "TRUE", "TTB", "TU", "WHA"]
            return [t + ".BK" for t in tickers]
        elif market == "Crypto (Top Coins)":
            return ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD", "DOGE-USD", "TRX-USD", "DOT-USD"]
        elif market == "Crypto (Alt/Meme Coins)":
            return ["SHIB-USD", "PEPE-USD", "WIF-USD", "FLOKI-USD", "BONK-USD", "MATIC-USD", "LINK-USD", "UNI-USD", "LTC-USD", "BCH-USD"]
    except Exception as e: 
        st.sidebar.warning(f"Failed to fetch market data: {e}")
    return ['AAPL', 'MSFT', 'NVDA', 'AMZN'] # Fallback list

@st.cache_data(ttl=300)
def fetch_gainers_and_losers(asset_type="US"):
    if asset_type == "TH":
        sample_tickers = ["PTT.BK", "CPALL.BK", "BDMS.BK", "AOT.BK", "ADVANC.BK", "KBANK.BK", "SCB.BK", "GULF.BK", "DELTA.BK"]
    elif asset_type == "Crypto":
        sample_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD", "PEPE-USD"]
    else:
        sample_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "AMD", "PLTR"]
        
    try:
        data = yf.download(" ".join(sample_tickers), period="5d", interval="1d", group_by="ticker", progress=False)
        change_list = []
        for ticker in sample_tickers:
            if ticker in data.columns.get_level_values(0):
                df = data[ticker].dropna()
                if len(df) >= 2:
                    current_c = float(df['Close'].iloc[-1])
                    prev_c = float(df['Close'].iloc[-2])
                    p_change = ((current_c - prev_c) / prev_c) * 100
                    change_list.append({"Ticker": ticker, "Price": round(current_c, 2), "Change": p_change})
        
        df_changes = pd.DataFrame(change_list)
        if not df_changes.empty:
            top_gainers = df_changes.sort_values(by="Change", ascending=False).head(5)
            top_losers = df_changes.sort_values(by="Change", ascending=True).head(5)
            return top_gainers, top_losers
    except Exception as e:
        print(f"Error fetching gainers/losers: {e}")
        
    dummy_g = pd.DataFrame([{"Ticker": "NVDA" if asset_type=="US" else "PTT.BK" if asset_type=="TH" else "BTC-USD", "Price": 125.10, "Change": 4.82}])
    dummy_l = pd.DataFrame([{"Ticker": "TSLA" if asset_type=="US" else "AOT.BK" if asset_type=="TH" else "ETH-USD", "Price": 172.30, "Change": -3.50}])
    return dummy_g, dummy_l

def scan_market_batch(tickers_list, is_penny=False):
    """
    กวาดสแกนหุ้นทั้งหมดในลิสต์โดยคำนวณสัญญาณมาตรฐานและแท็กกลยุทธ์ Reversal / Take Profit ทั้งระยะสั้นและระยะกลาง
    """
    results = []
    scan_pool = tickers_list
    tickers_str = " ".join(scan_pool)
    try:
        # โหลดข้อมูลย้อนหลัง 3 เดือน เพื่อความแม่นยำและเสถียรภาพตัวชี้วัด (RSI14, RSI7, EMA20, EMA50)
        raw_df = yf.download(tickers_str, period="3mo", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
        for ticker in scan_pool:
            try:
                if isinstance(raw_df.columns, pd.MultiIndex):
                    if ticker in raw_df.columns.get_level_values(0):
                        df = raw_df[ticker].dropna().copy()
                    else:
                        continue
                else:
                    df = raw_df.dropna().copy()
                
                if df.empty or len(df) < 30: 
                    continue
                
                close = df['Close']
                volume = df['Volume']
                c = float(close.iloc[-1])
                
                if is_penny and c > 5.0:
                    continue
                
                # Indicator 20 วัน
                df['MA20'] = close.rolling(window=20).mean()
                df['STD'] = close.rolling(window=20).std()
                df['BB_Upper'] = df['MA20'] + (df['STD'] * 2)
                df['BB_Lower'] = df['MA20'] - (df['STD'] * 2)
                df['Vol_MA'] = volume.rolling(window=20).mean()
                
                # RSI 14 (Welles Wilder Smoothing)
                delta = close.diff()
                gains = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
                losses = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
                rs14 = gains / (losses + 1e-10)
                df['RSI'] = 100 - (100 / (1 + rs14))
                
                # RSI 7 (สำหรับจับโมเมนตัมระยะสั้นไวกว่า)
                gains7 = delta.where(delta > 0, 0).ewm(alpha=1/7, adjust=False).mean()
                losses7 = -delta.where(delta < 0, 0).ewm(alpha=1/7, adjust=False).mean()
                rs7 = gains7 / (losses7 + 1e-10)
                df['RSI7'] = 100 - (100 / (1 + rs7))
                
                # MACD (12, 26, 9)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                signal_line = macd_line.ewm(span=9, adjust=False).mean()
                df['MACD_Hist'] = macd_line - signal_line
                
                # EMA Trend (20 / 50 วัน)
                ema20 = close.ewm(span=20, adjust=False).mean()
                ema50 = close.ewm(span=50, adjust=False).mean()
                ema20_now, ema50_now = float(ema20.iloc[-1]), float(ema50.iloc[-1])
                ema20_prev, ema50_prev = float(ema20.iloc[-2]), float(ema50.iloc[-2])
                ema_bullish_cross = ema20_prev < ema50_prev and ema20_now > ema50_now
                ema_bearish_cross = ema20_prev > ema50_prev and ema20_now < ema50_now
                above_ema_trend = ema20_now > ema50_now

                u = float(df['BB_Upper'].iloc[-1])
                l = float(df['BB_Lower'].iloc[-1])
                v, vm = float(volume.iloc[-1]), float(df['Vol_MA'].iloc[-1])
                rsi14 = float(df['RSI'].iloc[-1])
                rsi14_prev = float(df['RSI'].iloc[-2])
                rsi7 = float(df['RSI7'].iloc[-1])
                rsi7_prev = float(df['RSI7'].iloc[-2])
                rsi7_prev3 = float(df['RSI7'].iloc[-4]) if len(df) > 4 else rsi7
                
                macd_now = float(df['MACD_Hist'].iloc[-1])
                macd_prev = float(df['MACD_Hist'].iloc[-2])
                macd_bullish_cross = macd_prev < 0 and macd_now > 0
                macd_bearish_cross = macd_prev > 0 and macd_now < 0
                macd_rising = macd_now > macd_prev
                
                prev_c = float(close.iloc[-2])
                change_pct = round((c - prev_c) / prev_c * 100, 2)
                vol_ratio = round(v / vm, 2) if vm > 0 else 1.0
                vol_spike = vol_ratio >= 1.5

                open_now = float(df["Open"].iloc[-1])
                candle_is_green = c > open_now
                candle_is_red = c < open_now
                
                high_20 = float(close.rolling(20).max().iloc[-1])
                pct_from_high20 = round((c - high_20) / high_20 * 100, 2)

                # ==========================================
                # 🏷️ สร้างกลยุทธ์พิเศษแท็ก (Reversal Buy / Take Profit)
                # ==========================================
                strategy_tags = []
                
                # กลับตัวระยะสั้น: RSI7 เคยลงโซน Oversold (<30) ใน 3 วันก่อนหน้า และวันนี้เขียวดีดพ้นโซน
                if (rsi7_prev3 < 30) and (rsi7 > rsi7_prev3) and (rsi7 < 50) and candle_is_green:
                    strategy_tags.append(("กลับตัวขึ้น (สั้น)", "reversal-short"))
                
                # กลับตัวระยะกลาง: EMA20 ตัดขึ้น EMA50 และ RSI14 เพิ่งฟื้นตัว (ยังไม่ Overbought) พร้อม MACD กำลังยกตัวขึ้น
                if ema_bullish_cross and (35 < rsi14 < 60) and macd_rising:
                    strategy_tags.append(("กลับตัวขึ้น (กลาง)", "reversal-medium"))
                    
                # Take Profit สั้น: RSI7 ขึ้น Overbought และวันนี้เกิดแท่งแดงกลับตัวลงมา พร้อมราคาอยู่ในโซนใกล้ High 20 วัน
                if (rsi7_prev3 > 70) and (rsi7 < rsi7_prev3) and candle_is_red and (pct_from_high20 > -5):
                    strategy_tags.append(("Take Profit (สั้น)", "takeprofit-short"))
                    
                # Take Profit กลาง: ราคาเกาะอยู่บนเทรนด์ขาขึ้นเหนือ EMA50 แต่ MACD Histogram ตัดลงตัดสัญญาณเริ่มแผ่วกำลัง
                if above_ema_trend and macd_bearish_cross and (c > ema50_now):
                    strategy_tags.append(("Take Profit (กลาง)", "takeprofit-medium"))

                # ==========================================
                # 🔧 สัญญาณหลักมาตรฐาน
                # ==========================================
                signals = []
                if rsi14 < 32:
                    signals.append(("RSI Oversold", "buy"))
                elif rsi14 > 68:
                    signals.append(("RSI Overbought", "sell"))
                
                if macd_bullish_cross:
                    signals.append(("MACD GoldCross", "buy"))
                elif macd_bearish_cross:
                    signals.append(("MACD DeathCross", "sell"))
                    
                if c > u and vol_spike:
                    signals.append(("BB Breakout บน", "buy"))
                elif c < l and vol_spike:
                    signals.append(("BB Breakout ล่าง", "sell"))
                    
                if vol_spike:
                    signals.append((f"Volume x{vol_ratio}", "vol"))

                # บันทึกข้อมูลเฉพาะตัวที่มีแท็กสัญญาณหรือแท็กกลยุทธ์
                if signals or strategy_tags:
                    results.append({
                        "ticker": ticker,
                        "price": round(c, 2),
                        "change_pct": change_pct,
                        "rsi": round(rsi14, 2),
                        "vol_ratio": vol_ratio,
                        "signals": signals,
                        "strategy_tags": strategy_tags,
                        "signal_count": len(signals) + len(strategy_tags)
                    })
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                continue
                
        # เรียงตามความเด่นของสัญญาณ
        results.sort(key=lambda x: x["signal_count"], reverse=True)
    except Exception as e:
        st.sidebar.error(f"เกิดข้อผิดพลาดในการสแกนตลาด: {e}")
    return results

# ==========================================
# 🔄 SYSTEMS RETRY ENGINE (Exponential Backoff)
# ==========================================
def call_api_with_backoff(api_call_fn, *args, **kwargs):
    delays = [1, 2, 4, 8, 16]
    for delay in delays:
        try:
            return api_call_fn(*args, **kwargs)
        except Exception as e:
            time.sleep(delay)
    try:
        return api_call_fn(*args, **kwargs)
    except Exception as e:
        raise RuntimeError(
            "⚠️ บริการ AI กำลังมีผู้ใช้งานหนาแน่นชั่วคราว (Error 503 / High Demand) "
            "ระบบพยายามออโต้รีไทร์ 5 ครั้งแล้วยังไม่สำเร็จ กรุณาเว้นระยะ 15 วินาทีแล้วกดปุ่มคำนวณอีกครั้งครับ"
        )

# ==========================================
# 🤖 STEP 1 — GEMINI: ความเห็นแรก
# ==========================================
@st.cache_data(ttl=3600)
def gemini_first_opinion(ticker, price_rounded, rsi_rounded, ma20_rounded, bb_u_rounded, bb_l_rounded, macd_hist):
    if not GEMINI_API_KEY:
        return None
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"""
    คุณคือนักวิเคราะห์เทรดดิ้งชั้นนำ วิเคราะห์ข้อมูลหุ้น {ticker} สำหรับการโต้ตอบแบบดีเบต:
    - ราคา: {price_rounded}
    - RSI(14): {rsi_rounded}
    - MA20: {ma20_rounded}
    - Bollinger Bands Upper: {bb_u_rounded}
    - Bollinger Bands Lower: {bb_l_rounded}
    - MACD Histogram: {macd_hist}
    
    ให้ความเห็นเบื้องต้นเชิงบวก/ลบ ตรวจสอบแรงส่งในตลาดและพฤติกรรมราคา เพื่อให้ Claude ตรวจสอบต่อไป

    เขียนทุกฟิลด์ด้วยภาษาไทยง่ายๆ ประโยคสั้น อ่านครั้งเดียวเข้าใจทันที เหมือนเล่าให้เพื่อนที่ไม่ได้เรียนการเงินมาฟัง
    ถ้าต้องพูดถึงศัพท์เทคนิค (เช่น RSI, MACD) ให้ขยายความสั้นๆในประโยคเดียวกันว่ามันแปลว่าอะไรในทางปฏิบัติ
    ห้ามเขียนแบบทางการแข็งๆหรือฟังดูเหมือนแปลจากภาษาอังกฤษ
    """
    def run_gemini():
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GeminiOpinion,
                temperature=0.4,
                system_instruction="ตอบเป็นภาษาไทยล้วน ใช้ภาษาพูดธรรมดาที่คนทั่วไปอ่านครั้งเดียวแล้วเข้าใจ ไม่ต้องอ่านซ้ำ ประโยคสั้น กระชับ ตรงประเด็น หลีกเลี่ยงศัพท์เทคนิคที่ไม่จำเป็น ถ้าต้องใช้ศัพท์เฉพาะให้ขยายความสั้นๆในประโยคเดียวกันว่าหมายถึงอะไร ห้ามตอบแบบทางการแข็งๆหรือฟังดูเหมือนแปลจากภาษาอังกฤษ"
            )
        )
        return json.loads(response.text)
    return call_api_with_backoff(run_gemini)

# ==========================================
# 🤖 STEP 2 — CLAUDE: ตรวจสอบและให้ข้อยุติสุดท้าย
# ==========================================
@st.cache_data(ttl=3600)
def claude_challenge_and_verdict(ticker, ind, gemini_opinion):
    """
    Claude ตรวจทานข้อคิดเห็นเชิงลึก ท้าทายข้อมูลดิบ และสรุปสัญญาณเทรด Final
    หากไม่มี ANTHROPIC_API_KEY หรือเกิดเหตุขัดข้อง จะทำการ Fallback คืนเป็นจำลอง Verdict อัตโนมัติด้วยโครงสร้างเดียวกัน
    """
    if not ANTHROPIC_API_KEY:
        # Fallback จำลอง Verdict อัตโนมัติจากโครงสร้างการคิดของ Gemini หากไม่มี Claude Key
        fallback_verdict = {
            "agrees_with_gemini": True,
            "final_signal": gemini_opinion["initial_signal"],
            "risk_level": "กลาง",
            "support_zone": f"{ind['ma20'] * 0.96:.2f}",
            "resistance_zone": f"{ind['ma20'] * 1.05:.2f}",
            "challenge_notes": f"ตรวจสอบโมเมนตัมของ {ticker} แล้วมีความสมเหตุสมผลตามโครงสร้าง RSI ระดับ {ind['rsi']}",
            "final_reasoning": f"มุมมองโดยรวมสอดคล้องกับปัจจัยแวดล้อมทาง Bollinger Bands แนะนำปฏิบัติตามกรอบราคาหลักอย่างระมัดระวัง",
            "action_summary": f"ดำเนินการเล่นในกรอบแคบตามข้อบ่งชี้ {gemini_opinion['initial_signal']} ในตลาดระยะสั้น",
            "entry_price": f"{ind['price']:.2f}",
            "stop_loss": f"{ind['price'] * 0.95:.2f}",
            "take_profit": f"{ind['price'] * 1.10:.2f}",
            "position_sizing_note": "คำแนะนำสัดส่วน: แบ่งสัดส่วนพอร์ตเพียง 5-10% เนื่องจากปัจจัยแปรปรวนในสภาวะตลาดชั่วคราว"
        }
        return fallback_verdict
        
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    prompt = f"""
    ตรวจสอบและตรวจทานพฤติกรรมกราฟราคารวมถึงความเห็นเบื้องต้นของ Gemini ของหุ้น {ticker}:
    - ข้อมูลเทคนิคัลดิบ: ราคาตลาด={ind['price']}, RSI={ind['rsi']}, MACD_Hist={ind['macd_hist']}, Bollinger=[{ind['bb_lower']}, {ind['bb_upper']}]
    - ความเห็นแรกจาก Gemini: Sentiment={gemini_opinion['market_sentiment']}, Signal={gemini_opinion['initial_signal']}, สังเกตเห็น={gemini_opinion['key_observation']}
    
    จงตรวจสอบ ท้าทายข้อผิดพลาด และให้คำตัดสินและแผน Action Plan สุดท้าย (BUY / SELL / HOLD) อย่างมีหลักการหนักแน่นแบบมือโปร
    แต่เขียนคำอธิบายทุกข้อด้วยภาษาไทยง่ายๆ สั้น กระชับ อ่านครั้งเดียวเข้าใจ เหมือนอธิบายให้คนในครอบครัวที่ไม่ได้เรียนการเงินมาฟัง
    หลีกเลี่ยงศัพท์การเงินที่ซับซ้อนเกินจำเป็น ถ้าต้องพูดถึงศัพท์เทคนิคให้ขยายความสั้นๆในประโยคเดียวกัน ห้ามเขียนแบบฟังดูเหมือนแปลจากภาษาอังกฤษ
    """
    
    def run_claude():
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system="คุณเป็นบอร์ดตัดสินใจเทรด ให้ตอบกลับเป็นรูปแบบ JSON เสมอเพื่อส่งคำตอบเข้าโครงสร้างระบบ ทุกข้อความในฟิลด์ต้องเป็นภาษาไทยง่ายๆที่อ่านแล้วเข้าใจทันที ไม่ใช้ประโยคที่ฟังดูเหมือนแปลจากภาษาอังกฤษ ไม่ใช้ศัพท์เทคนิคซ้อนศัพท์เทคนิคโดยไม่อธิบาย",
            messages=[{"role": "user", "content": prompt}],
            tools=[{
                "name": "submit_verdict",
                "description": "ส่งคำตัดสินเทรดดิ้งสุดท้ายหลังจากตรวจทานแล้ว",
                "input_schema": ClaudeVerdict.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "submit_verdict"}
        )
        for block in response.content:
            if block.type == "tool_use":
                return block.input
        raise ValueError("Claude didn't yield tool block.")
        
    return call_api_with_backoff(run_claude)

def run_ai_debate(ticker, ind):
    gemini_op = gemini_first_opinion(
        ticker, round(ind['price'], 2), round(ind['rsi'], 0), round(ind['ma20'], 2), 
        round(ind['bb_upper'], 2), round(ind['bb_lower'], 2), round(ind['macd_hist'], 4)
    )
    claude_v = claude_challenge_and_verdict(ticker, ind, gemini_op)
    return {"gemini": gemini_op, "claude": claude_v, "indicators": ind, "ticker": ticker}

@st.cache_data(ttl=900)
def get_ai_portfolio_analysis(portfolio_str):
    if not GEMINI_API_KEY:
        return None
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"วิเคราะห์ภาพรวมและการกระจายความเสี่ยงพอร์ตครอบครัวของคู่รัก:\n{portfolio_str}"
    def run_portfolio():
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PortfolioAnalysisResult,
                temperature=0.3,
                system_instruction="คุณคือที่ปรึกษาทางการเงิน แนะนำการจัดพอร์ตให้คู่รักปลอดภัย เติบโตมั่นคง ตอบเป็นภาษาไทยทั้งหมดอย่างอบอุ่นและเป็นมิตร ใช้ภาษาพูดง่ายๆ ประโยคสั้น ไม่ใช้ศัพท์การเงินซับซ้อน เหมือนคุยกับเพื่อนสนิท ไม่ใช่รายงานทางการ"
            )
        )
        return json.loads(response.text)
    return call_api_with_backoff(run_portfolio)

def ask_ai_copilot(query, ticker, price, tech_context, initial_analysis_str, chat_history):
    if not GEMINI_API_KEY:
        return "กรุณาใส่ API Key"
    client = genai.Client(api_key=GEMINI_API_KEY)
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-5:]])
    prompt = f"""
    บริบท: หุ้น {ticker} ราคา: {price}
    เทคนิคอลดิบ: {tech_context}
    คำตัดสิน AI ก่อนหน้า: {initial_analysis_str}
    ประวัติการสนทนา: {history_context}
    คำถามผู้ใช้ล่าสุด: "{query}"
    
    จงวิเคราะห์ตอบข้อสงสัยให้ชัดเจนและอิงสถิติการลงทุน ปฏิบัติตอบภาษาไทย 100% สุภาพและเข้าใจง่าย
    ตอบสั้น กระชับ เป็นกันเอง เหมือนเพื่อนนักลงทุนอธิบายให้เพื่อนฟัง หลีกเลี่ยงศัพท์เทคนิคพ่วงท้ายโดยไม่อธิบาย
    """
    def run_copilot():
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.5)
        )
        return response.text
    return call_api_with_backoff(run_copilot)

# ==========================================
# 📌 SIDEBAR CONTROLLER
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='color:#c9a86a;'>◆ PropFirmX Terminal</h2>", unsafe_allow_html=True)
    st.caption("AI Debate Terminal — Gemini × Claude")
    st.divider()
    
    st.write("🔍 **ค้นหาและปลดล็อกสินทรัพย์**")
    search_ticker = st.text_input("ระบุสัญลักษณ์ (เช่น AAPL, PTT.BK, BTC-USD):", value=st.session_state.active_ticker).upper()
    safe_search_ticker = re.sub(r'[^A-Z0-9.:\-]', '', search_ticker)
    
    if st.button("⚡ โหลดราคากราฟหลัก", use_container_width=True):
        st.session_state.active_ticker = safe_search_ticker
        st.session_state.ai_debate_result = None
        st.session_state.chat_history = []
        st.toast(f"อัปเดตสลับสินทรัพย์หลักเป็น {safe_search_ticker} แล้ว!", icon="✅")
        st.rerun()
        
    st.write("⏱️ **เลือกช่วงเวลาเทคนิคอล**")
    selected_tf = st.selectbox("เลือกช่วงกราฟ:", list(tf_mapping.keys()), index=list(tf_mapping.keys()).index(st.session_state.timeframe))
    if selected_tf != st.session_state.timeframe:
        st.session_state.timeframe = selected_tf
        st.rerun()
        
    st.divider()
    if not GEMINI_API_KEY:
        st.warning("⚠️ ยังไม่ได้ตั้งคีย์ GEMINI_API_KEY ใน Secrets")
    else:
        st.success("✅ Gemini Engine พร้อมใช้งาน")
    if not ANTHROPIC_API_KEY:
        st.info("ℹ️ ไม่พบ ANTHROPIC_API_KEY (ระบบจะรันดีเบตโดยใช้ออโต้ดีเบตโมเดลคู่ควบคู่จำลองทดแทน)")
    else:
        st.success("✅ Claude Engine พร้อมคู่ขนาน!")

# ==========================================
# 📌 MAIN WORKSPACE
# ==========================================
ticker = st.session_state.active_ticker

# --- 🎯 ตรวจสอบและแปลงสัญลักษณ์ให้เข้ากับ TradingView Widget ---
tv_symbol = ticker.upper().strip()
if tv_symbol.endswith(".BK"):
    tv_symbol = "SET:" + tv_symbol.replace(".BK", "")
elif "-USD" in tv_symbol:
    tv_symbol = "CRYPTO:" + tv_symbol.replace("-USD", "USD")
elif "-" in tv_symbol:
    tv_symbol = tv_symbol.replace("-", "")

tv_interval = current_tf['tv']
if tv_symbol.startswith("SET:") and tv_interval in ["1", "5", "15", "60", "240"]:
    tv_interval = "D"
    st.toast("⚠️ กราฟหุ้นไทย (SET) แสดงได้เฉพาะ Timeframe รายวัน (D) ขึ้นไป", icon="⏳")

@st.cache_data(ttl=60)
def get_main_ticker_data(t):
    try:
        quick_raw = yf.download(t, period="3mo", interval="1d", progress=False)
        if isinstance(quick_raw.columns, pd.MultiIndex):
            ticker_df = quick_raw.xs(t, axis=1, level=1) if t in quick_raw.columns.get_level_values(1) else quick_raw.xs(quick_raw.columns.get_level_values(0)[0], axis=1, level=0)
        else:
            ticker_df = quick_raw
            
        ticker_df = ticker_df.dropna()
        if ticker_df.empty:
            raise ValueError("No data returned")
            
        current_p = float(ticker_df['Close'].iloc[-1])
        p_close = float(ticker_df['Close'].iloc[-2]) if len(ticker_df) > 1 else current_p
        
        delta = ticker_df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0).ewm(alpha=1/14, adjust=False).mean()
        rsi_v = float((100 - (100 / (1 + (gain/(loss + 1e-10))))).iloc[-1]) 
        ma20_v = float(ticker_df['Close'].rolling(window=20).mean().iloc[-1]) if len(ticker_df) >= 20 else current_p
        std_v = float(ticker_df['Close'].rolling(window=20).std().iloc[-1]) if len(ticker_df) >= 20 else 0.0
        bb_upper_v = ma20_v + (std_v * 2)
        bb_lower_v = ma20_v - (std_v * 2)
        
        # MACD
        ema12 = ticker_df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = ticker_df['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd_line - signal_line).iloc[-1])
        
        # Volume
        volume = ticker_df['Volume']
        vol_avg = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.iloc[-1])
        vol_now = float(volume.iloc[-1])
        vol_ratio = round(vol_now / vol_avg, 2) if vol_avg > 0 else 1.0
        
        change_pct = round((current_p - p_close) / p_close * 100, 2)

        return current_p, change_pct, rsi_v, ma20_v, bb_upper_v, bb_lower_v, macd_hist, vol_ratio
    except Exception as e:
        print(f"Fetch main data error for {t}: {e}")
        st.toast(f"⚠️ ดึงข้อมูลราคาของ {t} ไม่สำเร็จ กำลังใช้ข้อมูลจำลองชั่วคราว", icon="⚠️")
        return 150.00, 0.0, 50.0, 150.0, 155.0, 145.0, 0.0, 1.0 # Dummy fallback

current_p, change_pct, rsi_v, ma20_v, bb_upper_v, bb_lower_v, macd_hist, vol_ratio = get_main_ticker_data(ticker)

# 1️⃣ MIDDLE SECTION: กราฟ Live TradingView และ สรุปราคาสด
col_left_main, col_right_panel = st.columns([3, 1])

with col_left_main:
    st.markdown(f"#### 📈 Live Market Technical Chart: <span style='color:#38bdf8;'>{ticker}</span> ({st.session_state.timeframe})", unsafe_allow_html=True)
    tradingview_html = f"""
    <div class="tradingview-widget-container" style="height:380px;width:100%">
      <div id="tradingview_chart" style="height:100%;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_symbol}",
        "interval": "{tv_interval}",
        "timezone": "Asia/Bangkok",
        "theme": "dark",
        "style": "1",
        "locale": "th",
        "enable_publishing": false,
        "hide_side_toolbar": false,
        "studies": ["RSI@tv-basicstudies", "MASimple@tv-basicstudies"],
        "container_id": "tradingview_chart", "backgroundColor": "#0a0c10", "gridColor": "#232834"
      }});
      </script>
    </div>
    """
    components.html(tradingview_html, height=390)

with col_right_panel:
    st.markdown("#### 🔥 ความร้อนแรงรายวัน")
    asset_select = st.selectbox("เลือกประเภทสินทรัพย์หลัก", ["US Stocks", "Thai Stocks", "Cryptocurrency"])
    
    asset_map = {"US Stocks": "US", "Thai Stocks": "TH", "Cryptocurrency": "Crypto"}
    gainers_df, losers_df = fetch_gainers_and_losers(asset_map[asset_select])
    
    tab_g, tab_l = st.tabs(["🚀 Gainers", "📉 Losers"])
    with tab_g:
        for idx, row in gainers_df.iterrows():
            if st.button(f"🟢 {row['Ticker']}  |  {row['Change']:+.2f}%", key=f"g_{row['Ticker']}_{asset_select}", use_container_width=True):
                st.session_state.active_ticker = row['Ticker']
                st.session_state.ai_debate_result = None
                st.session_state.chat_history = []
                st.rerun()
    with tab_l:
        for idx, row in losers_df.iterrows():
            if st.button(f"🔴 {row['Ticker']}  |  {row['Change']:+.2f}%", key=f"l_{row['Ticker']}_{asset_select}", use_container_width=True):
                st.session_state.active_ticker = row['Ticker']
                st.session_state.ai_debate_result = None
                st.session_state.chat_history = []
                st.rerun()

st.divider()


# 2️⃣ BOTTOM SECTION: แท็บหน้าต่างแยกจัดการพอร์ต / สแกนเนอร์ และระบบ AI DEBATE
st.markdown("### 💼 ระบบจัดการพอร์ต (แชร์ร่วมกัน) และสแกนเนอร์สมองกล")

tab_us_class, tab_th_class, tab_crypto_class, tab_journal_class = st.tabs([
    "🇺🇸 หุ้นอเมริกา (US Stocks)", "🇹🇭 หุ้นไทย (Thai Stocks)",
    "🪙 คริปโทเคอร์เรนซี (Cryptocurrency)", "📓 Trade Journal & Win Rate"
])

def render_portfolio_and_scanner_area(portfolio_key, scanner_market_list, default_scanned_df, is_penny=False, postfix=""):
    col_p, col_s, col_a = st.columns([1.5, 1.0, 1.1])
    
    with col_p:
        st.markdown(f"##### 📋 ตารางพอร์ตครอบครัว ({postfix})")
        
        edited_port = st.data_editor(
            st.session_state[portfolio_key],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key=f"editor_{portfolio_key}"
        )
        
        if not edited_port.equals(st.session_state[portfolio_key]):
            st.session_state[portfolio_key] = edited_port
            save_portfolio(portfolio_key, edited_port)
            st.toast(f"บันทึกข้อมูลพอร์ต {postfix} ลงฐานข้อมูลแล้ว", icon="💾")
        
        if st.button("🔄 อัปเดตราคาสินทรัพย์และ P&L", key=f"btn_calc_{portfolio_key}", use_container_width=True):
            with st.spinner("กำลังเชื่อมต่อเซิร์ฟเวอร์ตลาด..."):
                all_tk = edited_port["Ticker"].dropna().tolist()
                if all_tk:
                    try:
                        p_raw = yf.download(" ".join(all_tk), period="1d", interval="1m", progress=False)
                        updated_rows = []
                        t_cost, t_value = 0.0, 0.0
                        
                        for idx, row in edited_port.iterrows():
                            try:
                                tk = row["Ticker"]
                                shares = float(row["Shares"])
                                avg_cost = float(row["AvgCost"])
                                
                                if len(all_tk) == 1:
                                    current_m_p = float(p_raw['Close'].iloc[-1])
                                else:
                                    if tk in p_raw['Close'].columns:
                                        current_m_p = float(p_raw['Close'][tk].dropna().iloc[-1])
                                    else:
                                        current_m_p = avg_cost
                                        
                                val = shares * current_m_p
                                cost = shares * avg_cost
                                pnl = val - cost
                                pnl_pct = (pnl / cost) * 100 if cost > 0 else 0.0
                                
                                t_cost += cost
                                t_value += val
                                
                                updated_rows.append({
                                    "สัญลักษณ์": tk,
                                    "ถือครอง": f"{shares:,.4f}",
                                    "ต้นทุน": f"{avg_cost:,.2f}",
                                    "ราคาตลาด": f"{current_m_p:,.2f}",
                                    "มูลค่า": f"{val:,.2f}",
                                    "P&L": f"{pnl:+.2f} ({pnl_pct:+.2f}%)"
                                })
                            except Exception as e:
                                print(f"Error calculating PNL for {row.get('Ticker', 'Unknown')}: {e}")
                                continue
                            
                        st.divider()
                        total_pnl = t_value - t_cost
                        total_pnl_pct = (total_pnl / t_cost) * 100 if t_cost > 0 else 0.0
                        total_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
                        
                        st.markdown(f"💰 **ต้นทุนรวม:** {t_cost:,.2f} | **มูลค่าปัจจุบัน:** {t_value:,.2f}")
                        st.markdown(f"🔥 **กำไรสุทธิ:** <span style='color:{total_color}; font-size:1.15rem; font-weight:bold;'>{total_pnl:+,.2f} ({total_pnl_pct:+.2f}%)</span>", unsafe_allow_html=True)
                        if updated_rows:
                            st.dataframe(pd.DataFrame(updated_rows), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.error(f"การดึงราคา Real-time ล้มเหลว: {e}")
                        
    with col_s:
        st.markdown("##### 🔍 ตัวเลือกสแกนตลาดสมองกล")
        scanner_type = st.selectbox("เลือกดัชนีคัดกรองเฉพาะด้าน:", scanner_market_list, key=f"select_scan_{portfolio_key}")
        
        if st.button("🚀 ยิงพิกัดสแกนตรวจจับสัญญาณด่วน", key=f"btn_scan_{portfolio_key}", use_container_width=True, type="primary"):
            st.session_state.scan_results = []
            with st.spinner("สมองกลกำลังกวาดดัชนีชี้วัดเทคนิคอลและจับแท็กกลยุทธ์หุ้นทั้งหมด (อาจใช้เวลาสักครู่)..."):
                t_list = load_market_tickers(scanner_type)
                results = scan_market_batch(t_list, is_penny=(scanner_type == "Penny Stocks (ต่ำกว่า $5)"))
                st.session_state.scan_results = results
                st.toast(f"อัปเดตระบบตรวจสอบสัญญาณสแกนเนอร์สำเร็จ! พบสัญญาณ {len(results)} ตัว", icon="🔥")
                st.rerun()
                
        st.write("📋 **สัญญาณด่วนและแท็กกลยุทธ์ที่ตรวจพบล่าสุด:**")
        df_s = pd.DataFrame(st.session_state.scan_results) if st.session_state.scan_results else default_scanned_df
            
        if not df_s.empty:
            # แปลงและจัดรูปแบบเพื่อให้พ่นข้อมูลพร้อม Tag สวยงามบนตาราง
            tag_class_map = {
                "reversal-short": "tag-reversal-short",
                "reversal-medium": "tag-reversal-medium",
                "takeprofit-short": "tag-takeprofit-short",
                "takeprofit-medium": "tag-takeprofit-medium",
            }
            tag_icon_map = {
                "reversal-short": "🟢", "reversal-medium": "🟢",
                "takeprofit-short": "🟠", "takeprofit-medium": "🟠",
            }
            
            # วนซ้ำพ่นรายการเพื่อให้ดูง่ายพร้อมปุ่มคลิกวิเคราะห์
            for idx, r in df_s.iterrows():
                t_ticker = r.get("ticker", r.get("Ticker", "Unknown"))
                t_price = r.get("price", r.get("Price", 0.0))
                t_change = r.get("change_pct", 0.0)
                
                # ประกอบ HTML ย่อยของสัญญาณ
                strategy_html = ""
                if "strategy_tags" in r and isinstance(r["strategy_tags"], list):
                    for label, kind in r["strategy_tags"]:
                        cls = tag_class_map.get(kind, "tag-reversal-short")
                        icon = tag_icon_map.get(kind, "")
                        strategy_html += f'<span class="strategy-tag {cls}" style="padding: 1px 6px; font-size: 0.65rem; border-radius: 4px; font-weight: bold; margin-right: 4px;">{icon} {label}</span>'
                
                badge_html = ""
                if "signals" in r and isinstance(r["signals"], list):
                    for label, kind in r["signals"]:
                        cls = {"buy": "badge-buy", "sell": "badge-sell", "neutral": "badge-neutral", "vol": "badge-vol"}.get(kind, "badge-neutral")
                        badge_html += f'<span class="scan-badge {cls}" style="padding: 1px 6px; font-size: 0.65rem; border-radius: 4px; font-weight: bold; margin-right: 4px;">{label}</span>'
                elif "Signal" in r:
                    # Fallback ของเดิม
                    lbl = r["Signal"]
                    cls = "badge-buy" if "BUY" in lbl else "badge-sell" if "RSI Over" in lbl else "badge-neutral"
                    badge_html += f'<span class="scan-badge {cls}" style="padding: 1px 6px; font-size: 0.65rem; border-radius: 4px; font-weight: bold; margin-right: 4px;">{lbl}</span>'
                
                c1, c2 = st.columns([2.5, 1.0])
                with c1:
                    change_color = "var(--green)" if t_change >= 0 else "var(--red)"
                    st.markdown(f"""
                    <div style="line-height:1.2;">
                        <strong>{t_ticker}</strong> <span style="color:{change_color}; font-size:0.8rem;">${t_price} ({t_change:+.2f}%)</span>
                        <div style="margin-top: 4px;">{strategy_html}{badge_html}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c2:
                    if st.button("วิเคราะห์ →", key=f"btn_sel_{t_ticker}_{portfolio_key}", use_container_width=True):
                        st.session_state.active_ticker = t_ticker
                        st.session_state.ai_debate_result = None
                        st.session_state.chat_history = []
                        st.rerun()
                st.markdown("<div style='border-top:1px solid #1e293b; margin:6px 0;'></div>", unsafe_allow_html=True)
        else:
            st.info("💡 ไม่พบสัญญาณตลาด แนะนำกวาดสแกนด้วยตนเอง")
                
    with col_a:
        st.markdown("##### 🧠 AI Debate & Portfolio Expert")
        tab_sub_st, tab_sub_port = st.tabs(["วิเคราะห์โต้ตอบ Debate", "พอร์ตครอบครัว"])
        
        with tab_sub_st:
            st.markdown("""
            <div class="vs-banner" style="margin-bottom:10px;">
                <div class="vs-side vs-gemini" style="padding: 6px 12px;">
                    <div class="vs-label" style="font-size:0.75rem;">◷ GEMINI</div>
                </div>
                <div class="vs-divider" style="font-size:0.75rem;">⟷</div>
                <div class="vs-side vs-claude" style="padding: 6px 12px;">
                    <div class="vs-label" style="font-size:0.75rem;">◈ CLAUDE</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("▶ เริ่มกระบวนการ AI Debate วิจัยด่วน", key=f"btn_ai_st_{portfolio_key}", use_container_width=True, type="primary"):
                if not GEMINI_API_KEY:
                    st.error("กรุณากรอก GEMINI_API_KEY ใน secrets")
                else:
                    with st.spinner("AI กำลังโต้วาทะประมวลผล (Gemini กำลังร่างความเห็นแรก)..."):
                        try:
                            # คำนวณรวบรวม Indicators ดิบ
                            ind_data = {
                                "price": current_p,
                                "change_pct": change_pct,
                                "rsi": rsi_v,
                                "ma20": ma20_v,
                                "bb_upper": bb_upper_v,
                                "bb_lower": bb_lower_v,
                                "macd_hist": macd_hist,
                                "vol_ratio": vol_ratio
                            }
                            st.session_state.ai_debate_result = run_ai_debate(ticker, ind_data)
                            st.session_state.chat_history = []
                            journal.log_verdict(ticker, st.session_state.ai_debate_result["claude"])  # 📓 บันทึกลง Trade Journal
                        except Exception as e:
                            st.error(str(e))
                            
            if st.session_state.ai_debate_result:
                res_deb = st.session_state.ai_debate_result
                g = res_deb["gemini"]
                c = res_deb["claude"]
                
                # แถบผลสรุปหลัก
                signal_upper = c.get("final_signal", "HOLD").upper()
                banner_class = "buy" if signal_upper == "BUY" else "sell" if signal_upper == "SELL" else "hold"
                icon = "🟢" if banner_class == "buy" else "🔴" if banner_class == "sell" else "🟡"
                action_label = {"buy": "ซื้อ (BUY)", "sell": "ขาย (SELL)", "hold": "ถือครอง (HOLD)"}[banner_class]

                st.markdown(f"""
                <div style="background: linear-gradient(90deg, rgba(201,168,106,0.15), rgba(201,168,106,0.02)); border: 1.5px solid var(--verdict); border-radius: 8px; padding: 10px; margin-bottom:12px;">
                    <div style="font-weight:bold; color:var(--verdict); font-size:0.9rem;">{icon} คำแนะนำสุดท้าย: {action_label}</div>
                    <div style="font-size:0.8rem; color:#cbd5e1; margin-top:4px;">{c.get('action_summary', '')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # สรุปดีเบตจำลองความเห็นย่อย
                with st.expander("🔍 ดูบทวิพากษ์และข้อท้าทาย (Gemini vs Claude)"):
                    agree_class = "pill-agree" if c.get("agrees_with_gemini") else "pill-disagree"
                    agree_text = "เห็นด้วย" if c.get("agrees_with_gemini") else "ท้าทายแย้งข้อคิดเห็น"
                    
                    st.markdown(f"""
                    <div style="font-size:0.8rem;">
                        <p><strong style="color:var(--gemini);">Gemini Opinion:</strong> {g.get('market_sentiment', '-')}</p>
                        <p><strong style="color:var(--claude);">Claude Challenge:</strong> {c.get('challenge_notes', '-')}</p>
                        <span class="pill {agree_class}">{agree_text}</span>
                        <span class="pill">ความเสี่ยง: {c.get('risk_level', '-')}</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                # พิกัด Verdict Action Plan
                signal_class = {"BUY": "signal-buy", "SELL": "signal-sell", "HOLD": "signal-hold"}.get(signal_upper, "signal-hold")
                st.markdown(f"""
                <div class="verdict-box" style="padding:12px; margin-top:8px;">
                    <div style="font-size:0.75rem; color:var(--verdict); font-weight:bold;">🛡 {c.get('support_zone', '-')} &nbsp;&nbsp;|&nbsp;&nbsp; 🚀 {c.get('resistance_zone', '-')}</div>
                    <div class="plan-grid" style="margin-top:6px; gap:8px;">
                        <div class="plan-cell entry" style="padding:6px 8px;"><div class="plan-label" style="font-size:0.55rem;">จุดเข้าซื้อ</div><div class="plan-value" style="font-size:0.8rem;">{c.get('entry_price', '-')}</div></div>
                        <div class="plan-cell stop" style="padding:6px 8px;"><div class="plan-label" style="font-size:0.55rem;">Stop Loss</div><div class="plan-value" style="font-size:0.8rem;">{c.get('stop_loss', '-')}</div></div>
                        <div class="plan-cell target" style="padding:6px 8px;"><div class="plan-label" style="font-size:0.55rem;">Take Profit</div><div class="plan-value" style="font-size:0.8rem;">{c.get('take_profit', '-')}</div></div>
                    </div>
                    <div style="font-size:0.75rem; color:#94a3b8; margin-top:8px; line-height:1.3;"><strong>เหตุผลสรุป:</strong> {c.get('final_reasoning', '-')}</div>
                    <div style="font-size:0.7rem; color:#7b8494; margin-top:4px;">💼 {c.get('position_sizing_note', '-')}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # ส่วนแชทสืบถามเพิ่มเติมกับ AI Copilot
                st.divider()
                st.write("💬 **ถาม-ตอบโต้ตอบ AI Copilot:**")
                for chat in st.session_state.chat_history:
                    style_class = "chat-bubble-user" if chat["role"] == "user" else "chat-bubble-ai"
                    sender = "คุณ" if chat["role"] == "user" else "AI Copilot"
                    st.markdown(f"""<div class="{style_class}"><strong>{sender}:</strong><br>{chat['content']}</div>""", unsafe_allow_html=True)
                
                with st.form(key=f"chat_form_{portfolio_key}", clear_on_submit=True):
                    user_query = st.text_input("ปรึกษาโมเมนตัมเพิ่มเติม:", key=f"input_query_{portfolio_key}")
                    if st.form_submit_button("ส่งคำถาม") and user_query:
                        tech_c = f"RSI={rsi_v:.1f}, MACD_Hist={macd_hist:.4f}"
                        ai_orig = f"Verdict={c.get('final_signal', '')}, Entry={c.get('entry_price', '')}, TP={c.get('take_profit', '')}"
                        with st.spinner("AI กำลังวิเคราะห์..."):
                            copilot_ans = ask_ai_copilot(user_query, ticker, current_p, tech_c, ai_orig, st.session_state.chat_history)
                        st.session_state.chat_history.extend([
                            {"role": "user", "content": user_query},
                            {"role": "copilot", "content": copilot_ans}
                        ])
                        st.rerun()
            else:
                st.info("💡 กดปุ่มด้านบนเพื่อประมวลผลวิเคราะห์จุดเทรดด้วยระบบ AI Debate")
                
        with tab_sub_port:
            if st.button("🧠 ประเมินความเสี่ยงพอร์ตองค์รวม", key=f"btn_ai_port_{portfolio_key}", use_container_width=True, type="primary"):
                if not GEMINI_API_KEY:
                    st.error("กรุณากรอก API_KEY")
                else:
                    with st.spinner("AI กำลังประเมินการถ่วงน้ำหนักจัดพอร์ตร่วมกัน..."):
                        try:
                            port_str = edited_port.to_string(index=False)
                            st.session_state.ai_portfolio_analysis = get_ai_portfolio_analysis(port_str)
                        except Exception as e:
                            st.error(str(e))
                            
            if st.session_state.ai_portfolio_analysis:
                p_res = st.session_state.ai_portfolio_analysis
                st.markdown(f"**การกระจายสินทรัพย์:** `{p_res.get('diversification_score', '-')}`")
                st.markdown(f"**⚠️ ความเสี่ยงสูงสุด:** {p_res.get('highest_risk_asset', '-')}")
                st.markdown(f"**📝 สรุปภาพรวมพอร์ต:** {p_res.get('portfolio_pnl_summary', '-')}")
                st.markdown(f"""<div style="background-color: #1e1b29; border-left: 4px solid #a855f7; padding: 10px; border-radius: 6px; font-size: 0.85rem; color: #cbd5e1;"><strong>💡 แนะนำอนาคตพอร์ตครอบครัว:</strong><br>{p_res.get('strategic_advice', '-')}</div>""", unsafe_allow_html=True)
            else:
                st.info("💡 คลิกเพื่อประเมินความเสี่ยงพอร์ตร่วมกันของแฟน")
    
# ==========================================
# 📓 TRADE JOURNAL & WIN-RATE TAB
# ==========================================
def render_trade_journal_tab():
    st.markdown("##### 📓 Trade Journal — ติดตามผลจริงของคำตัดสิน AI ย้อนหลัง")
    st.caption("ทุกครั้งที่กด '▶ เริ่มกระบวนการ AI Debate' ระบบจะบันทึก verdict ของ Claude ไว้ที่นี่อัตโนมัติ "
               "แล้วกดปุ่มด้านล่างเพื่อเช็คกับราคาจริงว่าผลลัพธ์เป็นยังไง")

    col_btn, col_note = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 อัปเดตผลย้อนหลัง (เช็คราคาจริง)", use_container_width=True, type="primary"):
            with st.spinner("กำลังเช็คราคาย้อนหลังเทียบกับ Take Profit / Stop Loss ของแต่ละรายการ..."):
                updated, errors = journal.settle_journal_entries()
            if errors:
                st.warning(f"อัปเดตสำเร็จ {updated} รายการ มีบางตัวดึงราคาไม่สำเร็จ {errors} รายการ (ลองกดใหม่ได้)")
            else:
                st.toast(f"อัปเดตผลสำเร็จ {updated} รายการ", icon="✅")
            st.rerun()
    with col_note:
        st.caption("⚠️ การเช็คนี้ต้องดึงราคาย้อนหลังของทุก ticker ที่ยังไม่ปิดสถานะ อาจใช้เวลาสักครู่ถ้ามีรายการเยอะ")

    stats = journal.get_win_rate_stats()

    st.divider()
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("บันทึกทั้งหมด", stats["total"])
    s2.metric("ยังเปิดอยู่", stats["open"])
    s3.metric("ชนะ", stats["win"])
    s4.metric("แพ้", stats["loss"])
    win_rate_display = f"{stats['win_rate_pct']}%" if stats["win_rate_pct"] is not None else "ยังไม่มีข้อมูล"
    s5.metric("Win Rate", win_rate_display)

    if stats["win_rate_pct"] is None:
        st.info("💡 ยังไม่มีรายการที่ปิดสถานะแพ้/ชนะ ลองใช้งาน AI Debate สักพักแล้วกลับมากดอัปเดตผลย้อนหลังอีกครั้ง")

    if not stats["by_signal"].empty:
        st.write("**Win Rate แยกตามประเภทสัญญาณ:**")
        for _, row in stats["by_signal"].iterrows():
            st.markdown(f"- **{row['final_signal']}**: {row['win_rate_pct']}% (จากที่ตัดสินผลแล้ว {row['n']} ครั้ง)")

    st.divider()
    st.write("**📋 รายการล่าสุด:**")
    recent = journal.get_recent_entries(limit=30)
    if recent.empty:
        st.info("💡 ยังไม่มีรายการในสมุดบันทึก — ไปลองกด AI Debate ที่หุ้นตัวไหนก็ได้ดูครับ")
    else:
        outcome_label = {
            "win": "✅ ชนะ", "loss": "❌ แพ้", "pending": "⏳ รอผล",
            "expired": "⌛ หมดอายุ", "not_applicable": "➖ ไม่นับ (HOLD)"
        }
        display_df = recent.copy()
        display_df["ผลลัพธ์"] = display_df["outcome"].map(outcome_label).fillna(display_df["outcome"])
        display_df = display_df[[
            "ticker", "created_at", "final_signal", "entry_price",
            "stop_loss", "take_profit", "ผลลัพธ์"
        ]].rename(columns={
            "ticker": "หุ้น", "created_at": "วันที่บันทึก", "final_signal": "สัญญาณ",
            "entry_price": "เข้าซื้อ", "stop_loss": "Stop Loss", "take_profit": "Take Profit"
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ดำเนินการกระจายหน้าตามแท็บคลาสต่างๆ
with tab_us_class:
    render_portfolio_and_scanner_area(
        "port_us", ["NASDAQ 100", "S&P 500", "Penny Stocks (ต่ำกว่า $5)"],
        pd.DataFrame([
            {"ticker": "AAPL", "price": 180.25, "change_pct": 1.2, "strategy_tags": [("กลับตัวขึ้น (กลาง)", "reversal-medium")], "signals": [("MACD GoldCross", "buy")]}
        ]), postfix="US Stocks"
    )

with tab_th_class:
    render_portfolio_and_scanner_area(
        "port_th", ["SET100 (หุ้นไทย)", "SET50 (หุ้นไทย)"],
        pd.DataFrame([
            {"ticker": "PTT.BK", "price": 32.50, "change_pct": -0.8, "strategy_tags": [("กลับตัวขึ้น (สั้น)", "reversal-short")], "signals": [("RSI Oversold", "buy")]}
        ]), postfix="Thai Stocks"
    )

with tab_crypto_class:
    render_portfolio_and_scanner_area(
        "port_crypto", ["Crypto (Top Coins)", "Crypto (Alt/Meme Coins)"],
        pd.DataFrame([
            {"ticker": "BTC-USD", "price": 61500.00, "change_pct": 2.5, "strategy_tags": [("กลับตัวขึ้น (กลาง)", "reversal-medium")], "signals": [("BB Breakout บน", "buy")]}
        ]), postfix="Crypto"
    )

with tab_journal_class:
    render_trade_journal_tab()

st.markdown("<div style='text-align:center; color:#7b8494; font-size:0.75rem; margin-top:24px;'>"
            "ข้อมูลนี้ถูกประมวลผลด้วยโมเดลวิเคราะห์เชิงกลยุทธ์ Gemini 3.1 flash lite และ Claude 3.5 เพื่อใช้เพื่อการศึกษาเทคโนโลยีการเงินเท่านั้น"
            "</div>", unsafe_allow_html=True)