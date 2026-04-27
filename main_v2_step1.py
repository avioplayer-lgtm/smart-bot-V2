import os
import time
import uuid
import json
import sqlite3
import logging
import threading
import requests
import pandas as pd
import pytz
from datetime import datetime, date
from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────
# ENV CONFIG
# ─────────────────────────────────────────
BOT_TOKEN = os.environ.get("BOT_TOKEN")
CHAT_ID = os.environ.get("CHAT_ID")
DHAN_CLIENT_ID = (os.environ.get("DHAN_CLIENT_ID") or "").strip()
DHAN_ACCESS_TOKEN = (os.environ.get("DHAN_ACCESS_TOKEN") or "").strip()

if not BOT_TOKEN or not CHAT_ID:
    raise ValueError("BOT_TOKEN and CHAT_ID must be set as environment variables.")

CAPITAL = float(os.environ.get("CAPITAL", 30000))
RISK_PCT = 0.005
MAX_DAILY_LOSS = CAPITAL * 0.02
MIN_CONFIDENCE = 6
IST = pytz.timezone("Asia/Kolkata")

SYMBOLS = {
    "NIFTY": {"interval": 50, "lot": 65, "expiry_day": 3, "dhan_scrip": "13", "ws_scrip": 13},
    "BANKNIFTY": {"interval": 100, "lot": 30, "expiry_day": 3, "dhan_scrip": "25", "ws_scrip": 25},
}

STRATEGY_RANK = {"TRENDING": 3, "VOLATILE": 2, "SIDEWAYS": 1}

DHAN_HEADERS = {
    "access-token": DHAN_ACCESS_TOKEN,
    "client-id": DHAN_CLIENT_ID,
    "Content-Type": "application/json",
}

# ─────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
log = logging.getLogger("DhanSignalBot")

# ─────────────────────────────────────────
# SQLITE PERSISTENCE
# ─────────────────────────────────────────
conn = sqlite3.connect("state.db", check_same_thread=False)
conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
conn.commit()
log.info("SQLite state.db initialised")

_lock = threading.Lock()
regime_state = {name: {"last": None, "count": 0} for name in SYMBOLS}

state = {
    "active_trade": None,
    "pending_signals": {},
    "daily_loss": 0.0,
    "current_day": None,
    "rules_sent": {"open": False, "mid": False, "close": False},
    "last_heartbeat_hour": -1,
    "holiday_sent": False,
    "paused": False,
    "token_reminder_sent": False,
    "trade_confirmed": False,
}

def get_st(key):
    with _lock:
        return state[key]

def set_st(key, val):
    with _lock:
        state[key] = val
    _persist_state()

def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return str(v)

def _persist_state():
    try:
        payload = _jsonable(state)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                ("state", json.dumps(payload))
            )
    except Exception as e:
        log.error(f"_persist_state error: {e}")

def _load_state():
    try:
        row = conn.execute("SELECT v FROM kv WHERE k='state'").fetchone()
        if not row:
            log.info("No saved state found - starting fresh")
            return
        saved = json.loads(row[0])
        with _lock:
            for k, v in saved.items():
                if k in state:
                    state[k] = v
        log.info("State restored from SQLite")
    except Exception as e:
        log.error(f"_load_state error: {e}")

# ─────────────────────────────────────────
# DHAN CANDLE DATA
# ─────────────────────────────────────────
def get_dhan_candles(name):
    scrip_id = SYMBOLS[name]["dhan_scrip"]
    today = datetime.now(IST).date().isoformat()
    try:
        resp = requests.post(
            "https://api.dhan.co/v2/charts/intraday",
            headers=DHAN_HEADERS,
            json={
                "securityId": scrip_id,
                "exchangeSegment": "IDX_I",
                "instrument": "INDEX",
                "interval": "5",
                "fromDate": today,
                "toDate": today,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("open"):
            log.warning(f"{name}: Empty candle data from Dhan")
            return None
        df = pd.DataFrame({
            "Open": data["open"],
            "High": data["high"],
            "Low": data["low"],
            "Close": data["close"],
            "Volume": data["volume"],
        }, index=pd.to_datetime(data["timestamp"], unit="s", utc=True).tz_convert(IST))
        df = df.dropna()
        if len(df) < 20:
            log.warning(f"{name}: Too few candles ({len(df)})")
            return None
        log.info(f"{name}: Got {len(df)} candles from Dhan")
        return df
    except requests.exceptions.HTTPError as e:
        try:
            log.error(f"{name} candle HTTP {e.response.status_code}: {e.response.text}")
        except Exception:
            log.error(f"{name} candle HTTP error: {e}")
    except Exception as e:
        log.error(f"{name} get_dhan_candles error: {e}")
    return None

# ─────────────────────────────────────────
# DHAN INDEX LTP
# ─────────────────────────────────────────
def get_dhan_ltp(name):
    scrip_id = SYMBOLS[name]["ws_scrip"]
    try:
        resp = requests.post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            headers=DHAN_HEADERS,
            json={"NSE_INDEX": [scrip_id]},
            timeout=5,
        )
        resp.raise_for_status()
        data = resp.json()
        ltp = data.get("data", {}).get("NSE_INDEX", {}).get(str(scrip_id), {}).get("last_price")
        if ltp:
            return float(ltp)
        log.warning(f"{name}: LTP missing in response - {data}")
    except requests.exceptions.HTTPError as e:
        try:
            log.error(f"{name} LTP HTTP {e.response.status_code}: {e.response.text}")
        except Exception:
            log.error(f"{name} LTP HTTP error: {e}")
    except Exception as e:
        log.error(f"{name} get_dhan_ltp error: {e}")
    return None

# ─────────────────────────────────────────
# DHAN OPTION CHAIN
# ─────────────────────────────────────────
_expiry_cache = {}

def get_next_expiry(scrip_id):
    today = datetime.now(IST).date().isoformat()
    if scrip_id in _expiry_cache:
        cached_date, cached_expiry = _expiry_cache[scrip_id]
        if cached_date == today:
            return cached_expiry
    try:
        resp = requests.post(
            "https://api.dhan.co/v2/optionchain/expirylist",
            headers=DHAN_HEADERS,
            json={"UnderlyingScrip": int(scrip_id), "UnderlyingSeg": "IDX_I"},
            timeout=10,
        )
        resp.raise_for_status()
        expiries = resp.json().get("data", [])
        today_date = datetime.now(IST).date()
        for exp in sorted(expiries):
            if exp >= today_date.isoformat():
                _expiry_cache[scrip_id] = (today, exp)
                log.info(f"Dhan expiry for scrip {scrip_id}: {exp}")
                return exp
    except requests.exceptions.HTTPError as e:
        try:
            log.error(f"get_next_expiry HTTP {e.response.status_code}: {e.response.text}")
        except Exception:
            log.error(f"get_next_expiry HTTP error: {e}")
    except Exception as e:
        log.error(f"get_next_expiry error: {e}")
    return None

def get_live_premium(name, spot, strike, opt_type):
    cfg = SYMBOLS[name]
    scrip_id = cfg["dhan_scrip"]
    if not DHAN_CLIENT_ID or not DHAN_ACCESS_TOKEN:
        return estimate_premium(spot, strike, opt_type, days_to_expiry(name)), False, None, None

    expiry = get_next_expiry(scrip_id)
    if not expiry:
        return estimate_premium(spot, strike, opt_type, days_to_expiry(name)), False, None, None

    try:
        resp = requests.post(
            "https://api.dhan.co/v2/optionchain",
            headers=DHAN_HEADERS,
            json={
                "UnderlyingScrip": int(scrip_id),
                "UnderlyingSeg": "IDX_I",
                "Expiry": expiry,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        oc = data.get("data", {}).get("oc", {})
        key = opt_type.lower()

        best_strike_key = None
        best_diff = float("inf")
        for sk in oc:
            try:
                diff = abs(float(sk) - float(strike))
                if diff < best_diff:
                    best_diff = diff
                    best_strike_key = sk
            except ValueError:
                pass

        if best_strike_key:
            option_data = oc[best_strike_key].get(key, {})
            ltp = option_data.get("last_price")
            delta = option_data.get("delta")
            iv = option_data.get("implied_volatility")
            if ltp and float(ltp) > 0:
                log.info(f"{name} LIVE LTP {strike} {opt_type}: Rs.{ltp} | Delta:{delta} | IV:{iv}")
                return round(float(ltp)), True, delta, iv
            else:
                log.warning(f"{name}: LTP=0 for {strike} {opt_type} - using estimate")
        else:
            log.warning(f"{name}: No matching strike in option chain")
    except requests.exceptions.HTTPError as e:
        try:
            log.error(f"get_live_premium HTTP {e.response.status_code}: {e.response.text}")
        except Exception:
            log.error(f"get_live_premium HTTP error: {e}")
    except Exception as e:
        log.error(f"get_live_premium error: {e}")

    return estimate_premium(spot, strike, opt_type, days_to_expiry(name)), False, None, None

def get_trade_live_premium(trade):
    try:
        spot = get_dhan_ltp(trade["symbol"])
        if spot is None:
            return None
        prem, is_live, delta, iv = get_live_premium(
            trade["symbol"],
            spot,
            trade["atm_strike"],
            trade["direction"]
        )
        return prem
    except Exception as e:
        log.error(f"get_trade_live_premium error: {e}")
        return None

# ─────────────────────────────────────────
# TELEGRAM HELPERS
# ─────────────────────────────────────────
def _tg(endpoint, payload):
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/{endpoint}",
            json=payload, timeout=10,
        )
        return r.json()
    except Exception as e:
        log.error(f"Telegram error ({endpoint}): {e}")
        return {}

def send_text(text):
    res = _tg("sendMessage", {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"})
    return res.get("result", {}).get("message_id")

def send_with_buttons(text, signal_id):
    keyboard = {
        "inline_keyboard": [
            [
                {"text": "Take Trade", "callback_data": f"take|{signal_id}"},
                {"text": "Skip", "callback_data": f"skip|{signal_id}"},
            ],
            [{"text": "Remind in 5 min", "callback_data": f"remind|{signal_id}"}],
        ]
    }
    res = _tg("sendMessage", {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
        "reply_markup": keyboard,
    })
    return res.get("result", {}).get("message_id")

def edit_message(message_id, text, keep_buttons=False):
    payload = {
        "chat_id": CHAT_ID,
        "message_id": message_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    if not keep_buttons:
        payload["reply_markup"] = {"inline_keyboard": []}
    _tg("editMessageText", payload)

def answer_callback(callback_id, text=""):
    _tg("answerCallbackQuery", {"callback_query_id": callback_id, "text": text, "show_alert": False})

# ─────────────────────────────────────────
# TELEGRAM COMMANDS
# ─────────────────────────────────────────
def handle_command(message):
    text = message.get("text", "").strip().lower()
    log.info(f"Command received: {text}")

     if text == "/start":
        send_text(
            "*Dhan Signal Bot — Commands*\n\n"
            "/status — Active trade, daily loss, bot state\n"
            "/confirmed — Mark active trade as placed by you\n"
            "/exited — Manually close active trade in bot\n"
            "/cancel — Cancel all pending signals\n"
            "/pause — Pause signal scanning\n"
            "/resume — Resume signal scanning\n"
            "/help — Show this list"
        )

    if text == "/status":
        at = get_st("active_trade")
        dl = get_st("daily_loss")
        ps = get_st("pending_signals")
        pau = get_st("paused")
        at_str = (
            f"*Active Trade*\n"
            f"{at['symbol']} {at['atm_strike']} {at['direction']}\n"
            f"Entry: Rs.{at.get('entry_premium_actual', at['atm_prem'])} | SL: Rs.{at.get('sl_premium_current', at['sl_prem'])} | T1: Rs.{at.get('t1_premium', at['tgt_prem'])} | T2: Rs.{at.get('t2_premium', at['tgt_prem'])}\n"
            f"Stage: {at.get('trade_stage', 'NA')}\n"
            f"T1 hit: {'Yes' if at.get('t1_hit') else 'No'}\n"
            f"Confirmed: {'Yes' if at.get('trade_confirmed') else 'No - use /confirmed'}"
            if at else "No active trade"
        )

        send_text(
            f"*Bot Status*\n\n"
            f"{at_str}\n\n"
            f"Daily loss : Rs.{dl:.0f} / Rs.{MAX_DAILY_LOSS:.0f}\n"
            f"Pending sigs: {len(ps)}\n"
            f"Scanning : {'PAUSED' if pau else 'Active'}\n"
            f"Dhan token : {'SET' if DHAN_ACCESS_TOKEN else 'MISSING'}"
        )

    elif text == "/confirmed":
        at = get_st("active_trade")
        if not at:
            send_text("No active trade to confirm.")
            return
        with _lock:
            state["active_trade"]["trade_confirmed"] = True
            state["active_trade"]["trade_stage"] = "OPEN"
            state["active_trade"]["confirmed_at"] = now_ist().isoformat()
            state["trade_confirmed"] = True
            _persist_state()
        send_text(
            f"Trade confirmed: {at['symbol']} {at['atm_strike']} {at['direction']}\n"
            f"Premium monitor is now active.\n"
            f"SL: Rs.{at.get('sl_premium_current', at['sl_prem'])} | "
            f"T1: Rs.{at.get('t1_premium', at['tgt_prem'])} | "
            f"T2: Rs.{at.get('t2_premium', at['tgt_prem'])}"
        )
        log.info("Trade manually confirmed via /confirmed")

    elif text == "/exited":
        at = get_st("active_trade")
        if not at:
            send_text("No active trade to exit.")
            return
        with _lock:
            state["active_trade"] = None
            state["trade_confirmed"] = False
            _persist_state()
        send_text(
            f"Trade manually closed: {at['symbol']} {at['atm_strike']} {at['direction']}\n"
            f"Bot reset. Ready for next signal."
        )
        log.info(f"Trade manually exited via /exited: {at['symbol']} {at['atm_strike']} {at['direction']}")

    elif text == "/cancel":
        ps = get_st("pending_signals")
        count = len(ps)
        with _lock:
            state["pending_signals"] = {}
            _persist_state()
        send_text(f"Cancelled {count} pending signal(s). Watching for next scan.")
        log.info(f"Pending signals cancelled via /cancel: {count}")

    elif text == "/pause":
        set_st("paused", True)
        send_text("Bot scanning PAUSED. Use /resume to restart scanning.\nPremium monitoring still active.")
        log.info("Bot paused via /pause")

    elif text == "/resume":
        set_st("paused", False)
        send_text("Bot scanning RESUMED. Will scan at next 5-min candle.")
        log.info("Bot resumed via /resume")

    elif text == "/help":
        send_text(
            "*Dhan Signal Bot — Commands*\n\n"
            "/status — Active trade, daily loss, bot state\n"
            "/confirmed — Mark active trade as placed by you\n"
            "/exited — Manually close active trade in bot\n"
            "/cancel — Cancel all pending signals\n"
            "/pause — Pause signal scanning\n"
            "/resume — Resume signal scanning\n"
            "/help — Show this list"
        )

    else:
        send_text(f"Unknown command: {text}\nSend /help for list of commands.")

# ─────────────────────────────────────────
# CALLBACK HANDLER
# ─────────────────────────────────────────
def handle_callback(query):
    cb_id = query["id"]
    data = query.get("data", "")
    message_id = query.get("message", {}).get("message_id")

    if "|" not in data:
        answer_callback(cb_id, "Unknown action")
        return

    action, signal_id = data.split("|", 1)
    pending = get_st("pending_signals")
    signal = pending.get(signal_id)

    if not signal:
        answer_callback(cb_id, "Signal expired")
        if message_id:
            edit_message(message_id, "Signal expired - already handled or timed out.")
        return

    if action == "take":
        with _lock:
            if state["active_trade"]:
                ex = state["active_trade"]
                answer_callback(cb_id, "Trade already open!")
                edit_message(
                    message_id,
                    f"Blocked - active trade exists:\n"
                    f"{ex['symbol']} {ex['atm_strike']} {ex['direction']}\n"
                    f"Use /exited to close it first."
                )
                return

            signal["trade_confirmed"] = False
            signal["highest_premium_seen"] = signal["atm_prem"]
            signal["lowest_premium_seen"] = signal["atm_prem"]
            signal["entry_premium_actual"] = signal["atm_prem"]
            signal["trade_stage"] = "PENDING_CONFIRMATION"
            signal["t1_hit"] = False
            signal["t2_hit"] = False
            signal["trail_enabled"] = False
            signal["opened_at"] = now_ist().isoformat()

            state["active_trade"] = signal
            state["pending_signals"].pop(signal_id, None)
            _persist_state()

        answer_callback(cb_id, "Trade logged!")
        s = signal
        cost_per_lot = s["atm_prem"] * s["lot"]
        live_tag = "(live)" if s.get("prem_is_live") else "(est.)"
        delta_str = f"Delta : {s['delta']:.2f}\n" if s.get("delta") is not None else ""
        iv_str = f"IV : {s['iv']:.1f}%\n" if s.get("iv") is not None else ""

        edit_message(
            message_id,
            f"*Trade Taken*\n\n"
            f"*{s['symbol']}* {s['atm_strike']} {s['direction']}\n"
            f"Regime: {s['regime']} | Strategy: {s['strategy']}\n"
            f"Stage: {s['trade_stage']}\n\n"
            f"Entry {live_tag} : Rs.{s['atm_prem']} per unit\n"
            f"Cost of 1 lot : Rs.{cost_per_lot:,} ({s['lot']} units)\n"
            f"Stop Loss : Rs.{s['sl_premium_current']}\n"
            f"T1 : Rs.{s['t1_premium']}\n"
            f"T2 : Rs.{s['t2_premium']}\n"
            f"{delta_str}{iv_str}\n"
            f"Send /confirmed once you place the order on Dhan."
        )
        log.info(f"Trade taken: {s['symbol']} {s['atm_strike']} {s['direction']}")

    elif action == "skip":
        with _lock:
            pending.pop(signal_id, None)
            _persist_state()
        answer_callback(cb_id, "Skipped")
        s = signal
        edit_message(
            message_id,
            f"Skipped: {s['symbol']} {s['atm_strike']} {s['direction']}\n"
            f"Watching for next signal..."
        )
        log.info(f"Skipped: {s['symbol']} {s['atm_strike']} {s['direction']}")

    elif action == "remind":
        answer_callback(cb_id, "Will remind at next scan")
        if message_id:
            edit_message(
                message_id,
                f"Reminder set: {signal['symbol']} {signal['atm_strike']} {signal['direction']}\n"
                f"Bot will re-alert in ~5 min.",
                keep_buttons=True
            )
    else:
        answer_callback(cb_id, "Unknown action")

# ─────────────────────────────────────────
# TELEGRAM POLLING THREAD
# ─────────────────────────────────────────
def telegram_polling_thread():
    log.info("Telegram polling thread started")
    offset = 0
    while True:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
                params={
                    "offset": offset,
                    "timeout": 25,
                    "allowed_updates": ["callback_query", "message"],
                },
                timeout=30,
            )
            for upd in resp.json().get("result", []):
                offset = upd["update_id"] + 1
                if "callback_query" in upd:
                    try:
                        handle_callback(upd["callback_query"])
                    except Exception as e:
                        log.error(f"Callback error: {e}")
                elif "message" in upd:
                    msg = upd["message"]
                    if msg.get("text", "").startswith("/"):
                        try:
                            handle_command(msg)
                        except Exception as e:
                            log.error(f"Command error: {e}")
        except requests.exceptions.Timeout:
            pass
        except Exception as e:
            log.error(f"Polling thread error: {e}")
            time.sleep(5)

# ─────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────
def now_ist():
    return datetime.now(IST)

def compute_indicators(df):
    df = df.copy()
    df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    df["atr"] = (df["High"] - df["Low"]).rolling(10).mean()
    cum_vol = df["Volume"].cumsum()
    cum_pv = (df["Close"] * df["Volume"]).cumsum()
    df["vwap"] = cum_pv / cum_vol.replace(0, 1e-9)
    return df

# ─────────────────────────────────────────
# REGIME DETECTION
# ─────────────────────────────────────────
def detect_regime(df, atr, ema9, ema21):
    recent = df.iloc[-10:]
    rng = recent["High"].max() - recent["Low"].min()
    ema_diff = abs(ema9 - ema21)
    if ema_diff > atr * 0.6 and rng > atr * 4:
        return "TRENDING"
    if ema_diff < atr * 0.3 and rng < atr * 3:
        return "SIDEWAYS"
    if rng > atr * 5:
        return "VOLATILE"
    return "NORMAL"

def confirm_regime(symbol, new_regime):
    rs = regime_state[symbol]
    if rs["last"] == new_regime:
        rs["count"] += 1
    else:
        rs["last"] = new_regime
        rs["count"] = 1
    return new_regime if rs["count"] >= 2 else None

# ─────────────────────────────────────────
# STRATEGIES
# ─────────────────────────────────────────
def strategy_breakout(df, atr, ema9, ema21, rsi, vwap):
    orb_high = float(df.iloc[:3]["High"].max())
    orb_low = float(df.iloc[:3]["Low"].min())
    close = float(df.iloc[-1]["Close"])
    conf = 0

    if close > orb_high or close < orb_low:
        conf += 3
    if (ema9 > ema21 and close > orb_high) or (ema9 < ema21 and close < orb_low):
        conf += 3
    if (rsi > 55 and close > orb_high) or (rsi < 45 and close < orb_low):
        conf += 2

    if conf < MIN_CONFIDENCE:
        return None

    if close > orb_high and close > vwap and ema9 > ema21:
        return "CE", close, close - atr, close + atr * 2, conf
    if close < orb_low and close < vwap and ema9 < ema21:
        return "PE", close, close + atr, close - atr * 2, conf
    return None

def strategy_range_trade(df, atr, ema9, ema21, rsi):
    recent = df.iloc[-10:]
    high = float(recent["High"].max())
    low = float(recent["Low"].min())
    close = float(df.iloc[-1]["Close"])
    buffer = atr * 0.3
    conf = 0

    if abs(ema9 - ema21) < atr * 0.3:
        conf += 3
    if close <= low + buffer or close >= high - buffer:
        conf += 3
    if (close <= low + buffer and rsi < 40) or (close >= high - buffer and rsi > 60):
        conf += 2

    if conf < MIN_CONFIDENCE:
        return None

    if close <= low + buffer:
        return "CE", close, close - atr * 0.8, close + atr * 1.2, conf
    if close >= high - buffer:
        return "PE", close, close + atr * 0.8, close - atr * 1.2, conf
    return None

def strategy_momentum(df, atr, rsi):
    last = df.iloc[-1]
    body = abs(float(last["Close"]) - float(last["Open"]))
    rng = float(last["High"]) - float(last["Low"])
    close = float(last["Close"])
    conf = 0

    if rng > 0 and body > rng * 0.7:
        conf += 3
    if atr > float(df["atr"].iloc[-6:-1].mean()) * 1.4:
        conf += 3
    if (last["Close"] > last["Open"] and rsi > 60) or (last["Close"] < last["Open"] and rsi < 40):
        conf += 2

    if conf < MIN_CONFIDENCE:
        return None

    if last["Close"] > last["Open"]:
        return "CE", close, close - atr * 1.3, close + atr * 2.5, conf
    return "PE", close, close + atr * 1.3, close - atr * 2.5, conf

# ─────────────────────────────────────────
# EXPIRY & PREMIUM HELPERS
# ─────────────────────────────────────────
def days_to_expiry(name):
    cfg = SYMBOLS.get(name, {})
    exp_day = cfg.get("expiry_day")
    today = datetime.now(IST).date()
    diff = (exp_day - today.weekday()) % 7
    return diff if diff > 0 else 7

def is_expiry_today(name):
    return datetime.now(IST).weekday() == SYMBOLS.get(name, {}).get("expiry_day")

def estimate_premium(spot, strike, opt_type, dte):
    iv = 0.14
    intrinsic = max(0, spot - strike) if opt_type == "CE" else max(0, strike - spot)
    time_val = round(spot * iv * max(dte, 1) / 365)
    return max(10, round(intrinsic + time_val))

# ─────────────────────────────────────────
# GREEKS WARNINGS
# ─────────────────────────────────────────
def greeks_warnings(delta, iv):
    warnings = []
    if iv is not None:
        try:
            iv_f = float(iv)
            if iv_f > 25:
                warnings.append(f"High IV ({iv_f:.1f}%) - premium expensive")
            elif iv_f < 8:
                warnings.append(f"Low IV ({iv_f:.1f}%) - low volatility")
        except Exception:
            pass

    if delta is not None:
        try:
            delta_f = abs(float(delta))
            if delta_f < 0.35:
                warnings.append(f"Low Delta ({delta_f:.2f}) - weak directional signal")
        except Exception:
            pass
    return warnings

# ─────────────────────────────────────────
# SIGNAL SCANNER
# ─────────────────────────────────────────
def scan_symbol(name):
    cfg = SYMBOLS[name]
    interval = cfg["interval"]
    lot = cfg["lot"]

    try:
        df = get_dhan_candles(name)
        if df is None:
            return None

        df = compute_indicators(df)
        last = df.iloc[-1]
        close = float(last["Close"])
        atr = float(last["atr"])
        ema9 = float(last["ema9"])
        ema21 = float(last["ema21"])
        rsi = float(last["rsi"])
        vwap = float(last["vwap"])

        raw_regime = detect_regime(df, atr, ema9, ema21)
        regime = confirm_regime(name, raw_regime)
        if not regime or regime == "NORMAL":
            log.info(f"{name}: Regime={raw_regime} not confirmed - skip")
            return None

        if regime == "TRENDING":
            result = strategy_breakout(df, atr, ema9, ema21, rsi, vwap)
            strategy = "ORB Breakout"
        elif regime == "SIDEWAYS":
            result = strategy_range_trade(df, atr, ema9, ema21, rsi)
            strategy = "Range Fade"
        else:
            result = strategy_momentum(df, atr, rsi)
            strategy = "Momentum"

        if result is None:
            log.info(f"{name}: {regime} confirmed but no setup - skip")
            return None

        direction, entry, sl_idx, tgt_idx, conf = result
        atm_strike = round(close / interval) * interval
        otm_strike = (atm_strike + interval) if direction == "CE" else (atm_strike - interval)
        dte = days_to_expiry(name)

        atm_prem, atm_is_live, delta, iv = get_live_premium(name, close, atm_strike, direction)
        otm_prem, _, _, _ = get_live_premium(name, close, otm_strike, direction)

        if is_expiry_today(name):
            sl_prem = round(atm_prem * 0.35)
            tgt_prem = round(atm_prem * 1.60)
        else:
            sl_prem = round(atm_prem * 0.45)
            tgt_prem = round(atm_prem * 1.90)

        risk_per_lot = max(1, (atm_prem - sl_prem) * lot)
        sugg_lots = max(1, int((CAPITAL * RISK_PCT) / risk_per_lot))
        cost_per_lot = atm_prem * lot
        warnings = greeks_warnings(delta, iv)
        expiry = get_next_expiry(cfg["dhan_scrip"])
        t1_prem = round(atm_prem + (atm_prem - sl_prem))
        t2_prem = round(atm_prem + 2 * (atm_prem - sl_prem))

        return {
            "id": str(uuid.uuid4())[:8],
            "symbol": name,
            "direction": direction,
            "option_type": direction,
            "confidence": conf,
            "regime": regime,
            "strategy": strategy,
            "close": round(close, 2),
            "atm_strike": atm_strike,
            "otm_strike": otm_strike,
            "expiry": expiry,
            "atm_prem": atm_prem,
            "otm_prem": otm_prem,
            "entry_premium": atm_prem,
            "entry_premium_actual": atm_prem,
            "prem_is_live": atm_is_live,
            "delta": delta,
            "iv": iv,
            "greeks_warnings": warnings,
            "sl_prem": sl_prem,
            "sl_premium_current": sl_prem,
            "tgt_prem": tgt_prem,
            "t1_premium": t1_prem,
            "t2_premium": t2_prem,
            "sugg_lots": sugg_lots,
            "cost_per_lot": cost_per_lot,
            "lot": lot,
            "sl_idx": round(sl_idx, 2),
            "tgt_idx": round(tgt_idx, 2),
            "atr": round(atr, 2),
            "rsi": round(rsi, 1),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "vwap": round(vwap, 2),
            "dte": dte,
            "expiry_today": is_expiry_today(name),
            "trade_confirmed": False,
            "trade_stage": "NEW",
            "t1_hit": False,
            "t2_hit": False,
            "trail_enabled": False,
            "highest_premium_seen": atm_prem,
            "lowest_premium_seen": atm_prem,
        }

    except Exception as e:
        log.error(f"{name}: scan_symbol error - {e}")
        return None

# ─────────────────────────────────────────
# PREMIUM MONITOR (V2 STEP 1)
# ─────────────────────────────────────────
def check_premium_sl_target():
    trade = get_st("active_trade")
    if not trade:
        return
    if not trade.get("trade_confirmed"):
        return

    live_prem = get_trade_live_premium(trade)
    if live_prem is None:
        log.warning("Premium monitor: live premium unavailable")
        return

    with _lock:
        current_trade = state.get("active_trade")
        if not current_trade:
            return

        current_trade["highest_premium_seen"] = max(current_trade.get("highest_premium_seen", live_prem), live_prem)
        current_trade["lowest_premium_seen"] = min(current_trade.get("lowest_premium_seen", live_prem), live_prem)
        _persist_state()

        sl = current_trade.get("sl_premium_current", current_trade["sl_prem"])
        t1 = current_trade.get("t1_premium", current_trade["tgt_prem"])
        t2 = current_trade.get("t2_premium", current_trade["tgt_prem"])

    log.info(
        f"Premium Monitor {trade['symbol']} {trade['atm_strike']} {trade['direction']}: "
        f"live={live_prem} | SL={sl} | T1={t1} | T2={t2}"
    )

    if live_prem <= sl:
        send_text(
            f"PREMIUM STOP LOSS HIT\n\n"
            f"{trade['symbol']} {trade['atm_strike']} {trade['direction']}\n"
            f"Live premium : Rs.{live_prem}\n"
            f"SL premium : Rs.{sl}\n\n"
            f"EXIT NOW.\n"
            f"Then send /exited to reset bot."
        )
        with _lock:
            state["daily_loss"] += CAPITAL * RISK_PCT
            state["active_trade"] = None
            state["trade_confirmed"] = False
            _persist_state()
        log.info(f"Premium SL hit: {trade['symbol']} {trade['atm_strike']} {trade['direction']}")
        return

    if (not trade.get("t1_hit")) and live_prem >= t1:
        with _lock:
            if state["active_trade"]:
                state["active_trade"]["t1_hit"] = True
                state["active_trade"]["trade_stage"] = "T1_HIT"
                state["active_trade"]["sl_premium_current"] = state["active_trade"].get(
                    "entry_premium_actual", state["active_trade"]["atm_prem"]
                )
                _persist_state()

        send_text(
            f"T1 HIT\n\n"
            f"{trade['symbol']} {trade['atm_strike']} {trade['direction']}\n"
            f"Live premium : Rs.{live_prem}\n"
            f"New SL moved to cost : Rs.{trade.get('entry_premium_actual', trade['atm_prem'])}"
        )
        log.info(f"T1 hit: {trade['symbol']} {trade['atm_strike']} {trade['direction']}")
        return

    if live_prem >= t2:
        send_text(
            f"T2 HIT\n\n"
            f"{trade['symbol']} {trade['atm_strike']} {trade['direction']}\n"
            f"Live premium : Rs.{live_prem}\n"
            f"Book profit now.\n"
            f"Then send /exited to reset bot."
        )
        with _lock:
            state["active_trade"] = None
            state["trade_confirmed"] = False
            _persist_state()
        log.info(f"T2 hit: {trade['symbol']} {trade['atm_strike']} {trade['direction']}")

# ─────────────────────────────────────────
# LEGACY INDEX MONITOR (kept as fallback reference)
# ─────────────────────────────────────────
def check_sl_target():
    trade = get_st("active_trade")
    if not trade:
        return
    sym = trade["symbol"]
    dire = trade["direction"]
    sl = trade["sl_idx"]
    tgt = trade["tgt_idx"]
    live = get_dhan_ltp(sym)
    if live is None:
        log.warning(f"{sym}: LTP unavailable for SL/Target check")
        return
    sl_hit = (live <= sl) if dire == "CE" else (live >= sl)
    tgt_hit = (live >= tgt) if dire == "CE" else (live <= tgt)
    log.info(f"Monitor {sym}: live={live:.2f} | SL={sl} | Tgt={tgt}")
    if sl_hit:
        send_text(
            f"STOP LOSS HIT\n\n"
            f"{sym} {trade['atm_strike']} {dire}\n"
            f"Index now : {live:,.2f}\n"
            f"SL level : {sl:,.2f}\n\n"
            f"EXIT NOW - no waiting.\n"
            f"Then send /exited to reset bot."
        )
        with _lock:
            state["daily_loss"] += CAPITAL * RISK_PCT
            state["active_trade"] = None
            state["trade_confirmed"] = False
            _persist_state()
        log.info(f"SL hit: {sym} {trade['atm_strike']} {dire}")
    elif tgt_hit:
        send_text(
            f"TARGET HIT\n\n"
            f"{sym} {trade['atm_strike']} {dire}\n"
            f"Index now : {live:,.2f}\n"
            f"Target : {tgt:,.2f}\n\n"
            f"BOOK PROFIT NOW.\n"
            f"Then send /exited to reset bot."
        )
        with _lock:
            state["active_trade"] = None
            state["trade_confirmed"] = False
            _persist_state()
        log.info(f"Target hit: {sym} {trade['atm_strike']} {dire}")

# ─────────────────────────────────────────
# MESSAGE BUILDERS
# ─────────────────────────────────────────
def build_signal_msg(s):
    exp_line = (
        "EXPIRY DAY - SL tightened. Exit before 2:45 PM."
        if s["expiry_today"] else f"{s['dte']} day(s) to expiry"
    )
    active = get_st("active_trade")
    block = (
        f"\nActive trade: {active['symbol']} {active['atm_strike']} {active['direction']}\n"
        f"Use /exited to close it first." if active else ""
    )
    live_tag = "(live)" if s.get("prem_is_live") else "(est.)"
    delta_str = f"Delta : {float(s['delta']):.2f}\n" if s.get("delta") is not None else ""
    iv_str = f"IV : {float(s['iv']):.1f}%\n" if s.get("iv") is not None else ""
    warn_str = ""
    if s.get("greeks_warnings"):
        warn_str = "\nWarnings:\n" + "\n".join(f" {w}" for w in s["greeks_warnings"]) + "\n"

    return (
        f"DHAN SIGNAL - {s['symbol']} {s['direction']}\n"
        f"------------------------------\n\n"
        f"Regime : {s['regime']}\n"
        f"Strategy : {s['strategy']}\n"
        f"Conf : {s['confidence']}/8\n\n"
        f"Index : {s['close']:,.2f}\n"
        f"EMA9/21 : {s['ema9']:,.2f} / {s['ema21']:,.2f}\n"
        f"RSI : {s['rsi']:.1f}\n"
        f"VWAP : {s['vwap']:,.2f}\n"
        f"ATR : {s['atr']:,.2f}\n"
        f"{delta_str}{iv_str}{warn_str}\n"
        f"WHAT TO BUY\n"
        f"ATM {s['atm_strike']} {s['direction']} : Rs.{s['atm_prem']} {live_tag}\n"
        f"OTM {s['otm_strike']} {s['direction']} : Rs.{s['otm_prem']} {live_tag} (cheaper)\n\n"
        f"Cost of 1 lot : Rs.{s['cost_per_lot']:,} ({s['lot']} units x Rs.{s['atm_prem']})\n"
        f"Stop Loss : Rs.{s['sl_premium_current']}\n"
        f"T1 : Rs.{s['t1_premium']}\n"
        f"T2 : Rs.{s['t2_premium']}\n"
        f"Suggested : {s['sugg_lots']} lot(s) | Risk: Rs.{int(CAPITAL * RISK_PCT)}\n\n"
        f"{exp_line}{block}\n\n"
        f"Tap a button below"
    )

def build_multi_summary(signals, best):
    lines = [f"{len(signals)} signals fired simultaneously\n"]
    for s in signals:
        marker = "BEST ->" if s["symbol"] == best["symbol"] else " -"
        lines.append(
            f"{marker} {s['symbol']} {s['direction']} {s['atm_strike']}"
            f" | {s['regime']} | Conf:{s['confidence']}/8 | Rs.{s['atm_prem']}"
        )
    lines.append("\nIndividual signals with buttons follow below")
    return "\n".join(lines)

def build_rules_msg(period):
    at = get_st("active_trade")
    dl = get_st("daily_loss")
    at_str = (
        f"Active trade: {at['symbol']} {at['atm_strike']} {at['direction']}"
        if at else "No active trade"
    )
    live_str = "Live premiums: ON (Dhan API)" if DHAN_ACCESS_TOKEN else "Live premiums: OFF (estimates only)"

    if period == "open":
        return (
            f"Dhan Signal Bot - Market Open\n\n"
            f"{live_str}\n\n"
            f"RULES\n"
            f"1. ONE trade at a time\n"
            f"2. Pick highest confidence signal\n"
            f"3. Never override Stop Loss\n"
            f"4. Exit all positions by 3:15 PM\n"
            f"5. Daily loss limit Rs.{MAX_DAILY_LOSS:.0f} - then stop\n"
            f"6. Expiry day: tighter SL, exit before 2:45 PM\n"
            f"7. Send /confirmed after placing order on Dhan\n\n"
            f"{at_str}\n\n"
            f"Send /help for all commands"
        )

    if period == "mid":
        return (
            f"MIDDAY CHECK\n\n"
            f"Daily loss : Rs.{dl:.0f} / Rs.{MAX_DAILY_LOSS:.0f}\n"
            f"{at_str}\n\n"
            f"Stay disciplined. No overtrading."
        )

    if period == "close":
        return (
            f"PRE-CLOSE ALERT\n\n"
            f"Daily loss : Rs.{dl:.0f} / Rs.{MAX_DAILY_LOSS:.0f}\n"
            f"{at_str}\n\n"
            f"No new trades after 3:00 PM.\n"
            f"Close all open trades before 3:15 PM."
        )

    return ""

# ─────────────────────────────────────────
# TIME UTILITIES
# ─────────────────────────────────────────
def time_str():
    return now_ist().strftime("%H:%M")

def wait_next_5min():
    n = now_ist()
    secs = n.minute * 60 + n.second
    gap = ((secs // 300) + 1) * 300 - secs
    log.info(f"Sleeping {gap}s until next 5-min candle")
    time.sleep(gap)

def is_trading_window():
    n = now_ist()
    if n.weekday() >= 5:
        return False
    m = n.hour * 60 + n.minute
    return 9 * 60 + 20 <= m <= 15 * 60 + 25

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def main():
    log.info("=" * 55)
    log.info(" Dhan Signal Bot - Production Ready v4 / V2 Step 1")
    log.info("=" * 55)
    log.info(f"DHAN_CLIENT_ID : {'SET' if DHAN_CLIENT_ID else 'MISSING'}")
    log.info(
        f"DHAN_ACCESS_TOKEN : "
        f"{'SET (len=' + str(len(DHAN_ACCESS_TOKEN)) + ')' if DHAN_ACCESS_TOKEN else 'MISSING'}"
    )

    _load_state()

    at = get_st("active_trade")
    if at and not at.get("trade_confirmed"):
        send_text(
            f"Bot restarted with unconfirmed trade:\n"
            f"{at['symbol']} {at['atm_strike']} {at['direction']}\n\n"
            f"Send /confirmed if order was placed.\n"
            f"Send /exited if you already closed it."
        )

    poll = threading.Thread(target=telegram_polling_thread, daemon=True)
    poll.start()

    last_sl_check = time.time()

    while True:
        n = now_ist()
        t = time_str()
        wday = n.weekday()

        # Weekend
        if wday >= 5:
            if not get_st("holiday_sent"):
                send_text("Market closed today. See you Monday!")
                set_st("holiday_sent", True)
            time.sleep(3600)
            continue

        # Daily reset
        if get_st("current_day") != n.date():
            with _lock:
                state.update({
                    "current_day": n.date(),
                    "daily_loss": 0.0,
                    "active_trade": None,
                    "pending_signals": {},
                    "rules_sent": {"open": False, "mid": False, "close": False},
                    "last_heartbeat_hour": -1,
                    "holiday_sent": False,
                    "paused": False,
                    "token_reminder_sent": False,
                    "trade_confirmed": False,
                })
                for nm in regime_state:
                    regime_state[nm] = {"last": None, "count": 0}
                _expiry_cache.clear()
                _persist_state()
            log.info(f"New day reset: {n.date()}")

        # Token reminder
        if "21:30" <= t < "21:31" and not get_st("token_reminder_sent"):
            send_text(
                "Dhan Token Reminder\n\n"
                "Your access token expires in ~12 hours.\n\n"
                "Steps:\n"
                "1. Go to https://api.dhan.co\n"
                "2. Generate new Access Token\n"
                "3. Update DHAN_ACCESS_TOKEN in Railway env variables\n"
                "4. Redeploy / restart the bot\n\n"
                "Do this before 9:20 AM tomorrow."
            )
            set_st("token_reminder_sent", True)
            log.info("Token reminder sent")

        # Scheduled messages
        rs = get_st("rules_sent")

        if "09:20" <= t < "09:30" and not rs["open"]:
            send_text(build_rules_msg("open"))
            with _lock:
                state["rules_sent"]["open"] = True
                _persist_state()

        if "12:30" <= t < "12:40" and not rs["mid"]:
            send_text(build_rules_msg("mid"))
            with _lock:
                state["rules_sent"]["mid"] = True
                _persist_state()

        if "15:00" <= t < "15:10" and not rs["close"]:
            send_text(build_rules_msg("close"))
            with _lock:
                state["rules_sent"]["close"] = True
                _persist_state()

        if "15:30" <= t < "15:31":
            at = get_st("active_trade")
            dl = get_st("daily_loss")
            send_text(
                f"Market Closed\n\n"
                f"Daily loss : Rs.{dl:.0f} / Rs.{MAX_DAILY_LOSS:.0f}\n"
                f"Open trade : {at['symbol'] + ' ' + str(at['atm_strike']) if at else 'None'}\n\n"
                f"See you tomorrow at 9:20 AM"
            )
            set_st("active_trade", None)

        # Trading window
        if is_trading_window():
            if get_st("daily_loss") >= MAX_DAILY_LOSS:
                log.info("Daily loss limit reached - pausing this cycle")
                wait_next_5min()
                continue

            now_ts = time.time()
            if now_ts - last_sl_check >= 30:
                check_premium_sl_target()
                last_sl_check = now_ts

            if not get_st("paused"):
                try:
                    signals = []
                    for name in SYMBOLS:
                        result = scan_symbol(name)
                        if result:
                            signals.append(result)

                    if signals:
                        best = max(
                            signals,
                            key=lambda x: (STRATEGY_RANK.get(x["regime"], 0), x["confidence"])
                        )
                        if len(signals) > 1:
                            send_text(build_multi_summary(signals, best))

                        for sig in signals:
                            msg_id = send_with_buttons(build_signal_msg(sig), sig["id"])
                            with _lock:
                                state["pending_signals"][sig["id"]] = {**sig, "msg_id": msg_id}
                                _persist_state()
                    else:
                        log.info("No confirmed signals this scan")

                except Exception as e:
                    log.error(f"Main scan error: {e}")
            else:
                log.info("Bot paused - skipping scan")

        wait_next_5min()

if __name__ == "__main__":
    main()
