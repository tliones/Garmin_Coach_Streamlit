# app.py
import os
import re
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, date

import pandas as pd
import pytz
import streamlit as st

# Unofficial Garmin API + token helper
try:
    from garminconnect import (
        Garmin,
        GarminConnectAuthenticationError,
        GarminConnectConnectionError,
    )
except Exception:
    st.error("The 'garminconnect' package is required. Install with: pip install garminconnect")
    st.stop()

try:
    from garth.exc import GarthHTTPError
except Exception:
    st.error("The 'garth' package is required. Install with: pip install garth")
    st.stop()

# OpenAI client (API key from Streamlit secrets)
try:
    from openai import OpenAI
except Exception:
    st.error("The 'openai' package is required. Install with: pip install openai")
    st.stop()

st.set_page_config(page_title="Garmin Coach (Unofficial)", page_icon="🏁", layout="wide")
st.title("🏁 Garmin Coach (Unofficial)")
st.caption("Logs in to Garmin, summarizes your past week's activities, and generates a tailored 7-day coaching plan. MFA + token reuse enabled. Note: uses an **unofficial** Garmin library.")

# ---------------------------
# Session defaults
# ---------------------------
for k, v in {
    "data_ready": False,
    "df_week": None,
    "totals": None,
    "by_type": None,
    "plan_text": "",
    "plan_json": None,
    "df_plan": None,
}.items():
    st.session_state.setdefault(k, v)

# ---------------------------
# Credentials / Inputs
# ---------------------------
st.subheader("🔐 Garmin Login")
email_raw = st.text_input("Garmin email", value="", autocomplete="username")
password_raw = st.text_input("Garmin password", value="", type="password", autocomplete="current-password")
twofa_code = st.text_input("MFA code (paste after Garmin emails you)", type="password")

email = (email_raw or "").strip()
password = (password_raw or "").strip()

with st.expander("Advanced: Token storage & behavior"):
    default_tokens_dir = Path(os.getenv("GARMINTOKENS", "~/.garminconnect")).expanduser()
    tok_dir_str = st.text_input("Token directory", value=str(default_tokens_dir))
    force_fresh = st.checkbox("Force fresh login (ignore saved tokens)", value=False)
    hydrate_limit = st.number_input(
        "Max activities to hydrate with details",
        min_value=1, max_value=500, value=60, step=5,
        help="How many recent activities to enrich with detailed power metrics if missing."
    )

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    tz_name = st.selectbox("Time Zone",
        ["America/Chicago", "UTC", "America/New_York", "America/Los_Angeles"], index=0)
with colB:
    now_tz = datetime.now(pytz.timezone(tz_name)).replace(microsecond=0, second=0)
    start_dt = (now_tz - timedelta(days=7)).replace(microsecond=0, second=0)
    end_dt = now_tz
    st.write(f"**Range:** {start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M}")
with colC:
    fetch_clicked = st.button("🔄 Fetch past week")

# ---------------------------
# Athlete metrics + goal (always editable)
# ---------------------------
st.subheader("🏃 Athlete Metrics & Goal")
m1, m2, m3 = st.columns(3)
with m1:
    ftp = st.number_input("Current FTP (W)", min_value=0, max_value=2000, value=260, step=1)
    rhr = st.number_input("Resting HR", min_value=20, max_value=120, value=47, step=1)
with m2:
    mhr = st.number_input("Max HR", min_value=100, max_value=230, value=183, step=1)
    thr_hr = st.number_input("Threshold HR (LTHR)", min_value=80, max_value=230, value=170, step=1)
with m3:
    vo2 = st.number_input("VO₂max (ml/kg/min)", min_value=0, max_value=100, value=53, step=1)
goal_text = st.text_area("Training goal (include constraints like weekly race day, time limits, focus areas)", height=80,
                         placeholder="e.g., Raise FTP to 280W by December, Tuesday Zwift race, long Z2 Saturday, <8h weekly, threshold & VO2 touches…")

# ---------------------------
# Login helper (MFA-aware)
# ---------------------------
class MFARequired(Exception):
    pass

@st.cache_resource(show_spinner=False)
def get_client(email: str, password: str, mfa_code: str | None, tokens_dir: str, force_fresh: bool = False):
    tokens_path = Path(tokens_dir).expanduser()
    status = {"phase": None, "detail": None}

    if not force_fresh:
        try:
            g = Garmin()
            g.login(str(tokens_path))
            status.update({"phase": "token_login", "detail": "Token login OK"})
            return g, status
        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError, GarminConnectConnectionError) as e:
            status.update({"phase": "token_login_failed", "detail": str(e)})

    g = Garmin(email=email, password=password, return_on_mfa=True)
    try:
        result1, result2 = g.login()
        status.update({"phase": "cred_login", "detail": f"result1={result1}, token={bool(result2)}"})
    except GarminConnectAuthenticationError as e:
        status.update({"phase": "cred_login_failed", "detail": str(e)})
        raise

    if result1 == "needs_mfa":
        if not mfa_code:
            raise MFARequired("MFA required—approve any new-device prompt in the Connect app, paste the emailed code, then click again.")
        g.resume_login(result2, mfa_code)
        status.update({"phase": "mfa_resume", "detail": "MFA code accepted"})

    tokens_path.mkdir(parents=True, exist_ok=True)
    g.garth.dump(str(tokens_path))
    status.update({"phase": "tokens_saved", "detail": f"Saved to {tokens_path}"})
    return g, status

# ---------------------------
# Data helpers (fetch, hydrate, normalize)
# ---------------------------
def fetch_week_activities(client: Garmin, start_iso: str, end_iso: str):
    try:
        return client.get_activities_by_date(start_iso, end_iso)
    except Exception:
        acts = client.get_activities(0, 200)
        return [a for a in acts if start_iso <= (a.get("startTimeLocal", a.get("startTimeGMT", ""))[:10]) <= end_iso]

def _extract_power_from_detail(detail: dict):
    avg_p = max_p = np_p = None
    if not isinstance(detail, dict):
        return avg_p, max_p, np_p
    s = detail.get("summaryDTO") or {}
    avg_p = avg_p or s.get("averagePower")
    max_p = max_p or s.get("maxPower")
    np_p  = np_p  or s.get("normalizedPower")
    avg_p = avg_p or detail.get("averagePower") or detail.get("avgPower")
    max_p = max_p or detail.get("maxPower")
    np_p  = np_p  or detail.get("normalizedPower")
    pdat = detail.get("powerData") or {}
    if isinstance(pdat, dict):
        avg_p = avg_p or pdat.get("avgPower") or pdat.get("averagePower")
        max_p = max_p or pdat.get("maxPower")
        np_p  = np_p  or pdat.get("normPower") or pdat.get("normalizedPower")
    return avg_p, max_p, np_p

def hydrate_power_metrics(client: Garmin, activities: list[dict], max_to_hydrate: int = 60, sleep_sec: float = 0.2):
    if not activities:
        return activities
    hydrated = 0
    for a in activities:
        if isinstance(a.get("averagePower"), (int, float)) and isinstance(a.get("maxPower"), (int, float)):
            continue
        if hydrated >= max_to_hydrate:
            break
        act_id = a.get("activityId") or a.get("activityUUID", {}).get("uuid")
        if not act_id:
            continue
        detail = None
        for getter in ("get_activity", "get_activity_summary", "get_activity_details"):
            try:
                detail = getattr(client, getter)(act_id)
                if detail:
                    break
            except Exception:
                detail = None
        if detail:
            avg_p, max_p, np_p = _extract_power_from_detail(detail)
            if avg_p is not None: a["averagePower"] = avg_p
            if max_p is not None: a["maxPower"] = max_p
            if np_p  is not None: a["normalizedPower"] = np_p
            hydrated += 1
            time.sleep(sleep_sec)
    return activities

def normalize_activities(activities):
    if not activities:
        return pd.DataFrame()
    rows = []
    for a in activities:
        atype = a.get("activityType", {}).get("typeKey") or a.get("activityType", "")
        name = a.get("activityName") or a.get("activityId")
        start_local = a.get("startTimeLocal") or a.get("startTimeGMT")
        distance_m = a.get("distance")
        duration_s = a.get("duration")
        avg_hr = a.get("averageHR")
        max_hr = a.get("maxHR")
        avg_power = a.get("averagePower")
        max_power = a.get("maxPower")
        np_power  = a.get("normalizedPower")
        avg_speed = a.get("averageSpeed")
        try:
            when = pd.to_datetime(start_local)
        except Exception:
            when = pd.NaT
        rows.append({
            "date": when,
            "type": str(atype),
            "name": name,
            "distance_km": (distance_m/1000.0) if isinstance(distance_m, (int, float)) else None,
            "duration_min": (duration_s/60.0) if isinstance(duration_s, (int, float)) else None,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "avg_power": avg_power,
            "max_power": max_power,
            "np_power": np_power,
            "avg_speed_kmh": (avg_speed*3.6) if isinstance(avg_speed, (int, float)) else None,
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df.sort_values("date", ascending=False)

def summarize(df):
    if df.empty:
        return {}, pd.DataFrame()
    totals = {
        "activities": int(df.shape[0]),
        "total_time_hr": round(df["duration_min"].fillna(0).sum()/60.0, 2),
        "total_distance_km": round(df["distance_km"].fillna(0).sum(), 2),
        "avg_hr": round(df["avg_hr"].dropna().mean(), 1) if df["avg_hr"].notna().any() else None,
        "avg_speed_kmh": round(df["avg_speed_kmh"].dropna().mean(), 1) if df["avg_speed_kmh"].notna().any() else None,
        "avg_power": round(df["avg_power"].dropna().mean(), 0) if df["avg_power"].notna().any() else None,
        "np_power": round(df["np_power"].dropna().mean(), 0) if df["np_power"].notna().any() else None,
    }
    by_type = (
        df.groupby("type")
          .agg(
              activities=("type", "count"),
              time_hr=("duration_min", lambda s: round(s.fillna(0).sum()/60.0, 2)),
              distance_km=("distance_km", lambda s: round(s.fillna(0).sum(), 2)),
              avg_hr=("avg_hr", "mean"),
              avg_speed_kmh=("avg_speed_kmh", "mean"),
              avg_power=("avg_power", "mean"),
              np_power=("np_power", "mean"),
          )
          .reset_index()
          .sort_values("time_hr", ascending=False)
    )
    for col in ["avg_hr", "avg_speed_kmh", "avg_power", "np_power"]:
        if col in by_type:
            by_type[col] = by_type[col].round(1 if col in ["avg_hr", "avg_speed_kmh"] else 0)
    return totals, by_type

def _to_naive_ts(dt):
    ts = pd.Timestamp(dt)
    return ts.tz_convert(None) if ts.tz is not None else ts

# ---------------------------
# OpenAI helpers (Plan + Chat)
# ---------------------------
def get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OpenAI API key. Add OPENAI_API_KEY to Streamlit secrets.")
        st.stop()
    return OpenAI(api_key=api_key)

def extract_plan_json(text: str):
    m = re.search(r"```(?:plan_json)\s*(\{.*?\})\s*```", text, flags=re.S)
    if m: return m.group(1)
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if m: return m.group(1)
    return None

def build_context_payload(df_week: pd.DataFrame, totals: dict, inputs: dict):
    sample_cols = ["date","type","name","distance_km","duration_min","avg_hr","avg_power","np_power","avg_speed_kmh"]
    adf_light = (
        df_week[sample_cols].copy()
               .tail(20)
               .assign(date=lambda d: d["date"].astype(str))
               .to_dict("records")
    )
    weekly_by_type = (
        df_week.groupby("type")
               .agg(sessions=("type","count"),
                    time_min=("duration_min","sum"),
                    distance_km=("distance_km","sum"),
                    avg_hr=("avg_hr","mean"),
                    avg_power=("avg_power","mean"))
               .reset_index()
               .to_dict("records")
    )
    start_day = date.today() + timedelta(days=1)
    week_dates = [(start_day + timedelta(days=i)).isoformat() for i in range(7)]

    return {
        "generated_on": date.today().isoformat(),
        "plan_dates": week_dates,
        "goal_text": inputs.get("goal_text",""),
        "athlete_metrics": {
            "ftp_w": inputs.get("ftp"),
            "rhr_bpm": inputs.get("rhr"),
            "mhr_bpm": inputs.get("mhr"),
            "lthr_bpm": inputs.get("thr_hr"),
            "vo2max": inputs.get("vo2"),
        },
        "week_totals": totals,
        "weekly_by_type": weekly_by_type,
        "recent_activities_light": adf_light,
    }

def generate_plan(client, context_pack: dict):
    system_prompt = (
        "You are a seasoned cycling coach. Create a precise, practical 7-day plan.\n"
        "Use athlete metrics and the recent week summary. Respect constraints in goal_text.\n"
        "Favor longer Z2 where appropriate; include at least one threshold or VO2 touch if readiness allows.\n"
        "Keep weekly load reasonable; avoid sudden spikes."
    )
    schema_block = """{
      "week_overview": { "intended_load_sum": number, "key_objectives": [string, ...], "notes": string },
      "days": {
        "<ISO date>": {
          "workout_name": string,
          "focus_zone": string,
          "duration_min": number,
          "warmup": { "workout_name": string, "duration_min": number } | null,
          "brief_rationale": string,
          "optional_notes": string
        }
      }
    }"""
    ctx_json = json.dumps(context_pack, indent=2)
    user_prompt = (
        "Context (JSON):\n```json\n%(ctx)s\n```\n\n"
        "Output TWO sections:\n"
        "1) A concise human summary.\n"
        "2) A JSON object under a fenced block labeled plan_json following this schema:\n"
        "```json\n%(schema)s\n```\n"
    ) % {"ctx": ctx_json, "schema": schema_block}

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_prompt},
        ],
        temperature=0.3,
    )
    text = resp.choices[0].message.content or ""
    js = extract_plan_json(text)
    df_plan = None; plan = None
    if js:
        try:
            plan = json.loads(js)
            if isinstance(plan, dict) and "days" in plan:
                df_plan = (
                    pd.DataFrame.from_dict(plan["days"], orient="index")
                      .reset_index().rename(columns={"index":"date"})
                      .sort_values("date")
                )
        except Exception:
            pass
    return text, plan, df_plan

def chat_with_coach(client, plan_text: str, plan_json: dict, question: str):
    context = "Plan (raw):\n" + (plan_text[:6000] if plan_text else "N/A")
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a helpful cycling coach. Answer clearly and concisely with actionable advice."},
            {"role":"user","content": context},
            {"role":"user","content": question},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

# ---------------------------
# Fetch button: login + fetch + hydrate; store in session_state
# ---------------------------
if fetch_clicked:
    if not email or not password:
        st.warning("Please enter your Garmin email and password.")
        st.stop()

    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")

    with st.spinner("Signing in to Garmin..."):
        try:
            client, status = get_client(
                email=email,
                password=password,
                mfa_code=(twofa_code or "").strip() or None,
                tokens_dir=tok_dir_str,
                force_fresh=force_fresh,
            )
        except MFARequired as mfa_err:
            st.info("🔐 " + str(mfa_err))
            st.caption("If no email arrives, approve the new sign-in in Garmin Connect, then paste the code and click again.")
            st.stop()
        except GarminConnectAuthenticationError:
            st.error("Authentication failed. Check email/password (and MFA if required).")
            st.stop()
        except GarminConnectConnectionError:
            st.error("Connection error contacting Garmin.")
            st.stop()
        except GarthHTTPError as e:
            st.error(f"Login error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    with st.expander("Login status (debug)"):
        st.write(status)
    st.success(f"✅ Logged in. {status.get('detail','')}")

    with st.spinner("Fetching activities for the past week..."):
        try:
            activities = fetch_week_activities(client, start_iso, end_iso)
        except Exception as e:
            st.error(f"Failed to fetch activities: {e}")
            st.stop()

    with st.spinner("Hydrating power metrics from activity details..."):
        try:
            activities = hydrate_power_metrics(client, activities, max_to_hydrate=int(hydrate_limit), sleep_sec=0.2)
        except Exception as e:
            st.warning(f"Could not hydrate detailed power for some activities: {e}")

    df = normalize_activities(activities)
    if df.empty:
        st.info("No activities found in the past week.")
        st.stop()

    start_naive = _to_naive_ts(start_dt)
    end_naive = _to_naive_ts(end_dt)
    mask = (df["date"] >= start_naive) & (df["date"] <= end_naive)
    df_week = df.loc[mask].copy()
    totals, by_type = summarize(df_week)

    # Persist data across reruns
    st.session_state["data_ready"] = True
    st.session_state["df_week"] = df_week
    st.session_state["totals"] = totals
    st.session_state["by_type"] = by_type

# ---------------------------
# If data is ready, render summary + plan/chat UI (outside the fetch block)
# ---------------------------
if st.session_state["data_ready"]:
    df_week = st.session_state["df_week"]
    totals = st.session_state["totals"]
    by_type = st.session_state["by_type"]

    st.subheader("📊 Weekly Totals")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Activities", totals.get("activities", 0))
    c2.metric("Time (hr)", totals.get("total_time_hr", 0.0))
    c3.metric("Distance (km)", totals.get("total_distance_km", 0.0))
    c4.metric("Avg HR", totals.get("avg_hr", "—") if totals.get("avg_hr") is not None else "—")
    c5.metric("Avg Speed (km/h)", totals.get("avg_speed_kmh", "—") if totals.get("avg_speed_kmh") is not None else "—")
    c6.metric("Avg Power (W)", totals.get("avg_power", "—") if totals.get("avg_power") is not None else "—")
    c7.metric("NP (W)", totals.get("np_power", "—") if totals.get("np_power") is not None else "—")

    st.subheader("🗂️ By Activity Type")
    st.dataframe(by_type, use_container_width=True)

    st.subheader("📅 Activities (past 7 days)")
    show_cols = ["date","type","name","distance_km","duration_min","avg_hr","max_hr","avg_power","max_power","np_power","avg_speed_kmh"]
    st.dataframe(df_week[show_cols], use_container_width=True)

    # ---------- Generate Coaching Plan (separate button; no re-login) ----------
    st.markdown("---")
    st.subheader("🧠 Generate Coaching Plan")
    if st.button("✨ Create 7-day plan"):
        client_oa = get_openai_client()
        inputs = {"ftp": ftp, "rhr": rhr, "mhr": mhr, "thr_hr": thr_hr, "vo2": vo2, "goal_text": goal_text}
        ctx = build_context_payload(df_week, totals, inputs)
        with st.spinner("Assembling plan with OpenAI…"):
            plan_text, plan_json, df_plan = generate_plan(client_oa, ctx)
        st.session_state["plan_text"] = plan_text
        st.session_state["plan_json"] = plan_json
        st.session_state["df_plan"] = df_plan

    if st.session_state.get("plan_text"):
        st.markdown("### 📄 Coach Plan (summary + JSON)")
        st.write(st.session_state["plan_text"])
        if isinstance(st.session_state.get("df_plan"), pd.DataFrame):
            st.markdown("**Day-by-Day Table**")
            st.dataframe(st.session_state["df_plan"], use_container_width=True)

    # ---------- Coach Chat (uses stored plan as context) ----------
    st.markdown("---")
    st.subheader("💬 Chat with your coach")
    chat_q = st.text_input("Ask about the plan, substitutions, or adjustments:")
    if st.button("Send to coach"):
        if not chat_q.strip():
            st.warning("Type a question first.")
        elif not st.session_state.get("plan_text"):
            st.warning("Generate a plan first so the coach has context.")
        else:
            client_oa = get_openai_client()
            with st.spinner("Coach is thinking…"):
                ans = chat_with_coach(client_oa, st.session_state["plan_text"], st.session_state.get("plan_json"), chat_q.strip())
            st.markdown("**Coach:**")
            st.write(ans)

    # ---------- Export ----------
    st.download_button(
        "⬇️ Download weekly activities (CSV)",
        df_week.to_csv(index=False).encode("utf-8"),
        file_name="garmin_weekly_activities.csv",
        mime="text/csv",
    )
else:
    st.info("Enter your Garmin credentials and click **Fetch past week** to begin.")



