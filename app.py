# app.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

import pandas as pd
import pytz
import streamlit as st

# Third-party (unofficial Garmin API client + token helper)
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

st.set_page_config(page_title="Garmin Weekly Summary", page_icon="⏱️", layout="wide")
st.title("🏃‍♀️ Garmin Weekly Summary (Unofficial)")
st.caption("Logs in to Garmin, handles MFA, caches tokens, and summarizes your past week's activities. Now with power hydration from detailed endpoints.")

# ---------------------------
# Inputs
# ---------------------------
st.subheader("🔐 Garmin Login")
email_raw = st.text_input("Garmin email", value="", autocomplete="username")
password_raw = st.text_input("Garmin password", value="", type="password", autocomplete="current-password")
twofa_code = st.text_input("MFA code (paste after Garmin emails you)", type="password")

# Trim whitespace to avoid validation issues from pasted values
email = (email_raw or "").strip()
password = (password_raw or "").strip()

with st.expander("Advanced: Token storage & behavior"):
    default_tokens_dir = Path(os.getenv("GARMINTOKENS", "~/.garminconnect")).expanduser()
    tok_dir_str = st.text_input("Token directory", value=str(default_tokens_dir))
    force_fresh = st.checkbox("Force fresh login (ignore saved tokens)", value=False)
    hydrate_limit = st.number_input("Max activities to hydrate with details", min_value=1, max_value=500, value=60, step=5, help="How many recent activities to enrich with detailed power metrics if missing.")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    tz_name = st.selectbox(
        "Time Zone",
        ["America/Chicago", "UTC", "America/New_York", "America/Los_Angeles"],
        index=0,
    )
with colB:
    now_tz = datetime.now(pytz.timezone(tz_name)).replace(microsecond=0, second=0)
    start_dt = (now_tz - timedelta(days=7)).replace(microsecond=0, second=0)
    end_dt = now_tz
    st.write(f"**Range:** {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}")
with colC:
    fetch_clicked = st.button("🔄 Fetch past week")

# ---------------------------
# Login helper (MFA-aware)
# ---------------------------
class MFARequired(Exception):
    """Raised when Garmin indicates MFA is required but no code was provided."""
    pass

@st.cache_resource(show_spinner=False)
def get_client(
    email: str,
    password: str,
    mfa_code: str | None,
    tokens_dir: str,
    force_fresh: bool = False,
) -> tuple[Garmin, dict]:
    """
    Returns (garmin_client, status_info)
    status_info has small details to display in the UI for debugging.
    """
    tokens_path = Path(tokens_dir).expanduser()
    status = {"phase": None, "detail": None}

    # 1) Token-based login first (unless forcing fresh)
    if not force_fresh:
        try:
            g = Garmin()
            g.login(str(tokens_path))
            status.update({"phase": "token_login", "detail": "Token login OK"})
            return g, status
        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError, GarminConnectConnectionError) as e:
            status.update({"phase": "token_login_failed", "detail": str(e)})

    # 2) Fresh credential login with MFA handoff
    g = Garmin(email=email, password=password, return_on_mfa=True)
    try:
        result1, result2 = g.login()
        status.update({"phase": "cred_login", "detail": f"result1={result1}, token={bool(result2)}"})
    except GarminConnectAuthenticationError as e:
        status.update({"phase": "cred_login_failed", "detail": str(e)})
        raise

    # If Garmin requires MFA
    if result1 == "needs_mfa":
        if not mfa_code:
            raise MFARequired("MFA required—check your Connect app for any new-device approval, then paste the emailed code and click again.")
        g.resume_login(result2, mfa_code)
        status.update({"phase": "mfa_resume", "detail": "MFA code accepted"})

    # 3) Save tokens for reuse
    tokens_path.mkdir(parents=True, exist_ok=True)
    g.garth.dump(str(tokens_path))
    status.update({"phase": "tokens_saved", "detail": f"Saved to {tokens_path}"})
    return g, status

# ---------------------------
# Data helpers
# ---------------------------
def fetch_week_activities(client: Garmin, start_iso: str, end_iso: str):
    """Fetch activities in date range; fall back to batch+filter."""
    try:
        return client.get_activities_by_date(start_iso, end_iso)
    except Exception:
        acts = client.get_activities(0, 200)
        return [a for a in acts if start_iso <= (a.get("startTimeLocal", a.get("startTimeGMT", ""))[:10]) <= end_iso]

def _extract_power_from_detail(detail: dict) -> tuple[float | None, float | None, float | None]:
    """
    Try several known locations for power metrics. Returns (avg_power, max_power, np_power).
    """
    avg_p = max_p = np_p = None
    if not isinstance(detail, dict):
        return avg_p, max_p, np_p

    # 1) summaryDTO (most reliable)
    s = detail.get("summaryDTO") or {}
    avg_p = avg_p or s.get("averagePower")
    max_p = max_p or s.get("maxPower")
    np_p  = np_p  or s.get("normalizedPower")

    # 2) direct keys (some endpoints flatten)
    avg_p = avg_p or detail.get("averagePower") or detail.get("avgPower")
    max_p = max_p or detail.get("maxPower")
    np_p  = np_p  or detail.get("normalizedPower")

    # 3) powerData subobject
    pdat = detail.get("powerData") or {}
    if isinstance(pdat, dict):
        avg_p = avg_p or pdat.get("avgPower") or pdat.get("averagePower")
        max_p = max_p or pdat.get("maxPower")
        np_p  = np_p  or pdat.get("normPower") or pdat.get("normalizedPower")

    return avg_p, max_p, np_p

def hydrate_power_metrics(client: Garmin, activities: list[dict], max_to_hydrate: int = 60, sleep_sec: float = 0.2) -> list[dict]:
    """
    For activities missing averagePower or maxPower, pull detailed info and fill values.
    Limits calls for rate-safety.
    """
    if not activities:
        return activities

    hydrated = 0
    for a in activities:
        # If already have power, skip
        if isinstance(a.get("averagePower"), (int, float)) and isinstance(a.get("maxPower"), (int, float)):
            continue

        # Only hydrate recent subset
        if hydrated >= max_to_hydrate:
            break

        act_id = a.get("activityId") or a.get("activityUUID", {}).get("uuid")
        if not act_id:
            continue

        detail = None
        # Try a few detail endpoints in order of richness/availability
        for getter in ("get_activity", "get_activity_summary", "get_activity_details"):
            try:
                detail = getattr(client, getter)(act_id)
                if detail:
                    break
            except Exception:
                detail = None

        if detail:
            avg_p, max_p, np_p = _extract_power_from_detail(detail)
            if avg_p is not None:
                a["averagePower"] = avg_p
            if max_p is not None:
                a["maxPower"] = max_p
            # Stash NP if useful later
            if np_p is not None:
                a["normalizedPower"] = np_p

            hydrated += 1
            time.sleep(sleep_sec)  # be gentle with API

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
            "distance_km": (distance_m / 1000.0) if isinstance(distance_m, (int, float)) else None,
            "duration_min": (duration_s / 60.0) if isinstance(duration_s, (int, float)) else None,
            "avg_hr": avg_hr,
            "max_hr": max_hr,
            "avg_power": avg_power,
            "max_power": max_power,
            "np_power": np_power,
            "avg_speed_kmh": (avg_speed * 3.6) if isinstance(avg_speed, (int, float)) else None,
        })

    df = pd.DataFrame(rows)

    # Normalize datetime: make tz-naive for consistent comparisons
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    df = df.sort_values("date", ascending=False)
    return df

def summarize(df):
    if df.empty:
        return {}, pd.DataFrame()

    totals = {
        "activities": int(df.shape[0]),
        "total_time_hr": round(df["duration_min"].fillna(0).sum() / 60.0, 2),
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
              time_hr=("duration_min", lambda s: round(s.fillna(0).sum() / 60.0, 2)),
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
# Button handler
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
            st.caption("If no email arrives yet, open the Garmin Connect app and approve the new sign-in/device, then click again.")
            st.stop()
        except GarminConnectAuthenticationError:
            st.error("Authentication failed. Double-check your email/password (and MFA code if required).")
            st.stop()
        except GarminConnectConnectionError:
            st.error("Connection error contacting Garmin. Please try again.")
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

    # ---------- NEW: hydrate detailed power metrics ----------
    with st.spinner("Hydrating power metrics from activity details..."):
        try:
            activities = hydrate_power_metrics(client, activities, max_to_hydrate=int(hydrate_limit), sleep_sec=0.2)
        except Exception as e:
            st.warning(f"Could not hydrate detailed power for some activities: {e}")

    df = normalize_activities(activities)
    if df.empty:
        st.info("No activities found in the past week.")
        st.stop()

    # Use tz-naive comparison
    start_naive = _to_naive_ts(start_dt)
    end_naive = _to_naive_ts(end_dt)
    mask = (df["date"] >= start_naive) & (df["date"] <= end_naive)
    df_week = df.loc[mask].copy()

    totals, by_type = summarize(df_week)

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
    show_cols = ["date", "type", "name", "distance_km", "duration_min", "avg_hr", "max_hr", "avg_power", "max_power", "np_power", "avg_speed_kmh"]
    st.dataframe(df_week[show_cols], use_container_width=True)

    import matplotlib.pyplot as plt

    # Time by Day
    day = df_week.copy()
    day["day"] = day["date"].dt.strftime("%a %m-%d")
    day_agg = day.groupby("day")["duration_min"].sum().reset_index()

    fig1, ax1 = plt.subplots()
    ax1.bar(day_agg["day"], day_agg["duration_min"])
    ax1.set_title("Total Minutes by Day")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Minutes")
    st.pyplot(fig1)

    # Distance by Type
    type_agg = df_week.groupby("type")["distance_km"].sum().reset_index()
    fig2, ax2 = plt.subplots()
    ax2.bar(type_agg["type"], type_agg["distance_km"])
    ax2.set_title("Total Distance by Activity Type")
    ax2.set_xlabel("Activity Type")
    ax2.set_ylabel("Distance (km)")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig2)

    st.download_button(
        "⬇️ Download weekly activities (CSV)",
        df_week.to_csv(index=False).encode("utf-8"),
        file_name="garmin_weekly_activities.csv",
        mime="text/csv",
    )
else:
    st.info("Enter your Garmin credentials and click **Fetch past week** to begin.")



