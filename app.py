import os
from pathlib import Path
from datetime import datetime, timedelta
import pytz
import pandas as pd
import streamlit as st

# Third-party (unofficial Garmin API)
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
    # 'garth' is a transitive dependency of garminconnect; import explicitly to handle token errors.
    st.error("The 'garth' package is required. Install with: pip install garth")
    st.stop()

st.set_page_config(page_title="Garmin Weekly Summary", page_icon="⏱️", layout="wide")
st.title("🏃‍♀️ Garmin Weekly Summary (Unofficial)")
st.caption("Logs in to Garmin, handles MFA, caches tokens, and summarizes your past week's activities.")

# --- Credential Inputs (always via UI) ---
st.subheader("🔐 Garmin Login")
email = st.text_input("Garmin email", value="", autocomplete="username")
password = st.text_input("Garmin password", value="", type="password", autocomplete="current-password")
twofa_code = st.text_input("MFA code (paste after Garmin emails you)", type="password")

# Optional: let the user set a token directory (useful on Streamlit Cloud/session containers)
with st.expander("Advanced: Token storage location"):
    default_tokens_dir = Path(os.getenv("GARMINTOKENS", "~/.garminconnect")).expanduser()
    tok_dir_str = st.text_input("Token directory", value=str(default_tokens_dir))
    st.caption("Tokens let you avoid repeated MFA on the same server.")

colA, colB, colC = st.columns([1,1,1])
with colA:
    tz_name = st.selectbox(
        "Time Zone",
        ["America/Chicago", "UTC", "America/New_York", "America/Los_Angeles"],
        index=0
    )
with colB:
    end_dt = datetime.now(pytz.timezone(tz_name)).replace(microsecond=0, second=0)
    start_dt = (end_dt - timedelta(days=7)).replace(microsecond=0, second=0)
    st.write(f"**Range:** {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}")
with colC:
    go = st.button("🔄 Fetch past week")

# --- MFA-aware login helper ---
class MFARequired(Exception):
    pass

@st.cache_resource(show_spinner=False)
def get_client(email: str, password: str, mfa_code: str | None, tokens_dir: str) -> Garmin:
    tokens_path = Path(tokens_dir).expanduser()
    # 1) Try token-based login first
    try:
        g = Garmin()
        g.login(str(tokens_path))
        return g
    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError, GarminConnectConnectionError):
        pass  # fall through

    # 2) Fresh credential login with return_on_mfa
    g = Garmin(email=email, password=password, return_on_mfa=True)
    try:
        result1, result2 = g.login()
    except GarminConnectAuthenticationError as e:
        # bad credentials or unexpected challenge
        raise

    if result1 == "needs_mfa":
        if not mfa_code:
            raise MFARequired("MFA required — check your email for the Garmin code, paste it above, then click again.")
        # Finish the login with the provided MFA code
        g.resume_login(result2, mfa_code)

    # Save tokens for reuse
    tokens_path.mkdir(parents=True, exist_ok=True)
    g.garth.dump(str(tokens_path))
    return g

def fetch_week_activities(client: Garmin, start_iso: str, end_iso: str):
    # Preferred by-date API (not always available for all accounts/versions)
    try:
        return client.get_activities_by_date(start_iso, end_iso)
    except Exception:
        # Fallback: fetch a batch and filter
        acts = client.get_activities(0, 200)
        return [a for a in acts if start_iso <= (a.get("startTimeLocal", a.get("startTimeGMT", ""))[:10]) <= end_iso]

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
            "avg_speed_kmh": (avg_speed*3.6) if isinstance(avg_speed, (int, float)) else None,
        })

    df = pd.DataFrame(rows).sort_values("date", ascending=False)
    return df

def summarize(df):
    if df.empty:
        return {}, pd.DataFrame()

    totals = {
        "activities": int(df.shape[0]),
        "total_time_hr": round(df["duration_min"].fillna(0).sum()/60.0, 2),
        "total_distance_km": round(df["distance_km"].fillna(0).sum(), 2),
        "avg_hr": round(df["avg_hr"].dropna().mean(), 1) if df["avg_hr"].notna().any() else None,
        "avg_speed_kmh": round(df["avg_speed_kmh"].dropna().mean(), 1) if df["avg_speed_kmh"].notna().any() else None,
    }
    by_type = (
        df.groupby("type")
          .agg(
              activities=("type", "count"),
              time_hr=("duration_min", lambda s: round(s.fillna(0).sum()/60.0, 2)),
              distance_km=("distance_km", lambda s: round(s.fillna(0).sum(), 2)),
              avg_hr=("avg_hr", "mean"),
              avg_speed_kmh=("avg_speed_kmh", "mean"),
          )
          .reset_index()
          .sort_values("time_hr", ascending=False)
    )
    by_type["avg_hr"] = by_type["avg_hr"].round(1)
    by_type["avg_speed_kmh"] = by_type["avg_speed_kmh"].round(1)
    return totals, by_type

if go:
    if not email or not password:
        st.warning("Please enter your Garmin email and password.")
        st.stop()

    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso = end_dt.strftime("%Y-%m-%d")

    with st.spinner("Signing in to Garmin..."):
        try:
            client = get_client(email, password, twofa_code.strip() or None, tok_dir_str)
        except MFARequired as mfa_err:
            st.info("🔐 " + str(mfa_err))
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

    # If we reached here, login is complete. Let the user know where tokens were saved (first-time only).
    st.success(f"✅ Logged in. Tokens saved to: {Path(tok_dir_str).expanduser()}")

    with st.spinner("Fetching activities for the past week..."):
        try:
            activities = fetch_week_activities(client, start_iso, end_iso)
        except Exception as e:
            st.error(f"Failed to fetch activities: {e}")
            st.stop()

    df = normalize_activities(activities)
    if df.empty:
        st.info("No activities found in the past week.")
        st.stop()

    # Restrict exactly to the time window
    mask = (df["date"] >= pd.to_datetime(start_dt)) & (df["date"] <= pd.to_datetime(end_dt))
    df_week = df.loc[mask].copy()

    totals, by_type = summarize(df_week)

    st.subheader("📊 Weekly Totals")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Activities", totals.get("activities", 0))
    c2.metric("Time (hr)", totals.get("total_time_hr", 0.0))
    c3.metric("Distance (km)", totals.get("total_distance_km", 0.0))
    c4.metric("Avg HR", totals.get("avg_hr", "—") if totals.get("avg_hr") is not None else "—")
    c5.metric("Avg Speed (km/h)", totals.get("avg_speed_kmh", "—") if totals.get("avg_speed_kmh") is not None else "—")

    st.subheader("🗂️ By Activity Type")
    st.dataframe(by_type, use_container_width=True)

    st.subheader("📅 Activities (past 7 days)")
    show_cols = ["date", "type", "name", "distance_km", "duration_min", "avg_hr", "max_hr", "avg_power", "max_power", "avg_speed_kmh"]
    st.dataframe(df_week[show_cols], use_container_width=True)

    # Charts (matplotlib, single-plot each, no explicit colors)
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
