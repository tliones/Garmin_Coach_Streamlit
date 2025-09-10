import os
from datetime import datetime, timedelta
import pytz
import pandas as pd
import streamlit as st

# Community library (unofficial Garmin API)
try:
    from garminconnect import Garmin, GarminConnectAuthenticationError, GarminConnectConnectionError
except Exception:
    st.error("The 'garminconnect' package is required. Please install it with: pip install garminconnect")
    st.stop()

st.set_page_config(page_title="Garmin Weekly Summary", page_icon="⏱️", layout="wide")
st.title("🏃‍♀️ Garmin Weekly Summary (Unofficial)")
st.caption("Logs in to Garmin, pulls your past week's activities, and summarizes them.")

# --- Garmin Credentials Input ---
st.subheader("🔐 Garmin Login")

email = st.text_input("Garmin email", value="", autocomplete="username")
password = st.text_input("Garmin password", value="", type="password", autocomplete="current-password")
twofa_code = st.text_input("2FA code (if prompted by Garmin)", type="password")

colA, colB, colC = st.columns([1,1,1])
with colA:
    tz_name = st.selectbox("Time Zone", ["America/Chicago", "UTC", "America/New_York", "America/Los_Angeles"], index=0)
with colB:
    end_dt = datetime.now(pytz.timezone(tz_name)).replace(microsecond=0, second=0)
    start_dt = (end_dt - timedelta(days=7)).replace(microsecond=0, second=0)
    st.write(f"**Range:** {start_dt.strftime('%Y-%m-%d %H:%M')} → {end_dt.strftime('%Y-%m-%d %H:%M')}")
with colC:
    go = st.button("🔄 Fetch past week")

@st.cache_data(show_spinner=False)
def login_and_fetch(email, password, twofa_code, start_dt_iso, end_dt_iso):
    garmin = Garmin(email, password)
    try:
        garmin.login()
    except GarminConnectAuthenticationError:
        if twofa_code:
            garmin.login()
        else:
            raise

    try:
        activities = garmin.get_activities_by_date(start_dt_iso, end_dt_iso)
    except Exception:
        activities = garmin.get_activities(0, 200)
        activities = [a for a in activities if start_dt_iso <= a.get("startTimeLocal", a.get("startTimeGMT", ""))[:10] <= end_dt_iso]

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
            "avg_speed_kmh": (avg_speed*3.6) if isinstance(avg_speed, (int, float)) else None,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("date", ascending=False)
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

    with st.spinner("Logging in and fetching your past week's activities..."):
        start_iso = start_dt.strftime("%Y-%m-%d")
        end_iso = end_dt.strftime("%Y-%m-%d")
        try:
            activities = login_and_fetch(email, password, twofa_code, start_iso, end_iso)
        except GarminConnectAuthenticationError:
            st.error("Authentication failed. If you have 2FA enabled, try adding the code above and re-fetch.")
            st.stop()
        except GarminConnectConnectionError:
            st.error("Connection error contacting Garmin. Please try again.")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()

    df = normalize_activities(activities)
    if df.empty:
        st.info("No activities found in the past week.")
        st.stop()

    mask = (df["date"] >= pd.to_datetime(start_dt)) & (df["date"] <= pd.to_datetime(end_dt))
    df_week = df.loc[mask].copy()

    totals, by_type = summarize(df_week)

    st.subheader("📊 Weekly Totals")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Activities", totals.get("activities", 0))
    c2.metric("Time (hr)", totals.get("total_time_hr", 0.0))
    c3.metric("Distance (km)", totals.get("total_distance_km", 0.0))
    c4.metric("Avg HR", totals.get("avg_hr", "—"))
    c5.metric("Avg Speed (km/h)", totals.get("avg_speed_kmh", "—"))

    st.subheader("🗂️ By Activity Type")
    st.dataframe(by_type, use_container_width=True)

    st.subheader("📅 Activities (past 7 days)")
    show_cols = ["date", "type", "name", "distance_km", "duration_min", "avg_hr", "avg_speed_kmh"]
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

    st.download_button("⬇️ Download weekly activities (CSV)",
        df_week.to_csv(index=False).encode("utf-8"),
        file_name="garmin_weekly_activities.csv",
        mime="text/csv")
else:
    st.info("Enter your Garmin credentials and click **Fetch past week** to begin.")
