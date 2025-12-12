import json
import pandas as pd
import pydeck as pdk
import streamlit as st
from pathlib import Path

STATIONS_CSV = "stations_with_land_only_coords.csv"
US_GEOJSON_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
    "master/data/geojson/us-states.json"
)

INFO_CSV = "ev_charging_patterns.csv"

TARGET_CITIES = [
    "Houston",
    "San Francisco",
    "Los Angeles",
    "Chicago",
    "New York",
]

st.set_page_config(page_title="EV Charging Stations", layout="wide")

@st.cache_data
def load_stations():
    df = pd.read_csv(STATIONS_CSV)

    required_cols = [
        "Charging Station Location",
        "Charger Type",
        "lat",
        "lon",
    ]

    #missing = [c for c in required_cols if c not in df.columns]
    #if missing:
    #    st.error(f"Missing required columns in CSV: {missing}")
    #    st.stop()

    # Filter to target cities
    df = df[df["Charging Station Location"].isin(TARGET_CITIES)]
    # Drop rows without coords
    df = df.dropna(subset=["lat", "lon"])
    return df

@st.cache_data
def load_info():
    df =pd.read_csv("ev_charging_patterns_new.csv")
    return df

@st.cache_data
def load_city_geojson():
    import requests

    STATE_GEOJSON_URLS = {
        "CA": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/ca.json",
        "NY": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/ny.json",
        "IL": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/il.json",
        "TX": "https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states/tx.json",
    }

    CITIES_BY_STATE = {
        "CA": {"San Francisco", "Los Angeles"},
        "NY": {"New York"},
        "IL": {"Chicago"},
        "TX": {"Houston"},
    }

    features = []

    for state, url in STATE_GEOJSON_URLS.items():
        data = requests.get(url).json()
        wanted = CITIES_BY_STATE[state]

        for feat in data.get("features", []):
            props = feat.get("properties", {})
            name = props.get("NAME") or props.get("name")
            if name in wanted:
                features.append(feat)

    return {
        "type": "FeatureCollection",
        "features": features,
    }


df = load_stations()
info_df=load_info()
city_geojson = load_city_geojson()

st.title("EV Charging Stations in Selected US Cities")

# ~~~SIDEBAR START~~~
st.sidebar.header("Filters")

select_city_sidebar = st.sidebar.selectbox(
    "Select City",
    options=sorted(df["Charging Station Location"].unique())
)

select_charger_types_sidebar= st.sidebar.multiselect(
    "Select Charger Types",
    options=sorted(df["Charger Type"].unique()),
    default=sorted(df["Charger Type"].unique()),
)

st.sidebar.subheader("Information on Hover")

with st.container(border=True):

    central_frequency = st.sidebar.radio(
        "Central Tendency",
        options=["Mean", "Median", "Min", "Max"],
        index=0,  # default = Mean Average,
        horizontal=True
    )

    options = ["Frequent User Type", "Charging Duration", "Energy Consumed", "Frequent Vehicle Model"]

    select_info = st.sidebar.pills("Select Information", options, selection_mode="multi")

point_size_label_sidebar = st.sidebar.radio(
        "Point Size",
        options=["Very Small", "Small", "Medium", "Large"],
        index=2,  # default = Medium,
        horizontal=True
    )
POINT_SIZE_MAP = {
    "Very Small": 10,
    "Small": 250,
    "Medium": 500,
    "Large": 2000,
}

point_radius = POINT_SIZE_MAP[point_size_label_sidebar]

CENTRAL_TENDENCY_MAP = {
    "Mean": "mean",
    "Median": "median",
    "Min": "min",
    "Max": "max",
}

agg_func = CENTRAL_TENDENCY_MAP[central_frequency]

# Numeric stats: duration + energy per station
numeric_stats = (
    info_df
    .groupby("station_id")[["Charging Duration (hours)", "Energy Consumed (kWh)"]]
    .agg(agg_func)
    .round(2)
)

# Frequent (mode) user type / vehicle model per station
user_mode = (
    info_df
    .groupby("station_id")["User Type"]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    .rename("Frequent User Type")
)

vehicle_mode = (
    info_df
    .groupby("station_id")["Vehicle Model"]
    .agg(lambda x: x.mode().iat[0] if not x.mode().empty else None)
    .rename("Frequent Vehicle Model")
)

# Combine all stats into one DataFrame
stats_per_station = (
    numeric_stats
    .join(user_mode)
    .join(vehicle_mode)
    .reset_index()
)

# ~~~SIDEBAR END~~~

df = df.merge(stats_per_station, on="station_id", how="left")

city_df = df[df["Charging Station Location"] == select_city_sidebar]

if select_charger_types_sidebar:
    df_filtered = city_df[city_df["Charger Type"].isin(select_charger_types_sidebar)].copy()
else:
    #keep map center from city_df if no charger type is selected
    df_filtered = city_df.iloc[0:0].copy()  # empty frame with same columns

# ~~~ SUMMARY START ~~~

mean_duration = df_filtered["Charging Duration (hours)"].mean()
mean_energy = df_filtered["Energy Consumed (kWh)"].mean()

mean_duration = round(mean_duration, 2) if pd.notna(mean_duration) else "—"
mean_energy = round(mean_energy, 2) if pd.notna(mean_energy) else "—"

with st.container():
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Charging Stations", len(df_filtered))
    col2.metric("Average Charging Duration per charging session (hrs)",mean_duration)
    col3.metric("Average Energy Consumed per charging session (kWh)", mean_energy)

st.subheader("Map of Charging Stations")

st.caption("Legend: 🔴 = DC Fast Charging  🟢 = Level 1 🔵 = Level 2")

# ~~~ SUMMARY END ~~~

# ~~~Map START~~~

# Background is the cities polygons
geo_layer = pdk.Layer(
    "GeoJsonLayer",
    data=city_geojson,
    pickable=False,
    stroked=True,
    filled=True,
    get_fill_color="[240, 240, 240, 120]",
    get_line_color="[120, 120, 120, 180]",
    line_width_min_pixels=1,
)

COLOR_MAP = {
    "DC Fast Charger": [220, 53, 69, 180],# red
    "Level 1": [25, 135, 84, 180],# green
    "Level 2": [13, 110, 253, 180],# blue
}

df_filtered["color"] = df_filtered["Charger Type"].apply(
    lambda x: COLOR_MAP.get(x, [160, 160, 160, 180])
)

station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_filtered,
    get_position="[lon, lat]",
    get_radius=point_radius,
    get_fill_color="color",
    pickable=True,
)

#Information

base_tooltip = """
<b>{Charging Station Location}</b><br/>
{Charger Type}
"""

tooltip_html = base_tooltip

if "Charging Duration" in select_info:
    tooltip_html += f"Charging Duration (hours): " \
                    "{Charging Duration (hours)}<br/>"

if "Energy Consumed" in select_info:
    tooltip_html += f"Energy Consumed (kWh): " \
                    "{Energy Consumed (kWh)}<br/>"

if "Frequent User Type" in select_info:
    tooltip_html += "Frequent User Type: {Frequent User Type}<br/>"

if "Frequent Vehicle Model" in select_info:
    tooltip_html += "Frequent Vehicle Model: {Frequent Vehicle Model}<br/>"

tooltip = {
    "html": tooltip_html,
    "style": {
        "backgroundColor": "rgba(0,0,0,0.8)",
        "color": "white",
    },
}

if not city_df.empty:
    center_lat = city_df["lat"].mean()
    center_lon = city_df["lon"].mean()
    zoom = 10  # close zoom for a city
else:
    center_lat = df["lat"].mean()
    center_lon = df["lon"].mean()
    zoom = 3.4

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=zoom,
    pitch=0,
)

# keeps GeoJSON visible
deck = pdk.Deck(
    layers=[geo_layer, station_layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style=None,
)

st.pydeck_chart(deck, use_container_width=True)

# ~~~Map END~~~

with st.expander("Show station data"):
    st.dataframe(df_filtered, use_container_width=True)





