# Environments START
# #=============================================================================================
import json
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import plotly.express as px
import base64
import pydeck as pdk
from pathlib import Path
import matplotlib.pyplot as plt

# For Regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# For Classification
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
#=============================================================================================
# Environments END

# Loading images and GIFs START 
#=============================================================================================
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_b64 = get_base64_image("ev_charge2.png")

st.sidebar.image("logo.png", use_container_width=True)

st.markdown(
    f"""
    <style>
    .image-container {{
        position: relative;
        width: 100%;
    }}

    .banner-img {{
        width: 100%;
        display: block;
    }}

    </style>

    <div class="image-container">
        <img src="data:image/jpg;base64,{banner_b64}" class="banner-img">
    </div>
    """,
    unsafe_allow_html=True
)
#=============================================================================================
#Images and GIFs END 

# Page configurations and settings START
#=============================================================================================
st.set_page_config(page_title="EV Analysis", page_icon="🚗", layout="wide")

st.markdown(
    """
    <h1 style="color:#007700; font-weight:bold;">
        Electric Vehicles (EVs) and Charge Point Analysis
    </h1>
    <p style="color:#007700; font-weight:bold; font-size:1.1rem;">
        Plotting your next road trip with the nearest, cheapest and fastest EV charge point 
    </p>
    """,
    unsafe_allow_html=True
)
# Custom styling
page_style = """
<style>
/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #145A32 !important;
}

/* Sidebar inner content */
[data-testid="stSidebar"] > div:first-child {
    background-color: #145A32 !important;
}

/* Sidebar text color */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* Main page background */
[data-testid="stAppViewContainer"] {
    background-color: #E9FFE9 !important;
}

</style>

"""
st.markdown(page_style, unsafe_allow_html=True)
#=============================================================================================
# Page settings END 

#=============================================================================================
# Loading datasets START

stations_csv = "stations_with_land_only_coords.csv"
US_geoson_url = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/"
    "master/data/geojson/us-states.json"
)

@st.cache_data
def load_data():
    df = pd.read_csv("ev_charging_patterns.csv")
    return df

df = load_data()

# Load dataset 2
@st.cache_data
def load_data():
    df_market = pd.read_csv("electric_vehicles_spec_2025.csv")
    return df_market

df_market = load_data()

# Loading individual station coords  (3)
@st.cache_data
def load_stations():
    df3 = pd.read_csv(stations_csv)

    required_cols = [
        "Charging Station Location",
        "Charger Type",
        "lat",
        "lon",
    ]

    # Filter to target cities
    target_cities = [
    "Houston",
    "San Francisco",
    "Los Angeles",
    "Chicago",
    "New York",
]
    
    df3 = df3[df3["Charging Station Location"].isin(target_cities)]
    # Drop rows without coords
    df3 = df3.dropna(subset=["lat", "lon"])
    return df3

@st.cache_data # Load Geoson file - URL
def load_us_geojson():
    import requests
    return requests.get(US_geoson_url).json()

df3 = load_stations()
us_geojson = load_us_geojson()

# Load dataset 4 - Used for Availability and Cars Tabs
@st.cache_data
def load_data():
    df_usage = pd.read_csv('clean_ev_data.csv')
    return df_usage

df_usage = load_data()
#=============================================================================================
# Data uploads END

# Sidebar filters START 
#=============================================================================================
st.sidebar.header("Apply filters to explore charge points")

# Sidebar container
with st.sidebar:
    # Use st.radio label instead of separate markdown
    select_city_sidebar = st.radio(
        label="Select City",  # Label is now attached to radio
        options=sorted(df3["Charging Station Location"].unique()),
        index=0,  # default selection
        horizontal=False
    )

# Use the selected city
st.write("Selected city:", select_city_sidebar)

select_charger_types_sidebar= st.sidebar.multiselect(
    "Select Charger Types",
    options=sorted(df3["Charger Type"].unique()),
    default=sorted(df3["Charger Type"].unique()),
)

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

city_df = df3[df3["Charging Station Location"] == select_city_sidebar]

if select_charger_types_sidebar:
    df_filtered = city_df[city_df["Charger Type"].isin(select_charger_types_sidebar)].copy()
else:
    #keep map center from city_df if no charger type is selected
    df_filtered = city_df.iloc[0:0].copy()  # empty frame with same columns
#=============================================================================================
# Sidebar Filters END

# General Feature Engineering START
#=============================================================================================
# df 1
# Create Unique IDs
df["Charging Station Location"] = df["Charging Station Location"].astype(str)
df["User ID"] = df["User ID"].astype(str)
df["Unique ID"] = df["Charging Station Location"] + "_" + df["User ID"]

# Create Unit Cost $/kWh and drop outliers
df['Cost Per kWh ($)'] = df['Charging Cost (USD)'].astype(float)/df['Energy Consumed (kWh)'].astype(float)
# Ensure numeric
df['Cost Per kWh ($)'] = pd.to_numeric(df['Charging Cost (USD)'], errors='coerce') / \
                         pd.to_numeric(df['Energy Consumed (kWh)'], errors='coerce')
# Calculate Q1, Q3 and IQR
Q1 = df['Cost Per kWh ($)'].quantile(0.25)
Q3 = df['Cost Per kWh ($)'].quantile(0.75)
IQR = Q3 - Q1

# Filter out major outliers
df = df[(df['Cost Per kWh ($)'] >= (Q1 - 5 * IQR)) & 
                 (df['Cost Per kWh ($)'] <= (Q3 + 5 * IQR))]

# df 2 (renamed df_market)
# Data Engineering
# Estimate price based on segment (since 2025 data lacks price)
def estimate_price(segment):
    if pd.isna(segment): return 45000
    if 'A -' in segment or 'B -' in segment: return 30000  # Economy
    if 'C -' in segment or 'D -' in segment: return 48000  # Mid-Range
    if 'E -' in segment or 'F -' in segment: return 85000  # Premium
    if 'S -' in segment: return 75000 # Sports
    return 55000 

df_market['Estimated_Price'] = df_market['segment'].apply(estimate_price)

# Simplify segments
def simplify_segment(segment):
    if pd.isna(segment): return 'Mid-Range'
    if 'A -' in segment or 'B -' in segment: return 'Economy'
    if 'E -' in segment or 'F -' in segment or 'S -' in segment: return 'Premium'
    return 'Mid-Range'

df_market['Category'] = df_market['segment'].apply(simplify_segment)

# Standardise Car Body Type 
def simplify_body(body):
    if pd.isna(body): return 'Other'
    if 'SUV' in body: return 'SUV'
    if 'Sedan' in body: return 'Sedan'
    if 'Hatchback' in body: return 'Hatchback'
    return 'Other'

df_market['Body_Type'] = df_market['car_body_type'].apply(simplify_body)

#=============================================================================================
# General Feature Engineering END

# Headline Summary Stats START
#=============================================================================================
with st.container():
    st.subheader("Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Get filtered values from df_filtered
        selected_cities = df_filtered["Charging Station Location"].unique()
        selected_charger_types = df_filtered["Charger Type"].unique()

        # Apply those filters to the full df
        df_metric = df[
            (df["Charging Station Location"].isin(selected_cities)) &
            (df["Charger Type"].isin(selected_charger_types))
        ]

        # Count unique charging stations
        total_stations = df_metric["Unique ID"].nunique()

        st.metric("Total Charging Stations Selected", total_stations)

    with col2:
        # Calculate average cost per unit for filtered sessions
        avg_cost_per_kwh = df_metric["Cost Per kWh ($)"].mean()
        # Display metric (rounded to 2 decimal places)
        st.metric("Average Cost Per Unit ($/kWh)", f"${avg_cost_per_kwh:.2f}")
    with col3:
        # Calculate average spend per session for filtered data
        avg_spend_per_session = df_metric["Charging Cost (USD)"].mean()
        # Display metric (2 decimal places)
        st.metric("Average Spend Per Session ($)", f"${avg_spend_per_session:.2f}")
    with col4:
        # Calculate average kWh consumed per session for filtered data
        avg_kwh_per_session = df_metric["Energy Consumed (kWh)"].mean()
        # Display metric (2 decimal places)
        st.metric("Average Energy Per Session (kWh)", f"{avg_kwh_per_session:.0f} (kWh)")
#=============================================================================================
# Headline Summary Stats END

# Create tabs START
#=============================================================================================
tab1, tab2, tab3, tab4 = st.tabs(["Map Plot", "Pricing Analysis", "Availability Analysis", "EV Car Market"])
#=============================================================================================
# Create tabs END

# Tab 1 Mapping START
#=============================================================================================
with tab1:
   
    # Map
    st.subheader("Charging Stations ")

    # Background is the US polygons
    geo_layer = pdk.Layer(
        "GeoJsonLayer",
        data=us_geojson,
        pickable=False,
        stroked=True,
        filled=True,
        get_fill_color="[240, 240, 240, 120]",
        get_line_color="[120, 120, 120, 180]",
        line_width_min_pixels=1,
    )

    COLOR_MAP = {
        "DC Fast Charger": [220, 53, 69, 180],   # red
        "Level 1": [25, 135, 84, 180],          # green
        "Level 2": [13, 110, 253, 180],         # blue
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

    tooltip = {
        "html": """
        <b>{Charging Station Location}</b><br/>
        Charger: {Charger Type}<br/>
        """,
        "style": {"backgroundColor": "rgba(0,0,0,0.8)", "color": "white"},
    }

    if not city_df.empty:
        center_lat = city_df["lat"].mean()
        center_lon = city_df["lon"].mean()
        zoom = 10  # close zoom for a city
    else:
        # Safe fallback (should rarely happen)
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

    with st.expander("Show station data"):
        st.dataframe(df_filtered, use_container_width=True)

#=============================================================================================
# Tab 1 END

# Tab 2 START
#=============================================================================================
with tab2:
    # Basic info
    st.header("Plot The Cheapest Charge Points")
    st.markdown("""
    The scatter plot below shows the **average cost per unit ($/kWh)** by **charger type** over the course of a typical day. The points are split into:

    - **DC Fast Charger (Blue)**
    - **AC Level 1 (Orange)**
    - **AC Level 2 (Green)**
    
    DC Charging tends to be the more expensive given it is the fastest and also requires a **'converter'** to convert **grid electricity (AC) to car battery electricity (DC)**. Most cars also have an inbuilt converter meaning they can switch between charging options.
    """)
    selected_cities = df_filtered["Charging Station Location"].unique()
    df_tab2 = df[df["Charging Station Location"].isin(selected_cities)].copy()

    if df_tab2.empty:
        st.info("No data available to plot Cost per kWh by time of day.")
    else:
        # Ensure necessary columns are numeric/datetime
        df_tab2["Charging Start Time"] = pd.to_datetime(df_tab2["Charging Start Time"], errors="coerce")
        df_tab2["Cost Per kWh ($)"] = pd.to_numeric(df_tab2["Cost Per kWh ($)"], errors="coerce")

        # Filter by selected charger types
        df_plot = df_tab2[df_tab2["Charger Type"].isin(select_charger_types_sidebar)]

        # Extract time of day as decimal hours
        df_plot["Hour of Day"] = df_plot["Charging Start Time"].dt.hour + df_plot["Charging Start Time"].dt.minute / 60

        # Keep only valid hours (0-24)
        df_plot = df_plot[df_plot["Hour of Day"].between(0, 24)]

    if df_plot.empty:
        st.info("No valid data after filtering by charger type and time of day.")
    else:
        # Define custom colors
        color_map = {
            "Level 1": "orange",
            "Level 2": "green",
            "DC Fast Charger": "lightblue"
        }

        # Build title including filtered city/cities
        if len(selected_cities) == 1:
            plot_title = f"{selected_cities[0]} – Cost per Unit ($/kWh) by Charger Type"
        else:
            plot_title = f"Cost per Unit ($/kWh) by Charger Type – Multiple Cities"

        # Create scatter plot with trend lines
        fig_scatter = px.scatter(
            df_plot,
            x="Hour of Day",
            y="Cost Per kWh ($)",
            color="Charger Type",
            color_discrete_map=color_map,
            hover_data=["Charging Station Location", "User ID", "Charging Start Time"],
            trendline="ols",  # Add trend line
            trendline_scope="group",  # Separate trend line per charger type
            opacity=0.5,
            title=plot_title
        )

        # Make markers slightly larger for clarity
        fig_scatter.update_traces(marker=dict(size=25), selector=dict(mode='markers'))

        # Set x-axis to strictly 0-24 hours and y-axis title
        fig_scatter.update_layout(
            xaxis=dict(title="Hour of Day", range=[0, 24], tickmode="linear", dtick=1),
            yaxis=dict(title="Cost Per kWh ($)"),
            title=dict(x=0.25)  # Center the title
        )

        # Display the plot
        st.plotly_chart(fig_scatter, use_container_width=True)

    if df_tab2.empty:
        st.info("No data available for the selected city/cities.")
    else:
        # Aggregate: average cost per charger type and total kWh consumed
        summary_table = df_tab2.groupby("Charger Type", as_index=False).agg(
            **{
                "Average Cost per kWh ($)": ("Cost Per kWh ($)", "mean"),
                "Total kWh Used (kWh)": ("Energy Consumed (kWh)", "sum")
            }
        )

        # round numeric values for readability
        summary_table["Average Cost per kWh ($)"] = summary_table["Average Cost per kWh ($)"].round(2)
        summary_table["Total kWh Used (kWh)"] = summary_table["Total kWh Used (kWh)"].round(0)

        # Display in an expander without index
        with st.expander("Average Cost ($/kWh) By Charger Type"):
            st.dataframe(summary_table)  # as_index=False ensures no index is added
    
    if df_tab2.empty:
        st.info("No data available for the selected city/cities.")
    else:
        # Aggregate: average cost and total kWh by Charging Station ID and Charger Type
        summary_table = df_tab2.groupby(["Charging Station ID", "Charger Type"], as_index=False).agg(
            **{
                "Average Cost per kWh ($)": ("Cost Per kWh ($)", "mean"),
                "Total kWh Used (kWh)": ("Energy Consumed (kWh)", "sum")
            }
        )

        # round numeric values for readability
        summary_table["Average Cost per kWh ($)"] = summary_table["Average Cost per kWh ($)"].round(2)
        summary_table["Total kWh Used (kWh)"] = summary_table["Total kWh Used (kWh)"].round(0)

        # Display in an expander
        with st.expander("Average Cost ($/kWh) by Station ID"):
            st.dataframe(summary_table)
    st.markdown("---")

    col1, col2  = st.columns(2)

    with col1:
        st.write("##### Expenditure by Charge Type")
        # Filter main df using selected cities

        if df_tab2.empty:
            st.info("No matching data for selected filters.")
        else:
            # Ensure Charging Cost is numeric
            df_tab2["Charging Cost (USD)"] = pd.to_numeric(df_tab2["Charging Cost (USD)"], errors="coerce")
            
            # Aggregate total cost per Charger Type
            expenditure = df_tab2.groupby("Charger Type")["Charging Cost (USD)"].sum().reset_index()

            # Define chart title
            if len(selected_cities) == 1:
                chart_title = f"{selected_cities[0]} – Expenditure by Charger Type"
            else:
                chart_title = "Multiple Cities – Expenditure by Charger Type"

            # Define custom colors
            color_map = {
                "Level 1": "orange",        # Sun orange
                "Level 2": "green",         # Forest green
                "DC Fast Charger": "lightblue"  # Light blue
            }

            # Plot pie chart
            fig = px.pie(
                expenditure,
                names="Charger Type",
                values="Charging Cost (USD)",
                title=chart_title,
                hole=0.3,
                color="Charger Type",
                color_discrete_map=color_map
            )

            st.plotly_chart(fig, use_container_width=True)
        st.markdown("#### This pie chart shows the distribution of customer spending across different charger types.")
     
    with col2:
        st.markdown("##### Total Number of Installations")

        if df_tab2.empty:
            st.info("No data available to plot charger installation distribution.")
        else:
            # Count number of chargers by type within each city
            charger_counts = df_tab2.groupby(
                ["Charging Station Location", "Charger Type"]
            ).size().reset_index(name="Number of Chargers")

            # Define chart title
            if len(selected_cities) == 1:
                chart_title2 = f"{selected_cities[0]} – Number of Charge Points Installed"
            else:
                chart_title2 = "Multiple Cities – Number of Charge Points Installed"

            # Define custom colors
            color_map = {
                "Level 1": "orange",        # Sun orange
                "Level 2": "green",         # Forest green
                "DC Fast Charger": "lightblue"  # Light blue
            }

            # Plot grouped bar chart
            fig_bar = px.bar(
                charger_counts,
                x="Charging Station Location",
                y="Number of Chargers",
                color="Charger Type",
                barmode="group",  # use "stack" if you want stacked bars
                color_discrete_map=color_map,
                title=chart_title2,
                text="Number of Chargers"
            )

            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.markdown(
                "#### This bar chart shows how many charging stations of each type are installed"
            )
#=============================================================================================
# Tab 2 END

# Tab 3 START
#=============================================================================================
#=============================================================================================
# Tab 3 END


# Tab 4 START
#=============================================================================================
    with tab4:
        st.header("EV Car Market Analysis")
            # Show dataframe inside an expander
        unique_models = df_market["model"].nunique()
        st.markdown(f"""
                    In this section we analyse **{unique_models}** number of car models to help you buy the right model with insights in:
                    - Efficiency
                    - Cost
                    - Range
                    - Charging capabilities

                    The dataframe below is the data we evaluate in this app and can be downloaded onto your computer for further analysis
                    """)

        with st.expander("View EV Market Data"):
            st.dataframe(df_market, use_container_width=True)
            
        st.markdown("""
        The below box plot shows the efficiency of cars by body type (km/kWh - distance per unit of energy)
        and shows **hatchback cars** clearly have the **highest efficiency per kilometre** and are therefore the cheapest to run
        """)

        #Box plot body type vs. efficiency per km
        # Convert efficiency (Wh/km -> km/kWh)
        df_market["Efficiency_km_kWh"] = 1000 / df_market["efficiency_wh_per_km"]

        # Filter to main types
        main_types = df_market[df_market["Body_Type"].isin(["SUV", "Sedan", "Hatchback"])]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(
            data=main_types,
            x="Body_Type",
            y="Efficiency_km_kWh",
            hue="Body_Type",
            palette="Set2",
            legend=False,
            ax=ax
        )

        ax.set_title("Real-World Efficiency by Car Body Type")
        ax.set_ylabel("Efficiency (km per kWh)")

        # Display in Streamlit
        st.pyplot(fig)

        # Scatterplot showing efficiency vs. cost
        st.markdown("""
        The below scatter plot shows the efficiency of cars vs. cost (km/kWh - distance per unit of energy)
        interestingly we can see the low cost cars also tend to have the **highest efficiency per kilometre**
        """)
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Scatter plot
        sns.scatterplot(
            data=df_market,
            x='Efficiency_km_kWh',
            y='Estimated_Price',
            hue='Category',       # Color by segment
            style='Body_Type',    # Different marker for body type
            palette='viridis',
            s=100,
            alpha=0.8,
            ax=ax
        )

        # Title and labels
        ax.set_title("Market Efficiency: Price vs. Efficiency (by Segment & Type)")
        ax.set_xlabel("Efficiency (km per kWh)")
        ax.set_ylabel("Estimated Price ($)")

        # Place legend outside the plot
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

        # Display in Streamlit
        st.pyplot(fig)

        #Scatterplot showing charging capability vs. price
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Scatter plot
        sns.scatterplot(
            data=df_market,
            x='Estimated_Price',
            y='fast_charging_power_kw_dc',
            hue='Category',
            palette='magma',
            s=100,
            ax=ax
        )

        # Titles and labels
        ax.set_title("Charging Capability: Price vs. Max DC Charging Speed")
        ax.set_xlabel("Estimated Car Price ($)")
        ax.set_ylabel("Max DC Charging Speed (kW)")

        # Add threshold line
        ax.axhline(
            y=150,
            color='r',
            linestyle='--',
            label="High Speed Threshold (150 kW)"
        )

        # Place legend
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)

        # Tight layout for better spacing
        plt.tight_layout()

        # Display plot in Streamlit
        st.pyplot(fig)
#=============================================================================================
# Tab 4 END

# Footer
st.markdown("---")
st.markdown("**©** Copyright protected by The EV Plot est. 2026 (time-travel is the surest way to be ahead of your time)")

# Row container with quote on the right
def get_gif_base64(file_path):
    with open(file_path, "rb") as f:
        content = f.read()
        return base64.b64encode(content).decode("utf-8")

# Path to your local GIF
gif_path = 'charging_gif.gif'
gif_data = get_gif_base64(gif_path)

st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between; width: 100%; margin-top: 20px;">
        <img src="data:image/gif;base64,{gif_data}" alt="Animated GIF" width="300">
        <div style="color: #007700; font-size: 32px; font-weight: bold; margin-left: 20px; text-align: right;">
            "The only bad thing about EVs (aside from Elon Musk)<br>is finding the cheapest, nearby charging station"<br>– Anon
        </div>
    </div>
    """,
    unsafe_allow_html=True
)