# Global EV Charging Station Analysis

A comprehensive data analysis project aimed at **optimising the global Electric Vehicle (EV) charging network**. This project integrates four key business streams— **Infrastructure Mapping**, **Pricing Models**, **Availability Analysis**, and **Vehicle Economics**—to provide actionable insights for customers and network planners.

## 1. Project Overview

**Core Question:** How can we data driven insights to optimise the EV charging network and facilitate optimised pricing, station utilisation, and vehicle selections for diverse user groups?

This project analyses real-world charging sessiondata and 2025 ev car market data to answer this question. We have built a centralised dashboard that allows users to:

* **Find Stations:** View charger clusters and stats on an interactive map.
* **Detail Costs:** Estimate session prices based on time, location, and charger type.
* **Check Availability:** Check for most available charge points.
* **Compare Vehicles:** Calculate the "Cheapest Mile" running cost for specific car models.

> **Note:** The interactive dashboard file is called 'ev_charge.py'. Details on how to view and interact with the dashboard are shown below

---

## 2. Getting Started and running the dashboard

Instructions on how to get a copy of the project up and running on a local machine for development and testing purposes.

### Prerequisites

List of software and dependencies required to run the project notebooks and scripts:

* Jupyter Notebook or VS Code (for running `.ipynb` files)
* Python (version 3.10 or higher)
* Pandas
* Streamlit
* Seaborn
* Plotly
* Base64
* Pydeck
* Pathlib
* Matplotlib

### Installation

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/M-3rlin/GlobalEVChargingStationsAnalysis.git](https://github.com/M-3rlin/GlobalEVChargingStationsAnalysis.git)
    ```

2.  **Install packages:**
    ```bash
    pip install -r requirements.txt
    ```
    > **Note:** Ensure you create a `requirements.txt` file if one does not exist.

3. Open ev_charge.py file in VS Code
4. Open terminal and type: run streamlit ev_charge.py
5. Explore dashboard functionalities in your local browser
---

## 3. Key Features & Analysis Streams

Our analysis is divided into four specialised streams, each led by a team member:

### Stream 1: Mapping (Lead: Raphael)
* **Objective:** Visualise infrastructure gaps.
* **Key Feature:** Interactive geospatial mapping of stations including charger types and utilisation metrics.
* **Visuals:** Clustered map layers, Hover-over station tooltips.

### Stream 2: Pricing Models (Lead: Swathi)
* **Objective:** Understand cost drivers and predict session prices.
* **Key Feature:** Random Forest Regression model to estimate charging costs based on charger type (DC Fast vs. Level 2) and time of day.
* **Visuals:** Price distribution bar charts, scatter plot, box plots and correlation heatmaps.

### Stream 3: Availability (Lead: Arphaxad)
* **Objective:** Optimise station utilisation.
* **Key Feature:** Temporal analysis identifying peak usage windows (e.g., Weekday Evenings) and "charging deserts."
* **Visuals:** Utilisation heatmaps, peak time bar charts and line graphs.

### Stream 4: Cars & Economics (Lead: Abdul)
* **Objective:** Analyse vehicle efficiency and running costs.
* **Key Feature:** **"Cheapest Mile Predictor"** - A hybrid model using 2025 spec data and historical user logs to predict the cost-per-km for specific vehicles in specific cities.
* **Visuals:** Efficiency vs. Price scatter plots, Body Type comparisons (SUV vs. Sedan).
---

## 5. Team Members

This project was a collaborative effort by:

* **Sam** (Dashboard Integration & Project Management)
* **Abdul** (Cars Stream Analysis)
* **Swathi** (Pricing Stream Analysis)
* **Arphaxad** (Availability Analysis)
* **Raphael** (Mapping Analysis)

## 6. Data sources and credits:

* Ev charging patterns https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns
* EV car market [Abdul insert]
* Mapping geosojn files https://raw.githubusercontent.com/generalpiston/geojson-us-city-boundaries/master/states
* Chat GPT was used to help build the dashboard, in particular for writing CSS and HTML code blocks and debugging python code blocks
* Stackoverflow and Github repos assisted with mapping functionalities, image and GiF loading, and other formatting support
