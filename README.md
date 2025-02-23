# 📌 Overview
This Streamlit dashboard provides interactive visualizations and insights into weather trends and anomalies based on historical weather data. Users can explore temperature, humidity, precipitation, wind speed, and soil moisture trends across different time periods.

## 📊 Data Sources
The dashboard uses MERRA-2 reanalysis data, covering the following variables:

| Variable  | Description  |
|------------|------------|
| T2M      | Temperature at 2 Meters (°C)   |
| RH2M      | Relative Humidity at 2 Meters (%)     | 
| PRECTOTCORR      | Precipitation Corrected (mm/day)     | 
| WS2M      | Wind Speed at 2 Meters (m/s)     | 
| GWETTOP      | Surface Soil Wetness (index)     | 

## 🚀 Installation
Make sure you have Python installed (>=3.8). Then, follow these steps:
### Clone the repository
* git clone https://github.com/celine89/Omdn-AI-Farming.git
* cd Omdn-AI-Farming

### Create a virtual environment
* `python -m venv venv`
* On Windows use `venv\Scripts\activate`

### Install dependencies
`pip install -r requirements.txt`

## 📂 Adding Data
You can add your datasets in two ways:
* Place datasets in the raw_data/ folder before running the app.
* Upload datasets directly through the running application.

## 🏃‍♂️ Running the Dashboard
After installing dependencies, start the Streamlit app with:
* `streamlit run weather_dashboard.py`
