import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

class WeatherEDA:
    def __init__(self, file_path):
        # Load dataset
        self.df = self.load(path=file_path)

        # Extract filename (without extension) for titles and file naming
        try:
            self.dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        except:
            self.dataset_name = file_path.name
        # Create output directories
        # self.output_dirs = {
        #     "trends": f"plots/trends/",
        #     "seasons": f"plots/seasons/",
        #     "scatter": f"plots/scatter/",
        # }
        # for dir in self.output_dirs.values():
        #     os.makedirs(dir, exist_ok=True)

    @staticmethod
    def assign_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Autumn"
        
    def load(self,path):    
        try:        
            df = pd.read_csv(path, header=20)
            if df.isnull().any().any(): 
                print(f"⚠ Potential issue with header in {path}, check data manually.")

            # Ensure essential columns exist
            required_columns = {"YEAR", "DOY"}
            if not required_columns.issubset(df.columns):
                raise ValueError(f"Missing required columns in {path}")

            # Check for missing values in YEAR or DOY
            if df["YEAR"].isnull().any() or df["DOY"].isnull().any():
                raise ValueError(f"Missing YEAR or DOY values in {path}")

            # Date feature engineering
            df["date"] = df["YEAR"].astype(str) + "-" + df["DOY"].astype(str)
            df["date"] = pd.to_datetime(df["date"], format="%Y-%j")

            # Set date as index
            df.set_index("date", inplace=True)
            #df.drop(["YEAR", "DOY"], axis=1, inplace=True)
            df["month"] = df.index.month
            # Assign seasons
            df["season"] =df["month"].apply(self.assign_season)
            # Calculate moving averages
            df["T2M_7d_avg"] = df["T2M"].rolling(7).mean()
            df["T2M_30d_avg"] = df["T2M"].rolling(30).mean()

            # Rename precipitation column if it exists
            if "PRECTOTCORR" in df.columns:
                df.rename(columns={"PRECTOTCORR": "PRECIPITATION"}, inplace=True)

            print(f"📌 Successfully processed {path}: {df.shape}")
            return df
        
        except Exception as e:
            print(f"❌ Error processing {path}: {e}")
            return None

    # Plot Time Series
    def plot_time_series(self, column, title):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.df[column], label=title, color="blue")
        ax.set_title(f"{column} Over Time - {self.dataset_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{column}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    # Seasonal Decomposition
    def plot_seasonal_decomposition(self, column_name):
        decomposition = seasonal_decompose(self.df[column_name].dropna(), model="additive", period=365)
        
        fig, axs = plt.subplots(4, 1, figsize=(12, 8))
        
        axs[0].plot(self.df[column_name], label="Original", color="blue")
        axs[0].legend()
        
        axs[1].plot(decomposition.trend, label="Trend", color="orange")
        axs[1].legend()
        
        axs[2].plot(decomposition.seasonal, label="Seasonality", color="green")
        axs[2].legend()
        
        axs[3].plot(decomposition.resid, label="Residuals", color="red")
        axs[3].legend()
        
        plt.suptitle(f"Seasonal Decomposition of {column_name} - {self.dataset_name}")
        st.pyplot(fig)

    def plot_yearly_trends(self, columns):
        yearly_avg = self.df.groupby("YEAR")[columns].mean()
        # Normalize data using Min-Max Scaling
        scaler = MinMaxScaler()
        yearly_avg_scaled = pd.DataFrame(scaler.fit_transform(yearly_avg), columns=yearly_avg.columns, index=yearly_avg.index)

        fig, ax = plt.subplots(figsize=(12, 6))
        
        for col in yearly_avg_scaled.columns:
            ax.plot(yearly_avg_scaled.index, yearly_avg_scaled[col], label=col, marker="o")

        ax.legend()
        ax.set_title(f"Normalized Yearly Weather Trends (2014-2024)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Normalized Value (0-1)")
        ax.grid(True)
        st.pyplot(fig)

    # Z-Score Anomaly Detection
    def detect_anomalies_zscore(self, threshold=3):
        self.df["zscore"] = zscore(self.df["T2M"])
        self.df["anomaly_zscore"] = self.df["zscore"].abs() > threshold

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df["T2M"], label="Temperature", color="blue", alpha=0.6)
        ax.scatter(self.df[self.df["anomaly_zscore"]].index, self.df[self.df["anomaly_zscore"]]["T2M"], color="red", label="Anomalies", s=50)
        ax.set_title(f"Anomalies Detected via Z-Score")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Isolation Forest Anomaly Detection
    def detect_anomalies_isolation_forest(self, contamination=0.01):
        model = IsolationForest(contamination=contamination, random_state=42)
        self.df["anomaly_iforest"] = model.fit_predict(self.df[["T2M"]])
        self.df["anomaly_iforest"] = self.df["anomaly_iforest"] == -1  # Convert to boolean

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df["T2M"], label="Temperature", color="blue", alpha=0.6)
        ax.scatter(self.df[self.df["anomaly_iforest"]].index, self.df[self.df["anomaly_iforest"]]["T2M"], color="red", label="Anomalies", s=50)
        ax.set_title(f"Anomalies Detected via Isolation Forest")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)


    def plot_correlation_heatmaps(self, columns):
        """Plots and saves side-by-side Pearson & Spearman correlation heatmaps."""
        pearson_corr = self.df[columns].corr(method="pearson")
        spearman_corr = self.df[columns].corr(method="spearman")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axes[0])
        axes[0].set_title(f"Pearson Correlation")

        sns.heatmap(spearman_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=axes[1])
        axes[1].set_title(f"Spearman Correlation")
        st.pyplot(fig)
    
    def monthly_variable_trends(self, columns):
        monthly_avg = self.df.groupby("month")[columns].mean()
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(monthly_avg)

        # Create a DataFrame for normalized data
        normalized_df = pd.DataFrame(normalized_data, columns=monthly_avg.columns, index=monthly_avg.index)
        fig, ax = plt.subplots(figsize=(14, 6))
        # Plot the heatmap for normalized data
        plt.figure(figsize=(10, 6))
        sns.heatmap(normalized_df.T, cmap="coolwarm", annot=True, fmt=".2f")
        ax.set_title("Normalized Monthly Weather Variable Trends")
        ax.set_xlabel("Month")
        ax.set_ylabel("Weather Variable")
        st.pyplot(plt)

    def plot_seasonal_trends(self):
        """Generates seasonal bar plots and displays them in Streamlit."""

        # Aggregate seasonal data for four measures
        seasonal_rain = self.df.groupby("season")["PRECIPITATION"].sum().reset_index()
        seasonal_temp = self.df.groupby("season")["T2M"].mean().reset_index()
        seasonal_humidity = self.df.groupby("season")["RH2M"].mean().reset_index()
        seasonal_wind = self.df.groupby("season")["WS2M"].mean().reset_index()

        # Define seasonal metrics for looping
        metrics = [
            ("Total Precipitation per Season", seasonal_rain, "PRECIPITATION", "Blues"),
            ("Average Temperature per Season", seasonal_temp, "T2M", "Reds"),
            ("Average Humidity per Season", seasonal_humidity, "RH2M", "Greens"),
            ("Average Wind Speed per Season", seasonal_wind, "WS2M", "Purples"),
        ]

        st.subheader(f"Seasonal Weather Trends")

        # Create two columns for side-by-side plots
        col1, col2 = st.columns(2)

        for idx, (title, data, y_col, palette) in enumerate(metrics):
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(x="season", y=y_col, data=data, palette=palette, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Season")
            ax.set_ylabel(y_col)
            if idx % 2 == 0:
                col1.pyplot(fig)
            else:
                col2.pyplot(fig)

# Streamlit Dashboard
def main():
    st.title("🌍 Weather Data EDA Dashboard")
    # Help section using Expander
    with st.expander("Measurement Definitions"):
        st.markdown("""
    **T2M**: Temperature at 2 Meters (°C) — The air temperature measured at 2 meters above the Earth's surface.

    **T2M_MAX**: Temperature at 2 Meters Maximum (°C) — The maximum air temperature recorded at 2 meters above the Earth's surface during a specific period.

    **T2M_MIN**: Temperature at 2 Meters Minimum (°C) — The minimum air temperature recorded at 2 meters above the Earth's surface during a specific period.

    **RH2M**: Relative Humidity at 2 Meters (%) — The amount of moisture in the air at 2 meters above the Earth's surface, expressed as a percentage of the maximum moisture the air can hold at that temperature.

    **PRECTOTCORR**: Precipitation Corrected (mm/day) — The total precipitation measured (in millimeters per day), with adjustments or corrections made to the raw data.

    **WS2M**: Wind Speed at 2 Meters (m/s) — The average wind speed at 2 meters above the Earth's surface, measured in meters per second.

    **WS2M_MAX**: Wind Speed at 2 Meters Maximum (m/s) — The maximum wind speed measured at 2 meters above the Earth's surface during a specific period.

    **WS2M_MIN**: Wind Speed at 2 Meters Minimum (m/s) — The minimum wind speed measured at 2 meters above the Earth's surface during a specific period.

    **WD2M**: Wind Direction at 2 Meters (Degrees) — The direction from which the wind is blowing, measured at 2 meters above the Earth's surface, expressed in degrees (0° represents north, 90° represents east, etc.).

    **GWETTOP**: Surface Soil Wetness (1) — The moisture content in the top layer of soil, which can be used to assess surface soil wetness.

    **GWETROOT**: Root Zone Soil Wetness (1) — The moisture content in the soil within the root zone (typically 0-1 meter depth), which impacts plant growth.

    **GWETPROF**: Profile Soil Moisture (1) — The moisture content in the soil profile, typically measured from the surface to deeper layers (e.g., up to 3 meters or more).
    """)

    # Allow users to upload CSV files
    uploaded_file = st.file_uploader("Upload a weather dataset (CSV)", type=["csv"])

    # Use dropdown if multiple CSV files exist in a directory
    data_dir = "raw_files/"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]

    selected_file = st.selectbox("select a dataset:", csv_files) if csv_files else None

    # Load the selected or uploaded file
    if uploaded_file:
        weda = WeatherEDA(file_path=uploaded_file)
        dataset_name = uploaded_file.name
    elif selected_file:
        weda = WeatherEDA(file_path=os.path.join(data_dir, selected_file))
        #df = pd.read_csv(os.path.join(data_dir, selected_file))
        dataset_name = selected_file
    else:
        st.warning("Please upload or select a CSV file.")
        st.stop()
    
    columns = ['T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECIPITATION',
       'WS2M', 'WS2M_MAX', 'WS2M_MIN', 'WD2M', 'GWETTOP', 'GWETROOT',
       'GWETPROF',"T2M_7d_avg","T2M_30d_avg"]
    
    # Sidebar options
    st.sidebar.header("Options")
    column = st.sidebar.selectbox("Select Column For Seasonal Decomposition", columns)

    # Multi-select dropdown filter
    selected_columns = st.multiselect('Select Columns:', columns, default=['T2M','RH2M', 'PRECIPITATION'])
    
    # Moving Averages
    #df = calculate_moving_averages(df)
    weda.plot_time_series(column, "Temperature")
    
    #Seasonal Decomposition
    if st.sidebar.checkbox("Show Seasonal Decomposition"):
        weda.plot_seasonal_decomposition(column_name= column)
    
    # # Anomaly Detection (Z-Score & Isolation Forest)
    # if st.sidebar.checkbox("Show Z-Score Anomalies"):
    #     detect_anomalies_zscore(df)
    
    # if st.sidebar.checkbox("Show Isolation Forest Anomalies"):
    #     detect_anomalies_isolation_forest(df)

    # Yearly Trends Plot
    if st.sidebar.checkbox("Show Yearly Trends"):
        weda.plot_yearly_trends(columns= selected_columns)

    if st.sidebar.checkbox("Plot correlation heatmap"):
        weda.plot_correlation_heatmaps(columns= selected_columns)

    if st.sidebar.checkbox("Plot seasonal weather trends"):
        weda.plot_seasonal_trends()

    if st.sidebar.checkbox("Montly weather variable trends"):
        weda.monthly_variable_trends(columns=selected_columns)

if __name__ == "__main__":
    main()
