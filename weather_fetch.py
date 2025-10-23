import requests
import pandas as pd
from tqdm import tqdm

# NUTS regions with representative coordinates
regions = {
    "Norte": (41.2, -8.4),
    "Centro": (40.3, -8.2),
    "Área Metropolitana de Lisboa": (38.7, -9.1),
    "Alentejo": (38.0, -7.9),
    "Algarve": (37.0, -8.0),
    "Região Autónoma dos Açores": (37.8, -25.5),
    "Região Autónoma da Madeira": (32.7, -16.9)
}


def fetch_weather(lat, lon, start_date, end_date):
    """Fetch daily historical weather data from Open-Meteo Archive API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&daily=temperature_2m_mean,precipitation_sum,sunshine_duration,windspeed_10m_mean"
        "&timezone=Europe%2FLisbon"
    )
    response = requests.get(url)
    data = response.json()
    return pd.DataFrame(data["daily"])

# fetch weather data for each regionall_regions = []
all_regions = []

for region, (lat, lon) in tqdm(regions.items()):
    df = fetch_weather(lat, lon, "2019-01-01", "2025-03-31")
    df["date"] = pd.to_datetime(df["time"])
    df["region"] = region
    df["quarter"] = df["date"].dt.to_period("Q")

    # Aggregate to quarterly averages/sums
    agg = (
        df.groupby(["region", "quarter"])
          .agg({
              "temperature_2m_mean": "mean",
              "precipitation_sum": "sum",
              "sunshine_duration": "sum",
              "windspeed_10m_mean": "mean"
          })
          .reset_index()
    )
    all_regions.append(agg)

# Combine all regions
weather_quarterly = pd.concat(all_regions, ignore_index=True)

# Convert quarter to string for easy merging later
weather_quarterly["quarter"] = weather_quarterly["quarter"].astype(str)

# save to the csv file

output_file = "portugal_weather_quarterly_2019_2025.csv"
weather_quarterly.to_csv(output_file, index=False, encoding="utf-8")
print(f"Saved: {output_file}")
weather_quarterly.head()
