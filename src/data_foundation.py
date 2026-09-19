import requests
import pandas as pd

print("Downloading rainfall data...")

LAT = 28.496
LON = 77.089

url = (
    "https://api.open-meteo.com/v1/forecast"
    f"?latitude={LAT}"
    f"&longitude={LON}"
    "&hourly=precipitation"
    "&forecast_days=3"
    "&timezone=Asia%2FKolkata"
)

response = requests.get(url, timeout=30)

print("Weather API status:", response.status_code)

weather = response.json()

rainfall = pd.DataFrame({
    "time": weather["hourly"]["time"],
    "rain_mm": weather["hourly"]["precipitation"]
})

rainfall.to_csv(
    "data/raw/rainfall.csv",
    index=False
)

print("Rainfall data saved!")
print(rainfall.head())