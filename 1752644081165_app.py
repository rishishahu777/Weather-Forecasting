import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

# Ask user for CSV path
csv_path = input("Enter path to weather data CSV: ")

if not os.path.exists(csv_path):
    print("❌ File not found. Check the path.")
    exit()

# Output folder
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

try:
    # Load data
    df = pd.read_csv(csv_path)

    # Basic checks
    required_columns = {"City", "Latitude", "Longitude", "Humidity", "Pressure", "WindSpeed", "Temperature"}
    if not required_columns.issubset(df.columns):
        print("❌ CSV must include columns:", required_columns)
        exit()

    # Features and labels
    X = df[["Humidity", "Pressure", "WindSpeed"]]
    y = df["Temperature"]

    # Split and train
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=0)
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Predict and score
    predictions = model.predict(x_test)
    score = model.score(x_test, y_test)

    # Save predicted vs actual plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(y_test)), y_test.sort_values().values, label="Actual", color='blue')
    plt.plot(range(len(predictions)), sorted(predictions), label="Predicted", color='orange')
    plt.title("Actual vs Predicted Temperature")
    plt.xlabel("Sample Index")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.savefig(os.path.join(results_dir, "temperature_prediction.png"))
    plt.close()

    print(f"✅ Model R² Score: {round(score, 4)}")
    print("📈 Saved temperature prediction chart.")

    # Now generate a heatmap of India using predicted values
    # Predict temperature for all cities
    df["Predicted_Temperature"] = model.predict(df[["Humidity", "Pressure", "WindSpeed"]])

    # Plot heatmap using latitude and longitude
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(df["Longitude"], df["Latitude"], c=df["Predicted_Temperature"],
                     cmap="coolwarm", s=200, edgecolors='k')
    plt.colorbar(sc, label="Predicted Temperature (°C)")
    for i in range(len(df)):
        plt.text(df["Longitude"][i]+0.2, df["Latitude"][i], df["City"][i], fontsize=8)
    plt.title("📍 Predicted Temperature Heatmap (India)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "heatmap_india.png"))
    plt.close()

    print("🗺️ Heatmap saved as 'results/heatmap_india.png'.")

except Exception as e:
    print(f"❌ Error: {e}")
