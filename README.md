# 🏠 Household Energy Consumption Prediction

This project builds a machine learning model to predict **household energy consumption (kWh)** using real-world data.  
The focus is on **understanding ML fundamentals**, including feature engineering, normalization, polynomial regression, and **manual gradient descent**, rather than relying only on black-box libraries.

---

## 📊 Dataset

- **Source:** Kaggle  
- **Dataset:** `samxsam/household-energy-consumption`  
- **File Used:** `household_energy_consumption.csv`

### Columns Used
| Column | Description |
|------|------------|
| `Date` | Date of observation |
| `Household_Size` | Number of residents |
| `Avg_Temperature_C` | Average temperature (°C) |
| `Has_AC` | Whether the household has AC (Yes/No) |
| `Energy_Consumption_kWh` | Target variable |

---

## 🧠 Feature Engineering

### 1. Season Encoding
The `Date` column is converted to a season:
- **Winter (0):** September – February  
- **Summer (1):** March – August  

```python
df['Season'] = df['Date'].dt.month.apply(get_season)
