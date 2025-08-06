# XGBoost-PUN-prediction

# ⚡ PUN Forecasting with XGBoost

**Goal** Forecasting of "Prezzo Unico Nazionale" (PUN), the hourly electricity price in Italy using XGBoost.

## 📁 Structure

XGBoost-PUN-prediction/
│
├── data/ # Raw Data
├── notebooks/ # Data manipulation and models
├── models/ # Saved models
├── reports/ # Results Visualization
├── src/ # Useful functions
├── requirements.txt
├── README.md
├── .gitignore

## 📊 Data

- **Source**: Dataset Kaggle ([link](https://www.kaggle.com/datasets/gianpieroandrenacci/energy-pun-main-zones/data?select=energy_pun_main_zones.csv))
- **Covered Period**: April 2004 - March 2023
- **Frequency**: hourly
- **Variable**:PUN price, timestamp

## 🚧 Project status
✅ Data collection
🔄 on going: engineering and model setup
⬜ Backtesting 

## 📦 Requirements
```bash
pip install -r requirements.txt