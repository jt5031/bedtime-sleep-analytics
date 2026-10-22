# 💤 Bedtime Screentime & Sleep Debt Analytics

An end-to-end data science project analyzing how bedtime phone habits, screen brightness, caffeine intake, and physical activity correlate with user sleep debt and next-day fatigue.

## 🚀 Live App
Check out the interactive Streamlit web app here: [Insert Streamlit Public Cloud URL]

## 📊 Key Insights & Findings
- **Screentime Impact:** Users spending over 90 minutes on phones before bed exhibit significantly higher sleep latency and lower REM sleep percentages.
- **Caffeine & Blue Light:** Post-5 PM caffeine consumption compounds sleep debt, heavily spiking morning alarm snooze counts.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Streamlit, Joblib
- **App Deployment:** Streamlit Community Cloud

## 📂 Project Structure
```text
bedtime-sleep-analytics/
│
├── data/
│   └── bedtime_screentime_sleep_debt.csv   # Dataset
│
├── src/
│   └── train_model.py                      # Model training & serialization script
│
├── app.py                                  # Streamlit interactive web application
├── requirements.txt                        # Project dependencies
├── model.pkl                               # Trained model artifacts
├── label_encoders.pkl                      # Categorical feature encoders
└── target_encoder.pkl                      # Target class encoder