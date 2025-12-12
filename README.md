# 🚗 Car Price & Engine Health Studio  
AI-powered prediction of car resale value & engine condition — all in your browser.

Live App → **https://carprice-enginehealth.streamlit.app/**  
GitHub Repo → **https://github.com/vedkharat/CarPrice-EngineHealth**

---

## ⭐ Features  
### 🔹 Car Price Estimator  
- Predicts resale price using:
  - Manufacturer & model  
  - Year of manufacture  
  - Mileage  
  - Fuel type  
  - Condition  
  - State of registration  
  - Transmission type  

### 🔹 Engine Health Analyzer  
- Upload an engine sound clip (MP3/WAV/OGG)  
- ML model classifies sound as **Healthy** or **Faulty**

---

## 📁 Project Structure  
```
CarPrice-EngineHealth/
│
├── main.py                     
├── config.toml                
├── requirements.txt           
├── engine_health_catboost.pkl 
├── front_lights.mp4           
├── rear_lights.mp4            
├── README.md                  
└── .gitignore
```

---

## 🔧 Installation (Run Locally)

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/vedkharat/CarPrice-EngineHealth.git
cd CarPrice-EngineHealth
```

### 2️⃣ Create and activate a virtual environment  
```bash
python -m venv venv
source venv/bin/activate  
# Windows:
venv\Scripts\activate
```

### 3️⃣ Install dependencies  
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit app  
```bash
streamlit run main.py
```

---

## 🚀 Deployment  
Deployed on **Streamlit Cloud**.  
Large model files hosted on Google Drive via direct-download links.

---

## 📊 Models Used  
### 🔹 Car Price Model  
- CatBoost Regressor  
- Stored as: `car_price_catboost.pkl`

### 🔹 Engine Health Model  
- Binary classifier: “Healthy” / “Faulty”  
- Stored as: `engine_health_catboost.pkl`

---

## 🖥️ UI & Experience  
- Dark theme  
- Background animation videos  
- Two-tab layout  

---

## 🙌 Acknowledgments  
- CatBoost  
- Streamlit  
- Public datasets  
- Google Drive hosting  

---

## 📜 License  
Educational & portfolio use only.  
