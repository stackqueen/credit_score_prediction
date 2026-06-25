import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from datetime import datetime, timedelta
import hashlib
import pandas as pd
import numpy as np
import joblib

BASE_DIR = r"C:\Users\ASUS\Desktop\credit_behaviour_score"
app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app", "static")), name="static")

SECRET_KEY = "creditrisk2024secretkey"
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()
USERS = {"admin": hash_pw("admin123")}

model = joblib.load(os.path.join(BASE_DIR, "models", "xgb_model.pkl"))
submission = pd.read_csv(os.path.join(BASE_DIR, "reports", "submission.csv"))

def create_token(username):
    return jwt.encode({"sub": username, "exp": datetime.utcnow() + timedelta(hours=1)}, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username not in USERS: raise HTTPException(status_code=401)
        return username
    except JWTError:
        raise HTTPException(status_code=401)

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends()):
    if form.username not in USERS or USERS[form.username] != hash_pw(form.password):
        raise HTTPException(status_code=401, detail="Wrong credentials")
    return {"access_token": create_token(form.username), "token_type": "bearer"}

@app.get("/")
def login_page(): return FileResponse(os.path.join(BASE_DIR, "app", "templates", "login.html"))

@app.get("/dashboard")
def dashboard(): return FileResponse(os.path.join(BASE_DIR, "app", "templates", "dashboard.html"))

@app.get("/lookup")
def lookup_page(): return FileResponse(os.path.join(BASE_DIR, "app", "templates", "lookup.html"))

@app.get("/api/stats")
def get_stats(user: str = Depends(get_current_user)):
    total = len(submission)
    high_risk = int((submission['predicted_probability'] > 0.5).sum())
    med_risk = int(((submission['predicted_probability'] > 0.2) & (submission['predicted_probability'] <= 0.5)).sum())
    low_risk = int((submission['predicted_probability'] <= 0.2).sum())
    hist, bins = np.histogram(submission['predicted_probability'], bins=10)
    return {"total": total, "high_risk": high_risk, "med_risk": med_risk, "low_risk": low_risk,
            "avg_score": round(float(submission['predicted_probability'].mean()), 4),
            "histogram": {"counts": hist.tolist(), "bins": [round(b,2) for b in bins.tolist()]}}

@app.get("/api/lookup/{account_id}")
def lookup_account(account_id: int, user: str = Depends(get_current_user)):
    row = submission[submission['account_number'] == account_id]
    if row.empty: raise HTTPException(status_code=404, detail="Account not found")
    prob = float(row['predicted_probability'].values[0])
    risk = "High" if prob > 0.5 else "Medium" if prob > 0.2 else "Low"
    return {"account_number": account_id, "probability": round(prob, 4), "risk_level": risk}
