import os
import json
import logging
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv

##load_dotenv()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logging.basicConfig(level=logging.INFO)
    logging.warning(
        "GOOGLE_API_KEY bulunamadı. Gemini entegrasyonu devre dışı bırakıldı. .env dosyanıza `GOOGLE_API_KEY` ekleyin."
    )

class KeystrokeData(BaseModel):
    key: str
    code: str
    event_type: str 
    timestamp: float

class RegisterRequest(BaseModel):
    username: str
    email: str
    keystrokes: List[KeystrokeData]

@app.get("/", response_class=HTMLResponse)
async def read_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

@app.get("/api/get-text")
async def get_typing_text():
    if api_key:
        try:
            prompt = "Bana klavye dinamiklerini ölçmek için harf çeşitliliği yüksek, 50-60 kelimelik, Türkçe, resmi ve akıcı bir paragraf yaz. Sadece metni döndür."
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return {"text": response.text}
        except Exception as e:
            logging.warning(f"Gemini API hatası: {str(e)} - Mock data kullanılıyor")

    mock_text = "inanc inanc inanc inanc"
    print("Mock data döndürülüyor")
    return {"text": mock_text}

@app.post("/api/register")
async def register_user(data: RegisterRequest):
    try:
        data_dir = Path("keystroke_data")
        data_dir.mkdir(exist_ok=True)
        
        user_dir = data_dir / data.username
        user_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        raw_data_file = user_dir / f"raw_data_{timestamp}.json"
        raw_data = {
            "username": data.username,
            "email": data.email,
            "timestamp": timestamp,
            "total_keystrokes": len(data.keystrokes),
            "keystrokes": [k.dict() for k in data.keystrokes]
        }
        
        with open(raw_data_file, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        features = extract_keystroke_features(data.keystrokes)
        features_file = user_dir / f"features_{timestamp}.csv"
        save_features_to_csv(features, features_file, data.username)
        
        logging.info(f"{data.username} için veri kaydedildi: {len(data.keystrokes)} tuş vuruşu")
        
        return {
            "status": "success",
            "message": "Kayıt ve Biyometrik Veri Alındı",
            "username": data.username,
            "keystroke_count": len(data.keystrokes),
            "files_saved": [str(raw_data_file), str(features_file)]
        }
        
    except Exception as e:
        logging.error(f"Veri kaydetme hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Veri kaydedilemedi: {str(e)}")


def extract_keystroke_features(keystrokes: List[KeystrokeData]) -> dict:
    features = {
        "dwell_times": [],      
        "flight_times": [],     
        "digraph_times": [],    
        "keys_pressed": [],     
        "hold_times": []        
    }
    
    keydown_events = {}
    previous_keyup_time = None
    previous_key = None
    
    for event in keystrokes:
        key = event.key
        event_type = event.event_type
        timestamp = event.timestamp
        
        if event_type == "keydown":
            keydown_events[key] = timestamp
            
            if previous_keyup_time is not None:
                flight_time = timestamp - previous_keyup_time
                features["flight_times"].append(flight_time)
            
            if previous_key is not None:
                features["digraph_times"].append({
                    "from": previous_key,
                    "to": key,
                    "time": timestamp - (previous_keyup_time or timestamp)
                })
        
        elif event_type == "keyup":
            if key in keydown_events:
                dwell_time = timestamp - keydown_events[key]
                features["dwell_times"].append(dwell_time)
                features["hold_times"].append({
                    "key": key,
                    "dwell_time": dwell_time
                })
                del keydown_events[key]
            
            previous_keyup_time = timestamp
            previous_key = key
            features["keys_pressed"].append(key)
    
    return features


def save_features_to_csv(features: dict, filepath: Path, username: str):
    import csv
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "username",
            "feature_type",
            "key_from",
            "key_to", 
            "time_ms",
            "sequence_index"
        ])
        
        for idx, dwell in enumerate(features["dwell_times"]):
            writer.writerow([username, "dwell_time", "", "", dwell, idx])
        
        for idx, flight in enumerate(features["flight_times"]):
            writer.writerow([username, "flight_time", "", "", flight, idx])
        
        for idx, digraph in enumerate(features["digraph_times"]):
            writer.writerow([
                username,
                "digraph_time",
                digraph["from"],
                digraph["to"],
                digraph["time"],
                idx
            ])
        
        for idx, hold in enumerate(features["hold_times"]):
            writer.writerow([
                username,
                "hold_time",
                hold["key"],
                "",
                hold["dwell_time"],
                idx
            ])
    
    logging.info(f"Özellikler CSV'ye kaydedildi: {filepath}")