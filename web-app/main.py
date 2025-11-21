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

class SessionMetadata(BaseModel):
    correct_keys: int = 0
    incorrect_keys: int = 0
    accuracy_rate: float = 0.0
    target_text: str = ""
    total_keystrokes: int = 0

class RegisterRequest(BaseModel):
    username: str
    email: str
    keystrokes: List[KeystrokeData]
    metadata: SessionMetadata = None

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
            "keystrokes": [k.dict() for k in data.keystrokes],
            "session_metadata": data.metadata.dict() if data.metadata else {}
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
    import numpy as np
    
    features = {
        "dwell_times": [],
        "flight_times": [],
        "digraph_times": [],
        "trigram_times": [],
        "four_gram_times": [],
        "keys_pressed": [],
        "hold_times": [],
        "backspace_count": 0,
        "backspace_latencies": [],
        "error_corrections": [],
        "incorrect_keys": [],
        "shift_usage": [],
        "special_key_combos": [],
        "key_down_times": [],
        "key_up_times": [],
        "inter_key_delays": [],
        "typing_bursts": [],
        "pause_events": [],
        "speed_changes": [],
        "key_pressure_estimate": [],
        "rhythm_consistency": []
    }
    
    keydown_events = {}
    keyup_events = {}
    
    previous_keyup_time = None
    previous_key = None
    previous_keys_history = []
    
    typed_sequence = []
    expected_sequence = []
    
    last_event_time = None
    time_since_last = []
    
    for idx, event in enumerate(keystrokes):
        key = event.key
        event_type = event.event_type
        timestamp = event.timestamp
        
        if last_event_time is not None:
            inter_key_delay = timestamp - last_event_time
            features["inter_key_delays"].append(inter_key_delay)
            time_since_last.append(inter_key_delay)
        
        last_event_time = timestamp
        
        if event_type == "keydown":
            keydown_events[key] = timestamp
            features["key_down_times"].append(timestamp)
            
            if key == "Backspace":
                features["backspace_count"] += 1
                if previous_keyup_time is not None:
                    backspace_latency = timestamp - previous_keyup_time
                    features["backspace_latencies"].append(backspace_latency)
                
                if len(typed_sequence) > 0:
                    corrected_char = typed_sequence.pop()
                    features["error_corrections"].append({
                        "char": corrected_char,
                        "latency": backspace_latency if previous_keyup_time else 0,
                        "position": len(typed_sequence)
                    })
                continue
            
            if key == "Shift":
                features["shift_usage"].append({
                    "timestamp": timestamp,
                    "next_key": None
                })
                continue
            
            if previous_keyup_time is not None:
                flight_time = timestamp - previous_keyup_time
                features["flight_times"].append(flight_time)
                
                if len(features["flight_times"]) > 1:
                    speed_change = flight_time - features["flight_times"][-2]
                    features["speed_changes"].append(speed_change)
                
                if flight_time > 300:
                    features["pause_events"].append({
                        "duration": flight_time,
                        "after_key": previous_key,
                        "before_key": key,
                        "position": len(typed_sequence)
                    })
            
            if previous_key is not None:
                digraph_time = timestamp - (previous_keyup_time or timestamp)
                features["digraph_times"].append({
                    "from": previous_key,
                    "to": key,
                    "time": digraph_time
                })
            
            if len(previous_keys_history) >= 2:
                trigram_time = timestamp - (keyup_events.get(previous_keys_history[-2], timestamp))
                features["trigram_times"].append({
                    "keys": previous_keys_history[-2:] + [key],
                    "time": trigram_time
                })
            
            if len(previous_keys_history) >= 3:
                four_gram_time = timestamp - (keyup_events.get(previous_keys_history[-3], timestamp))
                features["four_gram_times"].append({
                    "keys": previous_keys_history[-3:] + [key],
                    "time": four_gram_time
                })
            
            if len(features["shift_usage"]) > 0 and features["shift_usage"][-1]["next_key"] is None:
                features["shift_usage"][-1]["next_key"] = key
        
        elif event_type == "keyup":
            keyup_events[key] = timestamp
            features["key_up_times"].append(timestamp)
            
            if key == "Shift" or key == "Backspace":
                continue
            
            if key in keydown_events:
                dwell_time = timestamp - keydown_events[key]
                features["dwell_times"].append(dwell_time)
                
                pressure_estimate = min(dwell_time / 200.0, 1.0)
                features["key_pressure_estimate"].append({
                    "key": key,
                    "pressure": pressure_estimate,
                    "dwell": dwell_time
                })
                
                features["hold_times"].append({
                    "key": key,
                    "dwell_time": dwell_time
                })
                
                del keydown_events[key]
            
            previous_keyup_time = timestamp
            previous_key = key
            features["keys_pressed"].append(key)
            typed_sequence.append(key)
            
            previous_keys_history.append(key)
            if len(previous_keys_history) > 5:
                previous_keys_history.pop(0)
    
    features["statistics"] = calculate_statistical_features(features)
    features["rhythm_metrics"] = calculate_rhythm_metrics(features)
    features["typing_speed"] = calculate_typing_speed(keystrokes, typed_sequence)
    
    return features


def calculate_statistical_features(features: dict) -> dict:
    import numpy as np
    
    stats = {}
    
    metrics = ["dwell_times", "flight_times", "inter_key_delays"]
    
    for metric in metrics:
        if metric in features and len(features[metric]) > 0:
            data = np.array(features[metric])
            
            stats[f"{metric}_mean"] = float(np.mean(data))
            stats[f"{metric}_std"] = float(np.std(data))
            stats[f"{metric}_median"] = float(np.median(data))
            stats[f"{metric}_min"] = float(np.min(data))
            stats[f"{metric}_max"] = float(np.max(data))
            stats[f"{metric}_q1"] = float(np.percentile(data, 25))
            stats[f"{metric}_q3"] = float(np.percentile(data, 75))
            stats[f"{metric}_iqr"] = stats[f"{metric}_q3"] - stats[f"{metric}_q1"]
            
            if stats[f"{metric}_mean"] > 0:
                stats[f"{metric}_cv"] = stats[f"{metric}_std"] / stats[f"{metric}_mean"]
            else:
                stats[f"{metric}_cv"] = 0
            
            if len(data) > 2:
                from scipy import stats as scipy_stats
                stats[f"{metric}_skewness"] = float(scipy_stats.skew(data))
                stats[f"{metric}_kurtosis"] = float(scipy_stats.kurtosis(data))
    
    return stats


def calculate_rhythm_metrics(features: dict) -> dict:
    import numpy as np
    
    rhythm = {}
    
    if len(features["flight_times"]) > 3:
        flight_array = np.array(features["flight_times"])
        
        rhythm["consistency_score"] = 1.0 / (1.0 + np.std(flight_array))
        
        consecutive_diffs = np.abs(np.diff(flight_array))
        rhythm["rhythm_stability"] = float(np.mean(consecutive_diffs))
        
        fast_threshold = 150
        rhythm["burst_ratio"] = float(np.sum(flight_array < fast_threshold) / len(flight_array))
        
        rhythm["continuity"] = 1.0 - (len(features["pause_events"]) / max(len(features["keys_pressed"]), 1))
    
    if len(features["key_pressure_estimate"]) > 0:
        pressures = [p["pressure"] for p in features["key_pressure_estimate"]]
        rhythm["pressure_consistency"] = 1.0 / (1.0 + np.std(pressures))
        rhythm["avg_pressure"] = float(np.mean(pressures))
    
    return rhythm


def calculate_typing_speed(keystrokes: List[KeystrokeData], typed_sequence: list) -> dict:
    speed = {}
    
    if len(keystrokes) < 2:
        return speed
    
    start_time = keystrokes[0].timestamp
    end_time = keystrokes[-1].timestamp
    duration_seconds = (end_time - start_time) / 1000.0
    
    if duration_seconds > 0:
        char_count = len([k for k in keystrokes if k.event_type == "keydown" and k.key != "Backspace"])
        speed["characters_per_second"] = char_count / duration_seconds
        
        speed["words_per_minute"] = (char_count / 5.0) * (60.0 / duration_seconds)
        
        speed["characters_per_minute"] = char_count * (60.0 / duration_seconds)
        
        actual_chars = len(typed_sequence)
        speed["net_wpm"] = (actual_chars / 5.0) * (60.0 / duration_seconds)
        
        if char_count > 0:
            speed["accuracy_rate"] = actual_chars / char_count
    
    speed["total_duration_seconds"] = duration_seconds
    speed["total_characters"] = len(typed_sequence)
    
    return speed


def save_features_to_csv(features: dict, filepath: Path, username: str):
    import csv
    import json
    
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        writer.writerow([
            "username",
            "feature_category",
            "feature_type",
            "key_from",
            "key_to",
            "value",
            "additional_data",
            "sequence_index"
        ])
        
        for idx, dwell in enumerate(features["dwell_times"]):
            writer.writerow([username, "timing", "dwell_time", "", "", dwell, "", idx])
        
        for idx, flight in enumerate(features["flight_times"]):
            writer.writerow([username, "timing", "flight_time", "", "", flight, "", idx])
        
        for idx, delay in enumerate(features["inter_key_delays"]):
            writer.writerow([username, "timing", "inter_key_delay", "", "", delay, "", idx])
        
        for idx, digraph in enumerate(features["digraph_times"]):
            writer.writerow([
                username, "n-gram", "digraph",
                digraph["from"], digraph["to"],
                digraph["time"], "", idx
            ])
        
        for idx, trigram in enumerate(features["trigram_times"]):
            keys_str = "->".join(trigram["keys"])
            writer.writerow([
                username, "n-gram", "trigram",
                keys_str, "",
                trigram["time"], "", idx
            ])
        
        for idx, four_gram in enumerate(features["four_gram_times"]):
            keys_str = "->".join(four_gram["keys"])
            writer.writerow([
                username, "n-gram", "four_gram",
                keys_str, "",
                four_gram["time"], "", idx
            ])
        
        writer.writerow([
            username, "behavior", "backspace_count",
            "", "", features["backspace_count"], "", 0
        ])
        
        for idx, latency in enumerate(features["backspace_latencies"]):
            writer.writerow([
                username, "behavior", "backspace_latency",
                "", "", latency, "", idx
            ])
        
        for idx, correction in enumerate(features["error_corrections"]):
            writer.writerow([
                username, "behavior", "error_correction",
                correction["char"], "",
                correction["latency"],
                json.dumps({"position": correction["position"]}),
                idx
            ])
        
        for idx, pause in enumerate(features["pause_events"]):
            writer.writerow([
                username, "rhythm", "pause",
                pause["after_key"], pause["before_key"],
                pause["duration"],
                json.dumps({"position": pause["position"]}),
                idx
            ])
        
        for idx, speed_change in enumerate(features["speed_changes"]):
            writer.writerow([
                username, "rhythm", "speed_change",
                "", "", speed_change, "", idx
            ])
        
        for idx, pressure in enumerate(features["key_pressure_estimate"]):
            writer.writerow([
                username, "pressure", "key_pressure",
                pressure["key"], "",
                pressure["pressure"],
                json.dumps({"dwell": pressure["dwell"]}),
                idx
            ])
        
        for idx, shift_usage in enumerate(features["shift_usage"]):
            writer.writerow([
                username, "behavior", "shift_combo",
                "", shift_usage.get("next_key", ""),
                shift_usage["timestamp"], "", idx
            ])
        
        if "statistics" in features:
            for stat_name, stat_value in features["statistics"].items():
                writer.writerow([
                    username, "statistics", stat_name,
                    "", "", stat_value, "", 0
                ])
        
        if "rhythm_metrics" in features:
            for metric_name, metric_value in features["rhythm_metrics"].items():
                writer.writerow([
                    username, "rhythm_metrics", metric_name,
                    "", "", metric_value, "", 0
                ])
        
        if "typing_speed" in features:
            for speed_name, speed_value in features["typing_speed"].items():
                writer.writerow([
                    username, "typing_speed", speed_name,
                    "", "", speed_value, "", 0
                ])
    
    logging.info(f"Gelişmiş biometrik özellikler CSV'ye kaydedildi: {filepath}")