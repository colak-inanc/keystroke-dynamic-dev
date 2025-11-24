import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf

from app.models import KeystrokeData, LoginRequest


APP_DIR = Path(__file__).resolve().parent
WEB_APP_DIR = APP_DIR.parent
PROJECT_ROOT = WEB_APP_DIR.parent
MODEL_PATH = PROJECT_ROOT / "model" / "keystroke_lstm_model.h5"
DATA_DIR = WEB_APP_DIR / "keystroke_data"

_model = None


def load_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        try:
            _model = tf.keras.models.load_model(str(MODEL_PATH))
            logging.info("Model yüklendi: %s", MODEL_PATH)
            try:
                _model.summary(print_fn=lambda x: logging.info("Model: %s", x))
                logging.info("Model input shape: %s", _model.input_shape)
                logging.info("Model output shape: %s", _model.output_shape)
            except Exception: 
                pass
        except Exception as exc:  
            logging.error("Model yükleme hatası: %s", exc)
    return _model


def prepare_sequence_for_model(keystrokes: List[KeystrokeData], features: dict) -> np.ndarray:
    loaded_model = load_model()
    if loaded_model is None:
        num_features = 3
    else:
        try:
            input_shape = loaded_model.input_shape
            if isinstance(input_shape, (list, tuple)) and len(input_shape) > 0:
                if isinstance(input_shape[0], (list, tuple)) and len(input_shape[0]) > 0:
                    num_features = input_shape[0][-1]
                else:
                    num_features = input_shape[-1] if len(input_shape) > 1 else 3
            else:
                num_features = 3
        except Exception:
            num_features = 3

    sequence = []
    keydown_events = {}
    keyup_events = {}
    previous_keyup_time = None
    previous_keydown_time = None
    previous_key = None

    for idx, event in enumerate(keystrokes):
        key = event.key
        event_type = event.event_type
        timestamp = event.timestamp

        if event_type == "keydown":
            keydown_events[key] = timestamp
            previous_keydown_time = timestamp

            if key in {"Backspace", "Shift"}:
                continue

            if previous_keyup_time is not None:
                flight_time = timestamp - previous_keyup_time
            else:
                flight_time = 0.0

            inter_key_delay = timestamp - keystrokes[idx - 1].timestamp if idx > 0 else 0.0

        elif event_type == "keyup":
            keyup_events[key] = timestamp

            if key in {"Shift", "Backspace"}:
                if key in keydown_events:
                    del keydown_events[key]
                continue

            dwell_time = 0.0
            keydown_time = keydown_events.get(key)
            if keydown_time is not None:
                dwell_time = timestamp - keydown_time
                del keydown_events[key]

            if previous_keyup_time is not None and keydown_time is not None:
                flight_time = keydown_time - previous_keyup_time
            else:
                flight_time = 0.0

            inter_key_delay = timestamp - keystrokes[idx - 1].timestamp if idx > 0 else 0.0

            normalized_dwell = min(dwell_time / 500.0, 2.0)
            normalized_flight = min(flight_time / 500.0, 2.0) if flight_time > 0 else 0.0
            normalized_delay = min(inter_key_delay / 1000.0, 2.0)

            feature_vector = [
                float(normalized_dwell),
                float(normalized_flight),
                float(normalized_delay)
            ]

            if num_features >= 4:
                key_pressure = min(dwell_time / 200.0, 1.0)
                feature_vector.append(float(key_pressure))

            if num_features >= 5:
                if previous_keydown_time is not None and keydown_time is not None:
                    down_down_latency = keydown_time - previous_keydown_time
                    normalized_down_down = min(down_down_latency / 500.0, 2.0)
                else:
                    normalized_down_down = 0.0
                feature_vector.append(float(normalized_down_down))

            while len(feature_vector) < num_features:
                feature_vector.append(0.0)

            sequence.append(feature_vector[:num_features])
            previous_keyup_time = timestamp
            previous_key = key

    target_length = 10

    if len(sequence) == 0:
        sequence = [[0.0] * num_features] * target_length
    elif len(sequence) < target_length:
        last_value = sequence[-1] if sequence else [0.0] * num_features
        while len(sequence) < target_length:
            sequence.append(last_value.copy())
    else:
        sequence = sequence[:target_length]

    for i in range(len(sequence)):
        for j in range(num_features):
            if np.isnan(sequence[i][j]) or np.isinf(sequence[i][j]):
                sequence[i][j] = 0.0

    return np.array([sequence], dtype=np.float32)


def calculate_similarity(user_features, stored_data):
    if len(user_features) < 5:
        return 0.0

    similarities = []

    for df in stored_data:
        stats = df[df["feature_category"] == "statistics"]
        rhythm = df[df["feature_category"] == "rhythm_metrics"]
        typing = df[df["feature_category"] == "typing_speed"]

        stored_features = []
        for _, row in stats.iterrows():
            stored_features.append(row["value"])
        for _, row in rhythm.iterrows():
            stored_features.append(row["value"])
        for _, row in typing.iterrows():
            stored_features.append(row["value"])

        timing_df = df[df["feature_category"] == "timing"]
        dwell_times = timing_df[timing_df["feature_type"] == "dwell_time"]["value"].values
        if len(dwell_times) > 0:
            stored_features.append(np.mean(dwell_times))
            stored_features.append(np.std(dwell_times))

        flight_times = timing_df[timing_df["feature_type"] == "flight_time"]["value"].values
        if len(flight_times) > 0:
            stored_features.append(np.mean(flight_times))
            stored_features.append(np.std(flight_times))

        backspace = df[df["feature_type"] == "backspace_count"]
        if len(backspace) > 0:
            stored_features.append(backspace["value"].iloc[0])

        keys_count = len(timing_df)
        stored_features.append(keys_count)

        min_len = min(len(user_features), len(stored_features))
        if min_len < 5:
            continue

        user_arr = np.array(user_features[:min_len])
        stored_arr = np.array(stored_features[:min_len])

        user_arr = np.nan_to_num(user_arr, 0)
        stored_arr = np.nan_to_num(stored_arr, 0)

        if np.std(user_arr) > 0 and np.std(stored_arr) > 0:
            user_norm = (user_arr - np.mean(user_arr)) / np.std(user_arr)
            stored_norm = (stored_arr - np.mean(stored_arr)) / np.std(stored_arr)

            correlation = np.corrcoef(user_norm, stored_norm)[0, 1]
            if not np.isnan(correlation):
                similarities.append((correlation + 1) / 2)

        distance = np.linalg.norm(user_arr - stored_arr)
        max_distance = np.linalg.norm(stored_arr) * 2
        if max_distance > 0:
            distance_similarity = 1 - min(distance / max_distance, 1)
            similarities.append(distance_similarity)

    if len(similarities) == 0:
        return 0.0

    return float(np.mean(similarities))


def save_login_session(username: str, keystrokes: List[KeystrokeData], features: dict, confidence: float, success: bool):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        user_dir = DATA_DIR / username
        user_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_data_file = user_dir / f"login_{timestamp}.json"
        letter_errors = summarize_letter_errors(features)

        raw_data = {
            "username": username,
            "timestamp": timestamp,
            "session_type": "login",
            "success": success,
            "confidence": confidence,
            "total_keystrokes": len(keystrokes),
            "keystrokes": [k.dict() for k in keystrokes],
            "session_metadata": {
                "accuracy_rate": features.get("typing_speed", {}).get("accuracy_rate", 0.0) * 100,
                "wpm": features.get("typing_speed", {}).get("words_per_minute", 0.0),
                "correct_keys": len([k for k in keystrokes if k.event_type == "keydown" and k.key != "Backspace"]),
                "total_keys": len(keystrokes)
            },
            "letter_error_summary": letter_errors
        }

        with open(raw_data_file, "w", encoding="utf-8") as file:
            json.dump(raw_data, file, indent=2, ensure_ascii=False)

        logging.info("Giriş oturumu kaydedildi: %s - %s", username, "Başarılı" if success else "Başarısız")
    except Exception as exc:  
        logging.error("Giriş oturumu kaydetme hatası: %s", exc)


async def login_user_legacy(data: LoginRequest):
    import pandas as pd  

    user_dir = DATA_DIR / data.username

    csv_files = list(user_dir.glob("features_*.csv"))
    if len(csv_files) < 3:
        return {
            "status": "error",
            "message": f"Yeterli eğitim verisi yok. En az 3 oturum gerekli (mevcut: {len(csv_files)})"
        }

    features = extract_keystroke_features(data.keystrokes)
    letter_summary = summarize_letter_errors(features)

    user_features = []
    for stat_value in features.get("statistics", {}).values():
        user_features.append(stat_value)

    for metric_value in features.get("rhythm_metrics", {}).values():
        user_features.append(metric_value)

    for speed_value in features.get("typing_speed", {}).values():
        user_features.append(speed_value)

    if len(features["dwell_times"]) > 0:
        user_features.append(np.mean(features["dwell_times"]))
        user_features.append(np.std(features["dwell_times"]))

    if len(features["flight_times"]) > 0:
        user_features.append(np.mean(features["flight_times"]))
        user_features.append(np.std(features["flight_times"]))

    user_features.append(features["backspace_count"])
    user_features.append(len(features["keys_pressed"]))

    stored_data = []
    for csv_file in csv_files[:5]:
        df = pd.read_csv(csv_file)
        stats_df = df[df["feature_category"] == "statistics"]
        if len(stats_df) > 0:
            stored_data.append(df)

    if len(stored_data) == 0:
        return {
            "status": "error",
            "message": "Kayıtlı veri formatı uyumsuz"
        }

    confidence = calculate_similarity(user_features, stored_data)
    THRESHOLD = 0.7

    if confidence >= THRESHOLD:
        logging.info("Başarılı giriş (Legacy): %s (güven: %.2f)", data.username, confidence)

        try:
            save_login_session(data.username, data.keystrokes, features, float(confidence), True)
        except Exception as exc:  
            logging.warning("Giriş oturumu kaydedilemedi: %s", exc)

        return {
            "status": "success",
            "username": data.username,
            "confidence": float(confidence),
            "message": "Kimlik doğrulandı",
            "letter_error_summary": letter_summary
        }

    logging.warning("Başarısız giriş denemesi (Legacy): %s (güven: %.2f)", data.username, confidence)

    try:
        save_login_session(data.username, data.keystrokes, features, float(confidence), False)
    except Exception as exc:  
        logging.warning("Giriş oturumu kaydedilemedi: %s", exc)

        return {
        "status": "error",
            "message": f"Klavye dinamikleri eşleşmiyor (güven: {confidence*100:.1f}%). Bu hesap size ait olmayabilir.",
            "letter_error_summary": letter_summary
    }


def get_user_stats_data(username: str):
    import pandas as pd  

    user_dir = DATA_DIR / username

    if not user_dir.exists():
        return {
            "total_sessions": 0,
            "avg_accuracy": 0,
            "avg_wpm": 0,
            "sessions": [],
            "accuracy_trend": []
        }

    sessions = []
    accuracy_values = []
    wpm_values = []

    json_files = list(user_dir.glob("*.json"))
    json_files.sort(key=lambda path: path.stat().st_mtime, reverse=True)

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            timestamp_str = data.get("timestamp", "")
            if len(timestamp_str) == 15:
                date_str = f"{timestamp_str[0:4]}-{timestamp_str[4:6]}-{timestamp_str[6:8]} {timestamp_str[9:11]}:{timestamp_str[11:13]}:{timestamp_str[13:15]}"
            else:
                date_str = json_file.stem.replace("raw_data_", "").replace("login_", "")

            session_type = "Kayıt" if "login" not in json_file.name else "Giriş"

            metadata = data.get("session_metadata", {})
            if isinstance(metadata, dict):
                accuracy = metadata.get("accuracy_rate", 0.0)
                if accuracy < 1.0:
                    accuracy *= 100
                wpm = metadata.get("wpm", 0.0) or metadata.get("words_per_minute", 0.0)
            else:
                accuracy = 0.0
                wpm = 0.0
                features_file = user_dir / json_file.name.replace("raw_data_", "features_").replace("login_", "features_").replace(".json", ".csv")
                if features_file.exists():
                    try:
                        df = pd.read_csv(features_file)
                        speed_df = df[df["feature_category"] == "typing_speed"]
                        if len(speed_df) > 0:
                            wpm_row = speed_df[speed_df["feature_type"] == "words_per_minute"]
                            if len(wpm_row) > 0:
                                wpm = float(wpm_row.iloc[0]["value"])
                            accuracy_row = speed_df[speed_df["feature_type"] == "accuracy_rate"]
                            if len(accuracy_row) > 0:
                                accuracy = float(accuracy_row.iloc[0]["value"]) * 100
                    except Exception:  
                        pass

            confidence = data.get("confidence", 1.0)
            success = data.get("success", True)

            sessions.append({
                "date": date_str,
                "type": session_type,
                "accuracy": round(accuracy, 1) if accuracy > 0 else round(confidence * 100, 1),
                "wpm": round(wpm, 0),
                "status": "success" if success else "failed"
            })

            if accuracy > 0:
                accuracy_values.append(accuracy)
            if wpm > 0:
                wpm_values.append(wpm)

        except Exception as exc:  
            logging.warning("Dosya okuma hatası %s: %s", json_file, exc)
            continue

    total_sessions = len(sessions)
    avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0
    avg_wpm = sum(wpm_values) / len(wpm_values) if wpm_values else 0

    accuracy_trend = []
    if sessions:
        recent_sessions = sessions[:7]
        accuracy_trend = [session["accuracy"] for session in recent_sessions]
        while len(accuracy_trend) < 7:
            accuracy_trend.append(accuracy_trend[-1] if accuracy_trend else avg_accuracy)

    return {
        "total_sessions": total_sessions,
        "avg_accuracy": round(avg_accuracy, 1),
        "avg_wpm": round(avg_wpm, 0),
        "sessions": sessions[:20],
        "accuracy_trend": accuracy_trend[:7]
    }


def extract_keystroke_features(keystrokes: List[KeystrokeData]) -> dict:
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
        "key_events": [],
        "down_down_latencies": [],
        "up_down_latencies": [],
        "up_up_latencies": [],
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
    previous_keydown_time = None
    previous_key = None
    previous_key_code = None
    previous_keys_history = []

    typed_sequence = []
    last_event_time = None

    for idx, event in enumerate(keystrokes):
        key = event.key
        key_code = event.code
        event_type = event.event_type
        timestamp = event.timestamp

        if last_event_time is not None:
            inter_key_delay = timestamp - last_event_time
            features["inter_key_delays"].append(inter_key_delay)

        last_event_time = timestamp

        if event_type == "keydown":
            keydown_events[key] = {
                "timestamp": timestamp,
                "key": key,
                "code": key_code
            }
            features["key_down_times"].append(timestamp)

            key_event = {
                "event_type": "keydown",
                "timestamp": timestamp,
                "key_char": key,
                "key_code": key_code
            }
            features["key_events"].append(key_event)

            if previous_keydown_time is not None:
                down_down_latency = timestamp - previous_keydown_time
                features["down_down_latencies"].append({
                    "from_key": previous_key,
                    "from_code": previous_key_code,
                    "to_key": key,
                    "to_code": key_code,
                    "latency": down_down_latency
                })

            previous_keydown_time = timestamp

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
                up_down_latency = timestamp - previous_keyup_time
                features["up_down_latencies"].append({
                    "from_key": previous_key,
                    "to_key": key,
                    "latency": up_down_latency
                })
                features["flight_times"].append(up_down_latency)

                if len(features["flight_times"]) > 1:
                    speed_change = up_down_latency - features["flight_times"][-2]
                    features["speed_changes"].append(speed_change)

                if up_down_latency > 300:
                    features["pause_events"].append({
                        "duration": up_down_latency,
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
                prev_key = previous_keys_history[-2]
                prev_keyup_info = keyup_events.get(prev_key, {}).get("timestamp", timestamp)
                trigram_time = timestamp - prev_keyup_info
                features["trigram_times"].append({
                    "keys": previous_keys_history[-2:] + [key],
                    "time": trigram_time
                })

            if len(previous_keys_history) >= 3:
                prev_key = previous_keys_history[-3]
                prev_keyup_info = keyup_events.get(prev_key, {}).get("timestamp", timestamp)
                four_gram_time = timestamp - prev_keyup_info
                features["four_gram_times"].append({
                    "keys": previous_keys_history[-3:] + [key],
                    "time": four_gram_time
                })

            if len(features["shift_usage"]) > 0 and features["shift_usage"][-1]["next_key"] is None:
                features["shift_usage"][-1]["next_key"] = key

            previous_key_code = key_code

        elif event_type == "keyup":
            keyup_events[key] = {
                "timestamp": timestamp,
                "key": key,
                "code": key_code
            }
            features["key_up_times"].append(timestamp)

            key_event = {
                "event_type": "keyup",
                "timestamp": timestamp,
                "key_char": key,
                "key_code": key_code
            }
            features["key_events"].append(key_event)

            if previous_keyup_time is not None:
                up_up_latency = timestamp - previous_keyup_time
                features["up_up_latencies"].append({
                    "from_key": previous_key,
                    "to_key": key,
                    "latency": up_up_latency
                })

            if key in {"Shift", "Backspace"}:
                if key in keydown_events:
                    del keydown_events[key]
                continue

            keydown_info = keydown_events.get(key)
            if keydown_info:
                dwell_time = timestamp - keydown_info["timestamp"]
                features["dwell_times"].append(dwell_time)

                pressure_estimate = min(dwell_time / 200.0, 1.0)
                features["key_pressure_estimate"].append({
                    "key": key,
                    "key_code": key_code,
                    "pressure": pressure_estimate,
                    "dwell": dwell_time
                })

                features["hold_times"].append({
                    "key": key,
                    "key_code": key_code,
                    "dwell_time": dwell_time,
                    "keydown_timestamp": keydown_info["timestamp"],
                    "keyup_timestamp": timestamp
                })

                del keydown_events[key]

            previous_keyup_time = timestamp
            previous_key = key
            previous_key_code = key_code
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
    stats = {}

    metrics = ["dwell_times", "flight_times", "inter_key_delays"]

    if features.get("down_down_latencies"):
        down_down_values = [d["latency"] for d in features["down_down_latencies"]]
        data = np.array(down_down_values)
        stats["down_down_latency_mean"] = float(np.mean(data))
        stats["down_down_latency_std"] = float(np.std(data))
        stats["down_down_latency_median"] = float(np.median(data))

    if features.get("up_up_latencies"):
        up_up_values = [u["latency"] for u in features["up_up_latencies"]]
        data = np.array(up_up_values)
        stats["up_up_latency_mean"] = float(np.mean(data))
        stats["up_up_latency_std"] = float(np.std(data))
        stats["up_up_latency_median"] = float(np.median(data))

    if features.get("up_down_latencies"):
        up_down_values = [u["latency"] for u in features["up_down_latencies"]]
        data = np.array(up_down_values)
        stats["up_down_latency_mean"] = float(np.mean(data))
        stats["up_down_latency_std"] = float(np.std(data))
        stats["up_down_latency_median"] = float(np.median(data))

    for metric in metrics:
        metric_values = features.get(metric, [])
        if metric_values:
            data = np.array(metric_values)

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
    def safe_get_key(item: dict, primary: str, fallback: str) -> str:
        val = item.get(primary)
        if val is None or val == "":
            return item.get(fallback, "")
        return val

    with open(filepath, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

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

        hold_times = features.get("hold_times", [])
        if hold_times:
            for idx, hold in enumerate(hold_times):
                key_char = hold.get("key", "")
                dwell = hold.get("dwell_time", 0.0)
                additional = {
                    "keydown_timestamp": hold.get("keydown_timestamp"),
                    "keyup_timestamp": hold.get("keyup_timestamp"),
                    "key_code": hold.get("key_code")
                }
                writer.writerow([
                    username,
                    "timing",
                    "dwell_time",
                    key_char,
                    "",
                    dwell,
                    json.dumps(additional, ensure_ascii=False),
                    idx
                ])
        else:
            for idx, dwell in enumerate(features.get("dwell_times", [])):
                writer.writerow([
                    username,
                    "timing",
                    "dwell_time",
                    "",
                    "",
                    dwell,
                    "",
                    idx
                ])

        for idx, delay in enumerate(features.get("inter_key_delays", [])):
            writer.writerow([
                username,
                "timing",
                "inter_key_delay",
                "",
                "",
                delay,
                "",
                idx
            ])

        for idx, down_down in enumerate(features.get("down_down_latencies", [])):
            from_key = safe_get_key(down_down, "from_key", "from_code")
            to_key = safe_get_key(down_down, "to_key", "to_code")

            additional = {
                "from_code": down_down.get("from_code"),
                "to_code": down_down.get("to_code")
            }

            writer.writerow([
                username,
                "timing",
                "down_down_latency",
                from_key,
                to_key,
                down_down.get("latency", 0.0),
                json.dumps(additional, ensure_ascii=False),
                idx
            ])

        for idx, up_down in enumerate(features.get("up_down_latencies", [])):
            from_key = safe_get_key(up_down, "from_key", "from_code")
            to_key = safe_get_key(up_down, "to_key", "to_code")

            writer.writerow([
                username,
                "timing",
                "up_down_latency",
                from_key,
                to_key,
                up_down.get("latency", 0.0),
                "",
                idx
            ])

        for idx, up_up in enumerate(features.get("up_up_latencies", [])):
            from_key = safe_get_key(up_up, "from_key", "from_code")
            to_key = safe_get_key(up_up, "to_key", "to_code")

            writer.writerow([
                username,
                "timing",
                "up_up_latency",
                from_key,
                to_key,
                up_up.get("latency", 0.0),
                "",
                idx
            ])

        for idx, key_event in enumerate(features.get("key_events", [])):
            writer.writerow([
                username,
                "raw_event",
                key_event.get("event_type", ""),
                key_event.get("key_char", ""),
                key_event.get("key_code", ""),
                key_event.get("timestamp", 0.0),
                "",
                idx
            ])

        for idx, digraph in enumerate(features.get("digraph_times", [])):
            writer.writerow([
                username,
                "n-gram",
                "digraph",
                digraph.get("from", ""),
                digraph.get("to", ""),
                digraph.get("time", 0.0),
                "",
                idx
            ])

        for idx, trigram in enumerate(features.get("trigram_times", [])):
            keys_str = "->".join(trigram.get("keys", []))
            writer.writerow([
                username,
                "n-gram",
                "trigram",
                keys_str,
                "",
                trigram.get("time", 0.0),
                "",
                idx
            ])

        for idx, four_gram in enumerate(features.get("four_gram_times", [])):
            keys_str = "->".join(four_gram.get("keys", []))
            writer.writerow([
                username,
                "n-gram",
                "four_gram",
                keys_str,
                "",
                four_gram.get("time", 0.0),
                "",
                idx
            ])

        writer.writerow([
            username,
            "behavior",
            "backspace_count",
            "",
            "",
            features.get("backspace_count", 0),
            "",
            0
        ])

        for idx, latency in enumerate(features.get("backspace_latencies", [])):
            writer.writerow([
                username,
                "behavior",
                "backspace_latency",
                "",
                "",
                latency,
                "",
                idx
            ])

        for idx, correction in enumerate(features.get("error_corrections", [])):
            writer.writerow([
                username,
                "behavior",
                "error_correction",
                correction.get("char", ""),
                "",
                correction.get("latency", 0.0),
                json.dumps({"position": correction.get("position")}, ensure_ascii=False),
                idx
            ])

        for idx, pause in enumerate(features.get("pause_events", [])):
            writer.writerow([
                username,
                "rhythm",
                "pause",
                pause.get("after_key", ""),
                pause.get("before_key", ""),
                pause.get("duration", 0.0),
                json.dumps({"position": pause.get("position")}, ensure_ascii=False),
                idx
            ])

        for idx, speed_change in enumerate(features.get("speed_changes", [])):
            writer.writerow([
                username,
                "rhythm",
                "speed_change",
                "",
                "",
                speed_change,
                "",
                idx
            ])

        for idx, pressure in enumerate(features.get("key_pressure_estimate", [])):
            writer.writerow([
                username,
                "pressure",
                "key_pressure",
                pressure.get("key", ""),
                "",
                pressure.get("pressure", 0.0),
                json.dumps({"dwell": pressure.get("dwell")}, ensure_ascii=False),
                idx
            ])

        if "statistics" in features:
            for stat_name, stat_value in features["statistics"].items():
                writer.writerow([
                    username,
                    "statistics",
                    stat_name,
                    "",
                    "",
                    stat_value,
                    "",
                    0
                ])

        if "rhythm_metrics" in features:
            for metric_name, metric_value in features["rhythm_metrics"].items():
                writer.writerow([
                    username,
                    "rhythm_metrics",
                    metric_name,
                    "",
                    "",
                    metric_value,
                    "",
                    0
                ])

        if "typing_speed" in features:
            for speed_name, speed_value in features["typing_speed"].items():
                writer.writerow([
                    username,
                    "typing_speed",
                    speed_name,
                    "",
                    "",
                    speed_value,
                    "",
                    0
                ])

    logging.info("Biometrik özellikler CSV'ye kaydedildi: %s", filepath)


def summarize_letter_errors(features: dict, top_n: int = 3) -> dict:
    corrections = features.get("error_corrections", [])
    counter: Counter[str] = Counter()

    for correction in corrections:
        char = correction.get("char")
        if not char:
            continue
        counter[char] += 1

    total = sum(counter.values())

    summary = []
    for char, count in counter.most_common(top_n):
        ratio = count / total if total else 0.0
        summary.append({
            "char": char,
            "count": int(count),
            "ratio": round(ratio, 3)
        })

    return {
        "total_corrections": int(total),
        "top_errors": summary
    }

