import json
import logging
import os
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
import numpy as np
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from app.models import LoginRequest, RegisterRequest
from app.training import train_model
from app.services import (
    DATA_DIR,
    extract_keystroke_features,
    get_user_stats_data,
    load_model,
    login_user_legacy,
    prepare_sequence_for_model,
    save_features_to_csv,
    save_login_session,
    summarize_letter_errors,
)

WEB_APP_DIR = Path(__file__).resolve().parent.parent

templates = Jinja2Templates(directory=str(WEB_APP_DIR / "templates"))

router = APIRouter()

# for session management
SESSION_SECRET_KEY = os.getenv("SESSION_SECRET_KEY", "biometric-auth-secret-key-change-in-production")
SESSION_COOKIE_NAME = "biometric_session"
SESSION_MAX_AGE = 3600 * 24  

serializer = URLSafeTimedSerializer(SESSION_SECRET_KEY)


def create_session_token(username: str) -> str:
    return serializer.dumps({"username": username, "authenticated": True})


def verify_session_token(token: str) -> dict | None:
    try:
        data = serializer.loads(token, max_age=SESSION_MAX_AGE)
        if data.get("authenticated") and data.get("username"):
            user_dir = DATA_DIR / data["username"]
            if user_dir.exists():
                return data
        return None
    except (BadSignature, SignatureExpired):
        return None


def get_current_user(request: Request) -> str | None:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    session_data = verify_session_token(token)
    return session_data.get("username") if session_data else None


api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logging.basicConfig(level=logging.INFO)
    logging.warning(
        "GOOGLE_API_KEY bulunamadı. Gemini entegrasyonu devre dışı. .env dosyanıza `GOOGLE_API_KEY` ekleyin."
    )


@router.get("/register", response_class=HTMLResponse)
async def read_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@router.get("/", response_class=HTMLResponse)
async def read_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def read_dashboard(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        "dashboard.html", 
        {"request": request, "username": current_user}
    )


@router.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@router.get("/api/get-text")
async def get_typing_text(round: int = 1):
    if api_key:
        try:
            prompt = (
                "Bana klavye dinamiklerini ölçmek için harf çeşitliliği yüksek, 70-80 kelimelik, "
                "Türkçe, resmi ve akıcı bir paragraf yaz. Metinde herhangi bir noktalama işareti olmasın. "
                "Metindeki tüm harfler küçük olmalı."
                "Sadece metni döndür."
                "Türkçe karakterleri kullan."
                "Klavye dinamiklerini analiz edebileceğim şekilde çeşitli harf kombinasyonları kullan."
            )
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(prompt)
            return {"text": response.text}
        except Exception as exc:  
            logging.warning("Gemini API hatası: %s - Mock data kullanılıyor", exc)

    mock_text_1 = "teknolojinin hızla geliştiği günümüz dünyasında yazılım mühendisliği ve yapay zeka uygulamaları hayatımızın her alanında köklü bir değişim yaratmaktadır sürdürülebilir kalkınma hedefleri doğrultusunda verimliliği artırmak için çağdaş ve yenilikçi çözümler üretmek zorundayız özellikle kapsamlı veri analizi yöntemleri sayesinde karmaşık problemlerin üstesinden gelmek artık çok daha erişilebilir hale gelmiştir bu süreçte profesyonel yaklaşım ve disiplinli çalışma prensipleri mutlak başarının temel anahtarını oluşturmaktadır geleceği şekillendiren bu yenilikler insanlık adına umut verici bir potansiyel taşımaktadır"
    mock_text_2 = "küresel ölçekte yürütülen bilimsel çalışmalar toplumların refah seviyesini yükseltmek adına kritik bir rol oynamaktadır özellikle çevre bilincinin artmasıyla birlikte yenilenebilir enerji kaynaklarına yönelim hız kazanmıştır bu bağlamda rüzgar ve güneş enerjisi gibi alternatif yöntemler karbon ayak izini azaltmak için stratejik bir öneme sahiptir akademik disiplin ve sistematik araştırma teknikleri kullanılarak elde edilen veriler geleceğin inşasında sağlam bir temel oluşturur bilgiye erişimin kolaylaşmasıyla beraber inovasyon süreçleri de ivme kazanarak devam etmektedir"

    selected_text = mock_text_2 if round == 2 else mock_text_1
    
    logging.info(f"Mock data döndürülüyor (Round: {round})")
    return {"text": selected_text}



@router.post("/api/login")
async def login_user(data: LoginRequest):
    try:
        user_dir = DATA_DIR / data.username
        if not user_dir.exists():
            return {
                "status": "error",
                "message": f"Kullanıcı '{data.username}' bulunamadı. Lütfen önce kayıt olun."
            }

        loaded_model = load_model()

        if loaded_model is None:
            logging.warning("Model yüklenemedi, legacy yöntem kullanılıyor")
            return await login_user_legacy(data)

        features = extract_keystroke_features(data.keystrokes)
        letter_summary = summarize_letter_errors(features)
        sequence_data = prepare_sequence_for_model(data.keystrokes, features)

        if sequence_data is None:
            logging.warning("Sequence data is None")
            return await login_user_legacy(data)

        is_valid_input = False
        if isinstance(sequence_data, list):
             if len(sequence_data) == 2 and len(sequence_data[0].shape) == 3:
                 is_valid_input = True
                 logging.info("Multi-Input detected: Time shape %s, Key shape %s", sequence_data[0].shape, sequence_data[1].shape)
        elif isinstance(sequence_data, np.ndarray):
             if len(sequence_data.shape) == 3:
                 is_valid_input = True

        if not is_valid_input:
            logging.warning(
                "Sequence formatı uygun değil: %s",
                type(sequence_data)
            )
            return await login_user_legacy(data)

        predictions = loaded_model.predict(sequence_data, verbose=0)
        logging.info("Model prediction shape: %s", predictions.shape)

        # Get user specific index
        from app.services import get_user_label_index
        target_idx = get_user_label_index(data.username)
        
        window_confidences = []
        
        if target_idx is None:
            # User Not in Learning Map yet (maybe new registration before training completed?)
            logging.warning("User %s not in label map (Training pending?). Fallback to Legacy.", data.username)
            return await login_user_legacy(data)

        if len(predictions.shape) == 1:
             # Should not happen with batch predict usually, but handle just in case
             window_confidences = [predictions[target_idx]]
        elif predictions.shape[1] == 1: 
             # Binary Classification (0=Other, 1=User) - Assuming 1 is target if map says so?
             # Complex if map has multiple binary models. Assuming Multi-Class Softmax here.
             window_confidences = predictions.flatten()
        else:
             # Multi-class Softmax: Take probability of the SPECIFIC user's class
             # predictions shape: (windows, num_classes)
             window_confidences = predictions[:, target_idx]

        raw_output = float(np.mean(window_confidences))
        min_conf = float(np.min(window_confidences))
        max_conf = float(np.max(window_confidences))
        
        logging.info("Window confidences: %s", window_confidences)
        logging.info("Voting Result -> Mean: %.4f, Min: %.4f, Max: %.4f", raw_output, min_conf, max_conf)

        confidence = max(0.0, min(1.0, raw_output))
        logging.info("Model raw output: %.8f, Confidence: %.8f (%.4f%%)", raw_output, confidence, confidence * 100)
        THRESHOLD = 0.75

        if confidence >= THRESHOLD:
            logging.info("Başarılı giriş (Model): %s (güven: %.2f)", data.username, confidence)

            try:
                save_login_session(data.username, data.keystrokes, features, confidence, True)
            except Exception as exc:  
                logging.warning("Giriş oturumu kaydedilemedi: %s", exc)

            session_token = create_session_token(data.username)
            response = JSONResponse(content={
                "status": "success",
                "username": data.username,
                "confidence": confidence,
                "message": "Kimlik doğrulandı",
                "letter_error_summary": letter_summary
            })
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_token,
                max_age=SESSION_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=False  
            )
            return response

        logging.warning("Başarısız giriş denemesi (Model): %s (güven: %.2f)", data.username, confidence)
        try:
            save_login_session(data.username, data.keystrokes, features, confidence, False)
        except Exception as exc: 
            logging.warning("Giriş oturumu kaydedilemedi: %s", exc)
        return {
            "status": "error",
            "message": f"Klavye dinamikleri eşleşmiyor (güven: {confidence*100:.1f}%). Bu hesap size ait olmayabilir!",
            "letter_error_summary": letter_summary
        }

    except Exception as exc: 
        logging.error("Login hatası: %s", exc)
        raise HTTPException(status_code=500, detail=f"Doğrulama hatası: {exc}") from exc


@router.get("/api/user-stats")
async def get_user_stats(request: Request, username: str | None = None):
    current_user = get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Oturum açmanız gerekiyor")
    
    try:
        return get_user_stats_data(current_user)
    except Exception as exc:  
        logging.error("Kullanıcı istatistikleri hatası: %s", exc)
        raise HTTPException(status_code=500, detail=f"İstatistikler alınamadı: {exc}") from exc


@router.get("/api/me")
async def get_current_user_info(request: Request):
    current_user = get_current_user(request)
    if not current_user:
        raise HTTPException(status_code=401, detail="Oturum açmanız gerekiyor")
    
    return {"username": current_user, "authenticated": True}


@router.post("/api/logout")
async def logout(request: Request):
    response = JSONResponse(content={"status": "success", "message": "Oturum sonlandırıldı"})
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@router.post("/api/register")
async def register_user(data: RegisterRequest, background_tasks: BackgroundTasks):
    try:
        DATA_DIR.mkdir(exist_ok=True)
        user_dir = DATA_DIR / data.username
        user_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        
        def calculate_metadata(keystrokes, features, provided_metadata=None):
            if provided_metadata:
                return provided_metadata.dict()
            return {
                "accuracy_rate": features.get("typing_speed", {}).get("accuracy_rate", 0.0) * 100,
                "wpm": features.get("typing_speed", {}).get("words_per_minute", 0.0),
                "correct_keys": len([k for k in keystrokes if k.event_type == "keydown" and k.key != "Backspace"]),
                "total_keys": len(keystrokes)
            }

        features_1 = extract_keystroke_features(data.keystrokes)
        letter_summary_1 = summarize_letter_errors(features_1)
        session_metadata_1 = calculate_metadata(data.keystrokes, features_1, data.metadata)
        
        features_list = [features_1]
        
        features_2 = None
        letter_summary_2 = None
        session_metadata_2 = None
        
        if data.keystrokes_2:
            features_2 = extract_keystroke_features(data.keystrokes_2)
            letter_summary_2 = summarize_letter_errors(features_2)
            session_metadata_2 = calculate_metadata(data.keystrokes_2, features_2, data.metadata_2)
            features_list.append(features_2)
        
        raw_data = {
            "username": data.username,
            "email": data.email,
            "timestamp": timestamp,
            "session_type": "register",
            "success": True,
            "total_keystrokes": len(data.keystrokes),
            "keystrokes": [k.dict() for k in data.keystrokes],
            "session_metadata": session_metadata_1,
            "letter_error_summary": letter_summary_1
        }
        
        if data.keystrokes_2:
            raw_data.update({
                "total_keystrokes_2": len(data.keystrokes_2),
                "keystrokes_2": [k.dict() for k in data.keystrokes_2],
                "session_metadata_2": session_metadata_2,
                "letter_error_summary_2": letter_summary_2
            })

        raw_data_file = user_dir / f"raw_data_{timestamp}.json"
        with open(raw_data_file, "w", encoding="utf-8") as file:
            json.dump(raw_data, file, indent=2, ensure_ascii=False)

        features_file = user_dir / f"features_{timestamp}.csv"
        save_features_to_csv(features_list, features_file, data.username)

        count_msg = f"{len(data.keystrokes)}"
        if data.keystrokes_2:
            count_msg += f" + {len(data.keystrokes_2)}"

        logging.info("%s için veri kaydedildi: %s tuş vuruşu", data.username, count_msg)
        
        # Trigger training in background
        background_tasks.add_task(train_model)
        logging.info("Arka planda model eğitimi tetiklendi.")

        return {
            "status": "success",
            "message": "Kayıt ve Biyometrik Veri (2 Set) Alındı" if data.keystrokes_2 else "Kayıt ve Biyometrik Veri Alındı",
            "username": data.username,
            "keystroke_count": len(data.keystrokes),
            "files_saved": [str(raw_data_file), str(features_file)],
            "letter_error_summary": letter_summary_1
        }

    except Exception as exc:  
        logging.error("Veri kaydetme hatası: %s", exc)
        raise HTTPException(status_code=500, detail=f"Veri kaydedilemedi: {exc}") from exc

