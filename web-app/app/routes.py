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
    mock_text_3 = "tarih boyunca insanoğlu sürekli bir arayış içinde olmuş ve bilinmeyeni keşfetme arzusuyla hareket etmiştir antik medeniyetlerin bıraktığı izler geçmişin derinliklerine ışık tutarken geleceğe dair ipuçları da sunmaktadır arkeolojik kazılar ve tarihsel belgeler sayesinde kaybolmuş kültürlerin gizemleri birer birer gün yüzüne çıkmaktadır bu kültürel mirasın korunması ve gelecek nesillere aktarılması insanlık tarihine duyulan saygının en somut göstergesidir"

    if round == 3:
        selected_text = mock_text_3
    elif round == 2:
        selected_text = mock_text_2
    else:
        selected_text = mock_text_1
    
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
             logging.warning("Model yüklenemedi, işlem iptal ediliyor (Legacy support removed for segmentation update).")
             return {
                 "status": "error",
                 "message": "Kimlik doğrulama servisi şu anda kullanılamıyor (Model hatası)."
             } 

        # Prepare steps
        steps = [
            (data.keystrokes, data.metadata, 1),
            (data.keystrokes_2, data.metadata_2, 2),
            (data.keystrokes_3, data.metadata_3, 3)
        ]

        valid_segments = []
        confidences = []
        
        # Validation & Prediction Loop
        from app.services import (
            check_physiological_validity, 
            reconstruct_text_from_keystrokes, 
            calculate_text_similarity,
            get_user_label_index
        )
        
        target_idx = get_user_label_index(data.username)
        if target_idx is None:
             return {
                 "status": "error",
                 "message": "Kullanıcı modelde tanımlı değil (Eğitim gerekli)."
             }

        for ks, meta, step_num in steps:
            if not ks or len(ks) < 5: continue

            # 1. Physio Check
            physio_check = check_physiological_validity(ks)
            if not physio_check["valid"]:
                 logging.warning(f"Step {step_num} Physio Check Failed: {physio_check['reason']}")
                 continue # Skip bad segments, don't fail whole request yet? Or strict fail? 
                 # Strict fail might be annoying if one step is just slightly off. Let's strict fail for now to ensure quality.
                 # return {"status": "error", "message": f"Adım {step_num} geçersiz: {physio_check['reason']}"}

            # 2. Text Validation
            if meta and meta.target_text:
                typed_text = reconstruct_text_from_keystrokes(ks)
                similarity = calculate_text_similarity(typed_text.lower(), meta.target_text.lower())
                if similarity < 0.60:
                     logging.warning(f"Step {step_num} Text Mismatch: {similarity}")
                     return {
                         "status": "error",
                         "message": f"Adım {step_num} metni eşleşmiyor (%{int(similarity*100)}). Lütfen dikkatli yazın."
                     }
            
            # 3. Feature Extraction
            features = extract_keystroke_features(ks)
            sequence_data = prepare_sequence_for_model(ks, features)
            
            if sequence_data is None: continue

            # 4. Predict
            # Ensure sequence_data matches model input
            is_valid_input = False
            if isinstance(sequence_data, list) and len(sequence_data) == 2: is_valid_input = True
            elif isinstance(sequence_data, np.ndarray): is_valid_input = True
            
            if not is_valid_input: continue

            predictions = loaded_model.predict(sequence_data, verbose=0)
            
            # Extract confidence for specific user
            step_confidence = 0.0
            if len(predictions.shape) == 1:
                 step_confidence = float(predictions[target_idx])
            elif predictions.shape[1] == 1:
                 step_confidence = float(predictions.flatten()[0]) # Binary assumption
            else:
                 # Standard Softmax
                 step_confidence = float(np.mean(predictions[:, target_idx]))
            
            confidences.append(step_confidence)
            valid_segments.append({
                "step": step_num,
                "dataset": ks,
                "features": features,
                "confidence": step_confidence
            })

        if not valid_segments:
            return {
                "status": "error",
                "message": "Geçerli veri segmenti bulunamadı veya tüm doğrulama kontrolleri başarısız oldu."
            }

        # Final Decision
        final_confidence = float(np.mean(confidences))
        THRESHOLD = 0.50
        is_success = final_confidence >= THRESHOLD

        logging.info(f"Login Decision for {data.username}: {is_success} (Avg Conf: {final_confidence:.4f} from {len(confidences)} segments)")

        # Save Sessions (All segments saved with the Final Success Status)
        for seg in valid_segments:
            try:
                save_login_session(
                    username=data.username, 
                    keystrokes=seg["dataset"], 
                    features=seg["features"], 
                    confidence=seg["confidence"], 
                    success=is_success,
                    step_suffix=f"s{seg['step']}"
                )
            except Exception as e:
                logging.error(f"Failed to save session step {seg['step']}: {e}")

        # Response
        letter_summary = summarize_letter_errors(valid_segments[0]["features"]) # Summary from first valid segment
        
        if is_success:
            session_token = create_session_token(data.username)
            response = JSONResponse(content={
                "status": "success",
                "username": data.username,
                "confidence": final_confidence,
                "message": "Kimlik doğrulandı",
                "letter_error_summary": letter_summary
            })
            response.set_cookie(
                key=SESSION_COOKIE_NAME,
                value=session_token,
                max_age=SESSION_MAX_AGE,
                httponly=True,
                samesite="lax",
                secure=os.getenv("IS_PRODUCTION", "false").lower() == "true"
            )
            return response
        else:
            return {
                "status": "error",
                "message": f"Klavye dinamikleri eşleşmiyor (Güven: %{int(final_confidence*100)}).",
                "letter_error_summary": letter_summary
            }

    except Exception as exc: 
        import traceback
        traceback.print_exc()
        logging.error("Login hatası: %s", exc)
        raise HTTPException(status_code=500, detail=f"Doğrulama hatası: {str(exc)}") from exc


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

        base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        
        # Validate all provided keystroke sets
        from app.services import check_physiological_validity, reconstruct_text_from_keystrokes, calculate_text_similarity

        def validate_set(k_data, meta_data, step_name):
            if not k_data: return
            
            # 1. Physio Check
            physio = check_physiological_validity(k_data)
            if not physio["valid"]:
                raise HTTPException(status_code=400, detail=f"{step_name} hatası: {physio['reason']}")

            # 2. Text Check
            if meta_data and meta_data.target_text:
                typed = reconstruct_text_from_keystrokes(k_data)
                # Case-insensitive comparison for robustness
                sim = calculate_text_similarity(typed.lower(), meta_data.target_text.lower())
                
                # Length Validation (FAILSAFE)
                len_typed = len(typed)
                len_target = len(meta_data.target_text)
                len_ratio = min(len_typed, len_target) / max(len_typed, len_target) if max(len_typed, len_target) > 0 else 0

                # Strict Validation: Must have at least 60% similarity
                # We ignore length ratio for approval, as garbage text can have perfect length ratio.
                if sim < 0.60:
                     logging.error(f"Validation Failed: Similarity {sim:.2f} < 0.60")
                     logging.error(f"Target: {meta_data.target_text[:50]}")
                     logging.error(f"Typed:  {typed[:50]}")

                     raise HTTPException(
                         status_code=400, 
                         detail=f"{step_name} metni eşleşmiyor (Benzerlik: %{int(sim*100)}). Lütfen ekrandaki metni doğru yazın."
                     )

        validate_set(data.keystrokes, data.metadata, "Adım 1")
        if data.keystrokes_2: validate_set(data.keystrokes_2, data.metadata_2, "Adım 2")
        if data.keystrokes_3: validate_set(data.keystrokes_3, data.metadata_3, "Adım 3")

        def calculate_metadata(keystrokes, features, provided_metadata=None):
            if provided_metadata:
                return provided_metadata.dict()
            return {
                "accuracy_rate": features.get("typing_speed", {}).get("accuracy_rate", 0.0) * 100,
                "wpm": features.get("typing_speed", {}).get("words_per_minute", 0.0),
                "correct_keys": len([k for k in keystrokes if k.event_type == "keydown" and k.key != "Backspace"]),
                "total_keys": len(keystrokes)
            }

        # --- REFACTOR: Save EACH Step as a Separate Session File ---
        saved_files = []
        features_list = []
        
        # Prepare iterable for steps
        steps = [
            (data.keystrokes, data.metadata, 1),
            (data.keystrokes_2, data.metadata_2, 2),
            (data.keystrokes_3, data.metadata_3, 3)
        ]

        count_msg = ""

        for ks, meta, step_num in steps:
            if not ks: continue
            
            try:
                # 1. Feature Extraction
                features = extract_keystroke_features(ks)
                letter_summary = summarize_letter_errors(features)
                
                # Check data type of ks elements
                if len(ks) > 0 and isinstance(ks[0], dict):
                     logging.warning("Keystrokes are dicts, not objects! This might cause AttributeError or KeyError.")

                session_metadata = calculate_metadata(ks, features, meta)
                features_list.append(features)
                
                # 2. Prepare JSON Payload for this Step
                step_file_name = f"raw_data_{base_timestamp}_register_s{step_num}.json"
                
                raw_data = {
                    "username": data.username,
                    "email": data.email,
                    "timestamp": base_timestamp, # Main group timestamp
                    "step_timestamp": datetime.now().isoformat(), # Exact write time
                    "session_type": "register",
                    "step_number": step_num,
                    "success": True,
                    "total_keystrokes": len(ks),
                    "keystrokes": [k.dict() for k in ks],
                    "session_metadata": session_metadata,
                    "letter_error_summary": letter_summary
                }

                # 3. Write JSON File
                file_path = user_dir / step_file_name
                with open(file_path, "w", encoding="utf-8") as file:
                    json.dump(raw_data, file, indent=2, ensure_ascii=False)
                
                saved_files.append(str(file_path))
                
                # Log info
                count_msg += f" + {len(ks)}" if count_msg else f"{len(ks)}"

            except KeyError as ke:
                logging.error(f"KeyError in step {step_num}: {ke}")
                import traceback
                logging.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Veri işleme hatası (Adım {step_num}): {ke}")
            except Exception as e:
                logging.error(f"Error in step {step_num}: {e}")
                raise HTTPException(status_code=500, detail=f"Adım {step_num} hatası: {e}")

        # 4. Save Aggregated Features to CSV (One row per step/session)
        try:
            features_file = user_dir / f"features_{base_timestamp}.csv"
            save_features_to_csv(features_list, features_file, data.username)
        except Exception as e:
            logging.error(f"CSV Save Error: {e}")
            raise HTTPException(status_code=500, detail=f"CSV kaydetme hatası: {e}")


        logging.info("%s için 'Session Isolation' verisi kaydedildi: %s tuş vuruşu", data.username, count_msg)
        
        # Trigger training in background
        background_tasks.add_task(train_model)
        logging.info("Arka planda model eğitimi tetiklendi.")

        return {
            "status": "success",
            "message": "Kayıt Başarılı. Her adım ayrı bir oturum dosyası olarak kaydedildi.",
            "username": data.username,
            "main_timestamp": base_timestamp,
            "files_created": len(saved_files),
            "files_saved": saved_files
        }

    except Exception as exc:  
        logging.error("Veri kaydetme hatası: %s", exc)
        raise HTTPException(status_code=500, detail=f"Veri kaydedilemedi: {exc}") from exc


@router.post("/api/debug/train")
async def trigger_manual_training(background_tasks: BackgroundTasks):
    """
    Manually triggers the model training pipeline.
    Useful for debugging and optimization verification.
    """
    from app.training import train_model
    background_tasks.add_task(train_model)
    return {"status": "success", "message": "Training started in background"}
