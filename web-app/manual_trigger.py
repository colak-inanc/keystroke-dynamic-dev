import logging
import sys
import os

# Mevcut dizini path'e ekle ki app modülü bulunsun
sys.path.append(os.getcwd())

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    from app.training import train_model
except ImportError as e:
    print(f"Hata: 'app.training' modülü yüklenemedi. Lütfen scripti 'web-app' klasörü içinden çalıştırdığınızdan emin olun.")
    print(f"Detay: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("\nManuel Model Eğitimi Tetikleniyor...")
    print("=========================================")
    print("Lütfen bekleyin, veriler işleniyor ve model eğitiliyor...")
    
    try:
        # Eğitimi başlat
        train_model()
        
        print("\nİŞLEM BAŞARILI!")
        print("=========================================")
        print("Model başarıyla eğitildi ve kaydedildi.")
        print("Web sunucusu (servis) dosya değişimini algılayıp yeni modeli otomatik yükleyecektir.")
        
    except Exception as e:
        print("\nBİR HATA OLUŞTU!")
        print("=========================================")
        logging.error("Eğitim sırasında beklenmeyen bir hata:", exc_info=True)
