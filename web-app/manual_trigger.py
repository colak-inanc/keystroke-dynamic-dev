
import requests
import sys
import logging
import time

# Configure logging to mimic the app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_local_training():
    """
    Runs the training pipeline directly in this process.
    Logs will appear in THIS terminal.
    """
    print("\n🚀 Starting DIRECT training (Local Mode)...")
    print("---------------------------------------------")
    print("ℹ️  Logs will appear here. Please wait...")
    
    try:
        from app.training import train_model
        train_model()
        print("\n✅ Direct training completed successfully.")
    except ImportError:
        print("❌ Error: Could not import app.training. Make sure you run this from the 'web-app' directory.")
    except Exception as e:
        print(f"❌ Critical Error during direct training: {e}")

def trigger_via_api():
    """
    Triggers training on the running server (Logs appear in server terminal).
    """
    url = "http://localhost:8000/api/debug/train"
    print(f"Attempting to trigger training via API at {url}...")
    try:
        response = requests.post(url, timeout=2)
        if response.status_code == 200:
            print("✅ Success: Training started via API (Background Task).")
            print("Response:", response.json())
        else:
            print(f"⚠️ API returned error: {response.status_code}")
    except Exception as e:
        print(f"⚠️ API Unreachable: {e}")

if __name__ == "__main__":
    # Default to Local Training so user sees the logs
    # If you purely want to test the API endpoint, uncomment the line below:
    # trigger_via_api()
    
    run_local_training()
    
    # Preventing immediate exit so user can read logs if opened via click
    # time.sleep(2)
