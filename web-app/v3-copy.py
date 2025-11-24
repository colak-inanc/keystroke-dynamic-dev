import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

load_dotenv()

from app.routes import router 

app = FastAPI()

WEB_APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_APP_DIR / 'static'

if STATIC_DIR.exists():
    app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')
else:
    logging.warning('Static klasoru bulunamadi: %s', STATIC_DIR)

app.include_router(router)
