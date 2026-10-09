import os
import requests
from dotenv import load_dotenv

load_dotenv()

RAILRADAR_API_KEY = os.getenv("RAILRADAR_API_KEY")

BASE_URL = "https://api.railradar.in/v1"


def get_live_train_status(train_number: str):
    if not RAILRADAR_API_KEY:
        raise RuntimeError("RAILRADAR_API_KEY is missing from .env")

    url = f"{BASE_URL}/trains/{train_number}/live"

    headers = {
        "Authorization": f"Bearer {RAILRADAR_API_KEY}"
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("success"):
        raise RuntimeError(
            data.get("message", "RailRadar API returned an error")
        )

    return data["data"]