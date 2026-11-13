import os
import json
import requests
from datetime import datetime

# 1. Načítanie API kľúča z prostredia
API_KEY = os.environ.get("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("Chýba API kľúč! Nastav premennú API_FOOTBALL_KEY.")

# 2. Získanie dnešných zápasov z API-Football
dnes = datetime.now().strftime('%Y-%m-%d')
url = f"https://v3.football.api-sports.io/fixtures?date={dnes}"
headers = {"x-apisports-key": API_KEY}

response = requests.get(url, headers=headers)
response.raise_for_status()
fixtures = response.json().get("response", [])

if not fixtures:
    print(f"Na {dnes} sa nenašli žiadne zápasy.")
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    exit()

print(f"Našlo sa {len(fixtures)} zápasov na {dnes}.")

# 3. Generovanie predikcií (placeholder)
vsetky_predikcie = []
for match in fixtures:
    home_team = match['teams']['home']['name']
    away_team = match['teams']['away']['name']
    
    # TODO: Nahraďte tento blok skutočným modelom
    predikcia = {
        "datum": dnes,
        "domaci": home_team,
        "hostia": away_team,
        "predikcia_1X2": {
            "vyhra_doma": 0.333,
            "remiza": 0.333,
            "vyhra_hosti": 0.333
        }
    }
    vsetky_predikcie.append(predikcia)

# 4. Uloženie predikcií do JSON súboru
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(vsetky_predikcie, f, ensure_ascii=False, indent=2)

print(f"Predikcie boli uložené do predictions.json ({len(vsetky_predikcie)} zápasov).")
