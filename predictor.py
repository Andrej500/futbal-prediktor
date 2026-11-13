import os
import json
import requests
import pandas as pd
from datetime import datetime

# pyfootix (importuje sa ako footix)
from footix.data_io.footballdata import ScrapFootballData
from footix.models.basic_poisson import PoissonModel

# ---------------------------------------------------------------------------
# 1. Tréning modelu na historických dátach (Premier League 2024/25)
# ---------------------------------------------------------------------------
print("Sťahujem historické dáta pre Premier League 2024/25...")
dataset = ScrapFootballData(
    competition="ENG Premier League",   # kľúč pyfootix pre Premier League
    season="2024-2025",
    path="./data",
    force_reload=False
).get_fixtures()

dataset["date"] = pd.to_datetime(dataset["date"], dayfirst=True)
dataset = dataset.sort_values("date").reset_index(drop=True)

print(f"Natrénujem PoissonModel na {len(dataset)} zápasoch...")
# n_teams = počet tímov v lige (Premier League má 20)
model = PoissonModel(n_teams=20, n_goals=20)
model.fit(X_train=dataset)
print("Model je natrénovaný.")

# ---------------------------------------------------------------------------
# 2. Získanie dnešných zápasov z API-Football
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("API_FOOTBALL_KEY")
if not API_KEY:
    raise ValueError("Chýba API kľúč! Nastav premennú API_FOOTBALL_KEY.")

dnes = datetime.now().strftime('%Y-%m-%d')
url = f"https://v3.football.api-sports.io/fixtures?date={dnes}&league=39&season=2025"
headers = {"x-apisports-key": API_KEY}

response = requests.get(url, headers=headers)
response.raise_for_status()
fixtures = response.json().get("response", [])

if not fixtures:
    print(f"Na {dnes} sa v Premier League nenašli žiadne zápasy.")
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    exit()

print(f"Našlo sa {len(fixtures)} zápasov Premier League na {dnes}.")

# ---------------------------------------------------------------------------
# 3. Generovanie predikcií
# ---------------------------------------------------------------------------
vsetky_predikcie = []

for match in fixtures:
    home_team = match['teams']['home']['name']
    away_team = match['teams']['away']['name']

    try:
        # Model vráti GoalMatrix, z ktorého získame 1X2 pravdepodobnosti
        probas = model.predict(home_team=home_team, away_team=away_team).return_probas()

        predikcia = {
            "datum": dnes,
            "domaci": home_team,
            "hostia": away_team,
            "predikcia_1X2": {
                "vyhra_doma": round(float(probas[0]), 3),
                "remiza": round(float(probas[1]), 3),
                "vyhra_hosti": round(float(probas[2]), 3)
            }
        }
        vsetky_predikcie.append(predikcia)

    except Exception as e:
        print(f"Chyba pri predikcii {home_team} vs {away_team}: {e}")
        # Ak model tím nepozná (napr. nováčik alebo iný názov), zápas sa preskočí.
        continue

# ---------------------------------------------------------------------------
# 4. Uloženie výsledkov
# ---------------------------------------------------------------------------
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(vsetky_predikcie, f, ensure_ascii=False, indent=2)

print(f"Predikcie boli uložené do predictions.json ({len(vsetky_predikcie)} zápasov).")
