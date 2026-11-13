import os
import json
import requests
import pandas as pd
from datetime import datetime
from poisson_odds import Poisson
from scipy.stats import poisson
import numpy as np

# ---------------------------------------------------------------------------
# RIEŠENIE 1: Priamy import funkcie z edginghockeyscraper
# ---------------------------------------------------------------------------
try:
    from edginghockeyscraper import get_league_schedule
except ImportError:
    # Ak by priamy import zlyhal, skúsime alternatívnu cestu v balíčku
    try:
        from edginghockeyscraper.scraper import get_league_schedule
    except ImportError:
        raise ImportError(
            "Nepodarilo sa importovať get_league_schedule z edginghockeyscraper. "
            "Skontrolujte verziu balíčka (pip show edginghockeyscraper)."
        )

# ---------------------------------------------------------------------------
# 1. Pomocná funkcia na určenie sezóny (NHL začína v októbri)
# ---------------------------------------------------------------------------
def get_nhl_season():
    teraz = datetime.now()
    return teraz.year if teraz.month >= 10 else teraz.year - 1

# ---------------------------------------------------------------------------
# 2. Tréning: Výpočet priemerných gólov z poslednej sezóny
# ---------------------------------------------------------------------------
sezona = get_nhl_season()
print(f"Sťahujem historické dáta NHL pre sezónu {sezona}...")

# Volanie priamo importovanej funkcie (bez prefixu modulu)
schedule = get_league_schedule(sezona)

# Spracovanie do DataFrame
rows = []
for game in schedule:
    if 'score' not in game or 'homeTeam' not in game or 'awayTeam' not in game:
        continue
    rows.append({
        'home': game['homeTeam']['abbrev'],
        'away': game['awayTeam']['abbrev'],
        'home_goals': game['score']['home'],
        'away_goals': game['score']['away']
    })

df = pd.DataFrame(rows)

if df.empty:
    print("Nepodarilo sa získať žiadne historické dáta. Použijem ligové priemery.")
    league_home_avg = 3.0
    league_away_avg = 2.8
    home_avg = {}
    away_avg = {}
else:
    home_avg = df.groupby('home')['home_goals'].mean().to_dict()
    away_avg = df.groupby('away')['away_goals'].mean().to_dict()
    league_home_avg = df['home_goals'].mean()
    league_away_avg = df['away_goals'].mean()
    print(f"Model natrénovaný na {len(df)} zápasoch.")

# ---------------------------------------------------------------------------
# 3. Získanie dnešných zápasov z NHL API
# ---------------------------------------------------------------------------
dnes = datetime.now().strftime('%Y-%m-%d')
url = f"https://api-web.nhle.com/v1/schedule/{dnes}"
response = requests.get(url)
response.raise_for_status()
data = response.json()

games_today = data.get('games', [])
if not games_today:
    print(f"Na {dnes} sa v NHL nehrajú žiadne zápasy.")
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    exit()

print(f"Našlo sa {len(games_today)} zápasov NHL na {dnes}.")

# ---------------------------------------------------------------------------
# 4. Generovanie predikcií
# ---------------------------------------------------------------------------
vsetky_predikcie = []
for game in games_today:
    home_team = game['homeTeam']['abbrev']
    away_team = game['awayTeam']['abbrev']

    try:
        home_xg = home_avg.get(home_team, league_home_avg)
        away_xg = away_avg.get(away_team, league_away_avg)

        max_goals = 10
        home_probs = poisson.pmf(range(max_goals + 1), home_xg)
        away_probs = poisson.pmf(range(max_goals + 1), away_xg)

        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = home_probs[i] * away_probs[j]
                if i > j:
                    p_home += prob
                elif i == j:
                    p_draw += prob
                else:
                    p_away += prob

        total = p_home + p_draw + p_away
        if total > 0:
            p_home /= total
            p_draw /= total
            p_away /= total

        predikcia = {
            "datum": dnes,
            "domaci": home_team,
            "hostia": away_team,
            "predikcia_1X2": {
                "vyhra_doma": round(float(p_home), 3),
                "remiza": round(float(p_draw), 3),
                "vyhra_hosti": round(float(p_away), 3)
            }
        }
        vsetky_predikcie.append(predikcia)
    except Exception as e:
        print(f"Chyba pri predikcii {home_team} vs {away_team}: {e}")
        continue

with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(vsetky_predikcie, f, ensure_ascii=False, indent=2)

print(f"Predikcie boli uložené do predictions.json ({len(vsetky_predikcie)} zápasov).")
