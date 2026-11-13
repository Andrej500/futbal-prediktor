import os
import json
import requests
import pandas as pd
from datetime import datetime
from poisson_odds import Poisson
import edginghockeyscraper

# ---------------------------------------------------------------------------
# 1. Pomocná funkcia na určenie sezóny (NHL začína v októbri)
# ---------------------------------------------------------------------------
def get_nhl_season():
    teraz = datetime.now()
    # Ak je mesiac >= 10 (október), sezóna je aktuálny rok, inak predchádzajúci
    return teraz.year if teraz.month >= 10 else teraz.year - 1

# ---------------------------------------------------------------------------
# 2. Tréning: Výpočet priemerných gólov z poslednej sezóny
# ---------------------------------------------------------------------------
sezona = get_nhl_season()
print(f"Sťahujem historické dáta NHL pre sezónu {sezona}...")

# edginghockeyscraper vracia zoznam zápasov so skóre
schedule = edginghockeyscraper.get_league_schedule(sezona)

# Spracovanie do DataFrame
rows = []
for game in schedule:
    # Preskočíme zápasy bez výsledku (napr. odložené)
    if 'score' not in game or 'homeTeam' not in game or 'awayTeam' not in game:
        continue
    rows.append({
        'home': game['homeTeam']['abbrev'],
        'away': game['awayTeam']['abbrev'],
        'home_goals': game['score']['home'],
        'away_goals': game['score']['away']
    })

df = pd.DataFrame(rows)

# Výpočet priemerov pre každý tím (doma aj vonku)
home_avg = df.groupby('home')['home_goals'].mean().to_dict()
away_avg = df.groupby('away')['away_goals'].mean().to_dict()

# Ligaový priemer (pre tím, ktorý nemá dosť dát)
league_home_avg = df['home_goals'].mean()
league_away_avg = df['away_goals'].mean()

print(f"Model natrénovaný na {len(df)} zápasoch.")

# ---------------------------------------------------------------------------
# 3. Získanie dnešných zápasov z NHL API
# ---------------------------------------------------------------------------
# Použijeme priamo NHL API (netreba kľúč)
dnes = datetime.now().strftime('%Y-%m-%d')
url = f"https://api-web.nhle.com/v1/schedule/{dnes}"
response = requests.get(url)
response.raise_for_status()
data = response.json()

# Získanie zoznamu zápasov na dnes
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
        # Očakávané góly: priemer domácich v útoku vs. priemer hostí v obrane (zjednodušene)
        # Použijeme priemery z minulej sezóny
        home_xg = home_avg.get(home_team, league_home_avg)
        away_xg = away_avg.get(away_team, league_away_avg)

        # Poisson model (očakávané góly domácich, hostí)
        model = Poisson(home_xg, away_xg)

        # Získanie pravdepodobností pre 1X2 (domáci, remíza, hostia)
        # moneyline vracia fair odds, my chceme pravdepodobnosti
        # Môžeme použiť vstavanú metódu alebo si ich vypočítať
        # poisson-odds má atribút moneyline, ktorý vracia fair odds
        # Pravdepodobnosť = 1 / fair_odds
        fair_odds = model.moneyline  # Formát: "1-X-2: 5.291-4.878-1.669" (string)
        # Parsovanie je krehké, preto si radšej vypočítame pravdepodobnosti manuálne
        # Alebo použijeme model.print_probability_table_goal_draws() a parsujeme?
        # Jednoduchšie: použijeme vstavaný výpočet z poisson-odds ak existuje.
        # Predpokladajme, že model má metódu na získanie pravdepodobností.
        # Pre istotu použijeme scipy na výpočet 1X2 pravdepodobností.
        # Ale to by vyžadovalo scipy. Namiesto toho môžeme použiť fair odds.
        # Pre demonštráciu použijeme zjednodušený výpočet:
        # P(domáci) = home_xg / (home_xg + away_xg) atď.? To nie je presné.
        # Radšej použijeme metódu z poisson-odds, ak je dostupná.
        # Skúsime: model.probabilities() alebo podobne.
        # Keďže nevieme API, použijeme len fair odds a prevedieme na pravdepodobnosti.
        # Ale formát je string, to je nepraktické.
        # Namiesto toho použijeme vlastný výpočet pomocou scipy.stats.poisson.
        from scipy.stats import poisson
        import numpy as np

        max_goals = 10
        home_probs = poisson.pmf(range(max_goals + 1), home_xg)
        away_probs = poisson.pmf(range(max_goals + 1), away_xg)

        # Pravdepodobnosť výhry domácich = súčet P(home > away)
        # Pravdepodobnosť remízy = súčet P(home == away)
        # Pravdepodobnosť výhry hostí = súčet P(away > home)
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

        # Normalizácia (súčet by mal byť ~1, ale pre istotu)
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

# ---------------------------------------------------------------------------
# 5. Uloženie výsledkov
# ---------------------------------------------------------------------------
with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(vsetky_predikcie, f, ensure_ascii=False, indent=2)

print(f"Predikcie boli uložené do predictions.json ({len(vsetky_predikcie)} zápasov).")
