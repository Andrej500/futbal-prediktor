import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------
def get_nhl_season():
    """NHL sezóna začína v októbri, takže od októbra je to aktuálny rok."""
    teraz = datetime.now()
    return teraz.year if teraz.month >= 10 else teraz.year - 1

def fetch_schedule(date_str):
    """Stiahne rozvrh NHL pre daný dátum (API vráti celý týždeň)."""
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

def extract_games(data):
    """Vráti zoznam zápasov z odpovede API (rôzne verzie štruktúry)."""
    games = []
    if 'gameWeek' in data:
        for day in data['gameWeek']:
            games.extend(day.get('games', []))
    elif 'games' in data:
        games = data['games']
    return games

def extract_score(game):
    """Vráti (home_score, away_score) alebo (None, None), ak nie je výsledok."""
    home_score = game.get('homeTeam', {}).get('score')
    away_score = game.get('awayTeam', {}).get('score')
    if home_score is None and 'score' in game:
        home_score = game['score'].get('home')
        away_score = game['score'].get('away')
    return home_score, away_score

# ---------------------------------------------------------------------------
# 1. Tréning: Stiahnutie histórie z priameho NHL API
# ---------------------------------------------------------------------------
sezona = get_nhl_season()
print(f"Sťahujem históriu NHL pre sezónu {sezona}...")

season_prefix = str(sezona)
rows = []
seen_ids = set()
current = datetime(sezona, 10, 1)
end = datetime(sezona + 1, 6, 30)

while current <= end:
    date_str = current.strftime('%Y-%m-%d')
    try:
        data = fetch_schedule(date_str)
        for game in extract_games(data):
            gid = game.get('id')
            if gid in seen_ids:
                continue
            seen_ids.add(gid)
            
            # Filter na aktuálnu sezónu
            if not str(gid).startswith(season_prefix):
                continue
            # Len základná časť (2) a play-off (3)
            if game.get('gameType') not in (2, 3):
                continue
            # Len odohraté zápasy
            if game.get('gameState') not in ('OFF', 'FINAL'):
                continue
            
            home_score, away_score = extract_score(game)
            if home_score is None or away_score is None:
                continue
            
            home_abbrev = game.get('homeTeam', {}).get('abbrev')
            away_abbrev = game.get('awayTeam', {}).get('abbrev')
            if not home_abbrev or not away_abbrev:
                continue
            
            rows.append({
                'home': home_abbrev,
                'away': away_abbrev,
                'home_goals': home_score,
                'away_goals': away_score
            })
    except Exception as e:
        print(f"Chyba pri {date_str}: {e}")
    current += timedelta(days=7)

df = pd.DataFrame(rows)
print(f"Získaných {len(df)} odohratých zápasov.")

if df.empty:
    print("Žiadne historické dáta – použijem ligové priemery.")
    league_home_avg = 3.0
    league_away_avg = 2.8
    home_avg = {}
    away_avg = {}
else:
    home_avg = df.groupby('home')['home_goals'].mean().to_dict()
    away_avg = df.groupby('away')['away_goals'].mean().to_dict()
    league_home_avg = df['home_goals'].mean()
    league_away_avg = df['away_goals'].mean()
    print(f"Ligové priemery – domáci: {league_home_avg:.2f}, hostia: {league_away_avg:.2f}")

# ---------------------------------------------------------------------------
# 2. Dnešné zápasy
# ---------------------------------------------------------------------------
dnes = datetime.now().strftime('%Y-%m-%d')
print(f"Získavam zápasy na {dnes}...")
today_data = fetch_schedule(dnes)
today_games_raw = extract_games(today_data)
today_games = [g for g in today_games_raw if g.get('gameDate', '').startswith(dnes)]

if not today_games:
    print(f"Na {dnes} sa nehrajú žiadne zápasy NHL.")
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    exit()

print(f"Našlo sa {len(today_games)} zápasov.")

# ---------------------------------------------------------------------------
# 3. Predikcie
# ---------------------------------------------------------------------------
predictions = []
for game in today_games:
    home_team = game['homeTeam']['abbrev']
    away_team = game['awayTeam']['abbrev']
    
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
    
    predictions.append({
        "datum": dnes,
        "domaci": home_team,
        "hostia": away_team,
        "predikcia_1X2": {
            "vyhra_doma": round(float(p_home), 3),
            "remiza": round(float(p_draw), 3),
            "vyhra_hosti": round(float(p_away), 3)
        }
    })

with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"Uložených {len(predictions)} predikcií.")
