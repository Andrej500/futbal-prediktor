import json
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Pomocné funkcie
# ---------------------------------------------------------------------------
def get_nhl_season():
    teraz = datetime.now(timezone.utc)
    return teraz.year if teraz.month >= 10 else teraz.year - 1

def fetch_schedule(date_str):
    url = f"https://api-web.nhle.com/v1/schedule/{date_str}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return resp.json()

def extract_games(data):
    games = []
    if 'gameWeek' in data:
        for day in data['gameWeek']:
            games.extend(day.get('games', []))
    elif 'games' in data:
        games = data['games']
    return games

def extract_score(game):
    home_score = game.get('homeTeam', {}).get('score')
    away_score = game.get('awayTeam', {}).get('score')
    if home_score is None and 'score' in game:
        home_score = game['score'].get('home')
        away_score = game['score'].get('away')
    return home_score, away_score

def parse_start_time(game):
    start_str = game.get('startTimeUTC') or game.get('start_time_utc')
    if not start_str:
        return None
    try:
        return datetime.fromisoformat(start_str.replace('Z', '+00:00'))
    except ValueError:
        return None

def to_local(dt_utc):
    try:
        from zoneinfo import ZoneInfo
        return dt_utc.astimezone(ZoneInfo("Europe/Bratislava"))
    except Exception:
        return dt_utc + timedelta(hours=1)

# ---------------------------------------------------------------------------
# 1. Tréning: Stiahnutie histórie a výpočet Dixon-Coles parametrov
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
            if not str(gid).startswith(season_prefix) or game.get('gameType') not in (2, 3):
                continue
            if game.get('gameState') not in ('OFF', 'FINAL'):
                continue
            home_score, away_score = extract_score(game)
            home_abbrev = game.get('homeTeam', {}).get('abbrev')
            away_abbrev = game.get('awayTeam', {}).get('abbrev')
            if home_score is not None and away_score is not None and home_abbrev and away_abbrev:
                rows.append({'home': home_abbrev, 'away': away_abbrev, 'home_goals': home_score, 'away_goals': away_score})
    except Exception as e:
        print(f"Chyba pri {date_str}: {e}")
    current += timedelta(days=7)

df = pd.DataFrame(rows)
print(f"Získaných {len(df)} odohratých zápasov.")

if df.empty:
    print("Žiadne historické dáta – použijem ligové priemery.")
    league_avg = 3.0
    home_adv = 1.1
    attack_home, defense_home, attack_away, defense_away = {}, {}, {}, {}
else:
    # Dixon-Coles výpočty
    league_avg = (df['home_goals'].mean() + df['away_goals'].mean()) / 2
    home_adv = df['home_goals'].mean() / league_avg

    attack_home = (df.groupby('home')['home_goals'].mean() / league_avg).to_dict()
    defense_home = (df.groupby('home')['away_goals'].mean() / league_avg).to_dict()
    attack_away = (df.groupby('away')['away_goals'].mean() / league_avg).to_dict()
    defense_away = (df.groupby('away')['home_goals'].mean() / league_avg).to_dict()

    print(f"Ligový priemer: {league_avg:.2f}, Výhoda domácich: {home_adv:.2f}")

# ---------------------------------------------------------------------------
# 2. Dnešné zápasy (36 hodinové okno)
# ---------------------------------------------------------------------------
teraz_utc = datetime.now(timezone.utc)
dnes_utc = teraz_utc.strftime('%Y-%m-%d')
zajtra_utc = (teraz_utc + timedelta(days=1)).strftime('%Y-%m-%d')
pozajtra_utc = (teraz_utc + timedelta(days=2)).strftime('%Y-%m-%d')

print(f"Aktuálny čas UTC: {teraz_utc.strftime('%Y-%m-%d %H:%M')}")

all_recent = []
for d in (dnes_utc, zajtra_utc, pozajtra_utc):
    try:
        data = fetch_schedule(d)
        all_recent.extend(extract_games(data))
    except Exception as e:
        pass

seen_ids_today = set()
unique_games = []
for game in all_recent:
    gid = game.get('id')
    if gid not in seen_ids_today:
        seen_ids_today.add(gid)
        unique_games.append(game)

okno_koniec = teraz_utc + timedelta(hours=36)
today_games = []
for game in unique_games:
    start = parse_start_time(game)
    if start and teraz_utc <= start <= okno_koniec:
        today_games.append(game)

today_games.sort(key=lambda g: parse_start_time(g) or teraz_utc)

if not today_games:
    print("V nasledujúcich 36 hodinách sa nehrajú žiadne zápasy NHL.")
    with open("predictions.json", "w", encoding="utf-8") as f:
        json.dump([], f)
    exit()

print(f"Našlo sa {len(today_games)} zápasov v nasledujúcich 36 hodinách.")

# ---------------------------------------------------------------------------
# 3. Predikcie (BEZ FILTRA – ukladajú sa všetky zápasy)
# ---------------------------------------------------------------------------
predictions = []

for game in today_games:
    home_team = game['homeTeam']['abbrev']
    away_team = game['awayTeam']['abbrev']

    # Dixon-Coles očakávané góly (xG)
    a_h = attack_home.get(home_team, 1.0)
    d_h = defense_home.get(home_team, 1.0)
    a_a = attack_away.get(away_team, 1.0)
    d_a = defense_away.get(away_team, 1.0)
    
    home_xg = a_h * d_a * home_adv * league_avg
    away_xg = a_a * d_h * league_avg

    max_goals = 10
    home_probs = poisson.pmf(range(max_goals + 1), home_xg)
    away_probs = poisson.pmf(range(max_goals + 1), away_xg)

    p_home, p_draw, p_away = 0.0, 0.0, 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            prob = home_probs[i] * away_probs[j]
            if i > j: p_home += prob
            elif i == j: p_draw += prob
            else: p_away += prob

    total = p_home + p_draw + p_away
    if total > 0:
        p_home /= total; p_draw /= total; p_away /= total

    # --- ÚROVEŇ ISTOTY (podľa najvyššej pravdepodobnosti) ---
    max_prob = max(p_home, p_draw, p_away)
    if max_prob >= 0.55:
        uroven = "VYSOKA"
    elif max_prob >= 0.50:
        uroven = "STREDNA"
    else:
        uroven = "NIZKA"

    start_utc = parse_start_time(game)
    start_sk = to_local(start_utc) if start_utc else None

    predictions.append({
        "zaciatok_sk": start_sk.strftime('%Y-%m-%d %H:%M') if start_sk else None,
        "domaci": home_team,
        "hostia": away_team,
        "uroven_istoty": uroven,
        "predikcia_1X2": {
            "vyhra_doma": round(float(p_home), 3),
            "remiza": round(float(p_draw), 3),
            "vyhra_hosti": round(float(p_away), 3)
        }
    })

with open("predictions.json", "w", encoding="utf-8") as f:
    json.dump(predictions, f, ensure_ascii=False, indent=2)

print(f"Uložených {len(predictions)} predikcií (bez filtra).")
for p in predictions:
    print(f"  [{p['uroven_istoty']}] {p['zaciatok_sk']} SK | {p['domaci']} vs {p['hostia']} | "
          f"{p['predikcia_1X2']['vyhra_doma']:.0%} / "
          f"{p['predikcia_1X2']['remiza']:.0%} / "
          f"{p['predikcia_1X2']['vyhra_hosti']:.0%}")
