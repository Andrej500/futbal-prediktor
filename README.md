# Futbalový prediktor

Automatický systém, ktorý každé ráno stiahne zápasy na daný deň a vygeneruje predikcie.

## Ako to funguje
- GitHub Actions spúšťa skript `predictor.py` každý deň o 7:00 UTC.
- Skript získa zápasy z API-Football a uloží predikcie do `predictions.json`.

## Nastavenie
1. Vytvorte si API kľúč na [api-football.com](https://www.api-football.com/).
2. V GitHub repozitári choďte do Settings → Secrets and variables → Actions a pridajte secret `API_FOOTBALL_KEY`.
