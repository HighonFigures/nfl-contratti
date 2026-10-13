"""Watchability NFL: il DRAMMA di ogni partita conclusa, dal play-by-play di nflverse.

Per ogni partita legge il percorso della win probability con la linea di mercato (vegas_home_wp)
e scrive una riga in watchability_nfl.csv. Il foglio Google Watchability_NFL la legge con la voce
"4. Completa il post-partita" del menu: Apps Script non riesce a scaricare il play-by-play intero
(circa 6 MB compressi a meta' stagione, 20 a fine stagione), questo file pesa pochi KB.

Misure, sullo stesso percorso win probability chiuso sull'esito finale (1, 0 o 0,5):
  qv     somma dei quadrati delle variazioni di win probability (variazione quadratica). E' la
         grandezza che il Gambletron 2000 chiama hotness, calcolata sugli scambi di Betfair: sulle
         114 partite 2025-2026 confrontabili il logaritmo dei due correla a 0,93.
  ln_qv  logaritmo naturale di qv: e' il valore che il foglio usa (ancore 2,5o e 97,5o percentile
         2019-2025: -4,11 e -0,12).
  ei     somma dei valori assoluti delle variazioni (excitement index di inpredictable).
  ei_q4  la stessa somma limitata agli ultimi 15 minuti e ai supplementari.
  epa_*  EPA su passaggi e corse della squadra in possesso: riserva, se stats_team_week tarda.

Uso:  python watchability_nfl.py            (stagione in corso)
      python watchability_nfl.py 2025 2026  (stagioni scelte)
"""
import datetime as dt
import sys

import numpy as np
import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
COLS = ["game_id", "season", "week", "season_type", "home_team", "away_team", "qtr", "desc",
        "game_seconds_remaining", "vegas_home_wp", "home_score", "away_score", "posteam", "play_type", "epa"]
OUT = "watchability_nfl.csv"


def stagione_in_corso(oggi=None):
    oggi = oggi or dt.date.today()
    return oggi.year if oggi.month >= 3 else oggi.year - 1


def misure(g):
    wp = g["vegas_home_wp"].astype(float).to_numpy()
    t = g["game_seconds_remaining"].astype(float).to_numpy()
    ok = ~np.isnan(wp)
    wp, t = wp[ok], t[ok]
    hs, as_ = g["home_score"].iloc[0], g["away_score"].iloc[0]
    fine = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
    percorso = np.concatenate([wp, [fine]])
    d = np.diff(percorso)
    tard = np.concatenate([t, [0.0]])[1:] <= 900
    qv = float((d ** 2).sum())
    gioc = g[g["play_type"].isin(["pass", "run"]) & g["epa"].notna()]
    epa = gioc.groupby("posteam")["epa"].sum()
    return pd.Series({
        "home_score": int(hs), "away_score": int(as_), "ot": int(g["qtr"].max() > 4),
        "qv": round(qv, 5), "ln_qv": round(float(np.log(max(qv, 1e-6))), 4),
        "ei": round(float(np.abs(d).sum()), 4), "ei_q4": round(float(np.abs(d[tard]).sum()), 4),
        "epa_home": round(float(epa.get(g["home_team"].iloc[0], np.nan)), 2),
        "epa_away": round(float(epa.get(g["away_team"].iloc[0], np.nan)), 2),
        "giocate": int(len(wp)),
    })


def stagione(anno):
    df = pd.read_parquet(URL.format(anno), columns=COLS)
    # solo partite concluse: nflverse pubblica il play-by-play a partita finita, ma il controllo costa poco
    finite = df.groupby("game_id")["desc"].apply(lambda s: s.astype(str).str.contains("END GAME").any())
    df = df[df["game_id"].isin(finite[finite].index)]
    meta = df.groupby("game_id").agg(season=("season", "first"), week=("week", "first"),
                                     season_type=("season_type", "first"),
                                     home=("home_team", "first"), away=("away_team", "first"))
    righe = df.groupby("game_id", sort=False).apply(misure, include_groups=False)
    tab = meta.join(righe).reset_index()
    for c in ("home_score", "away_score", "ot", "giocate"):
        tab[c] = tab[c].astype(int)
    return tab


def main():
    anni = [int(a) for a in sys.argv[1:]] or [stagione_in_corso()]
    tab = pd.concat([stagione(a) for a in anni], ignore_index=True)
    tab = tab.sort_values(["season", "week", "game_id"])
    tab["aggiornato"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tab.to_csv(OUT, index=False)
    print(f"{len(tab)} partite scritte in {OUT}")


if __name__ == "__main__":
    main()
