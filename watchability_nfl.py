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
  pgwe_casa  post-game win expectancy della squadra di casa: quante volte vince chi gioca cosi'. Logistica
             sulla differenza di EPA di tutte le giocate (anche special team), senza termine per il campo:
             a parita' di gioco la sede non conta. Stimata su 1.861 partite 2019-2025 (AUC 0,986).

Senza spoiler, come faceva Thuuz:
  perche   una frase su che partita e' stata, che non dice chi vince ne' il punteggio.
  da_dove  da quando conviene guardarla: l'ultimo momento da cui resta almeno l'80% della tensione
           della partita. Pareto: il minimo di gioco per il grosso del segnale. Tensione = meta' sorpresa
           (variazione di win probability avvenuta) e meta' suspense (variazione attesa, dato punteggio e
           tempo: suspense_atteso.csv, 2021-2025), come in Ely, Frankel e Kamenica; la letteratura con dati
           di audience (Simonov e altri 2022, Buraimo e altri 2020, Kaplan 2020) trova effetti per entrambe.
           Sulle partite senza tensione (30% meno tese 2019-2025) il punto d'inizio non ha senso: highlights.
  da_min   lo stesso punto in minuti di gioco trascorsi (0 = calcio d'inizio). Mai oltre il 55':
           un punto d'inizio piu' tardi rivelerebbe un finale in volata o i supplementari.

Uso:  python watchability_nfl.py            (stagione in corso)
      python watchability_nfl.py 2025 2026  (stagioni scelte)
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"
COLS = ["game_id", "season", "week", "season_type", "home_team", "away_team", "qtr", "desc",
        "game_seconds_remaining", "vegas_home_wp", "home_score", "away_score", "posteam", "play_type", "epa"]
OUT = "watchability_nfl.csv"
PGWE_B = 0.4406         # logit della vittoria per punto di differenza di EPA
SUSPENSE = pd.read_csv(Path(__file__).with_name("suspense_atteso.csv"), index_col=0).to_numpy()

# Soglie dalle 1.936 partite di stagione regolare 2019-2025 (percentili)
LNQV_P30, LNQV_P50, LNQV_P70, LNQV_P90 = -2.289, -1.722, -1.212, -0.63
PUNTI_P15, PUNTI_P85 = 32, 60
EPA_TOT_P15, EPA_TOT_P85, EPA_MIN_P85 = -18.5, 18.8, 4.3
QUOTA = 0.80            # quota della tensione da non perdere
DALL_INIZIO = 10 * 60   # se il punto cade nei primi 10 minuti di gioco, si guarda dall'inizio
ULTIMO = 55 * 60        # mai dopo il 55' di gioco: non deve rivelare un finale o i supplementari


def stagione_in_corso(oggi=None):
    oggi = oggi or dt.date.today()
    return oggi.year if oggi.month >= 3 else oggi.year - 1


def tempo_trascorso(t, q):
    """Secondi di gioco trascorsi, monotoni; i supplementari proseguono oltre i 3600."""
    el = np.where(q <= 4, 3600 - t, 3600 + np.maximum(0, 600 - t))
    return np.maximum.accumulate(np.nan_to_num(el, nan=0.0))


def cambi_inerzia(percorso, basso=0.35, alto=0.65):
    """Quante volte la favorita cambia davvero: si conta solo chi attraversa tutta la fascia 35-65%."""
    stato, n = 0, 0
    for p in percorso:
        nuovo = 1 if p >= alto else (-1 if p <= basso else stato)
        if stato and nuovo != stato: n += 1
        stato = nuovo
    return n


def racconto(lnqv, tardi, primo_tempo, cambi, punti, epa_tot, epa_min):
    """Perche' vederla, senza spoiler: livello di tensione, dove sta, che stile di partita."""
    if lnqv >= LNQV_P90: liv = "Fra le partite piu' tese dell'anno"
    elif lnqv >= LNQV_P70: liv = "Partita tesa"
    elif lnqv >= LNQV_P50: liv = "Tensione a tratti"
    elif lnqv >= LNQV_P30: liv = "Poca suspense"
    else: liv = "Esito mai davvero in bilico"
    forma = ""
    if lnqv >= LNQV_P50:
        if cambi >= 3: forma = "l'inerzia cambia di mano piu' volte"
        elif tardi >= 0.5: forma = "il meglio e' tutto nel finale"
        elif primo_tempo >= 0.6: forma = "il meglio sta nel primo tempo"
        else: forma = "momenti caldi lungo tutta la partita"
    stile = ""
    if punti >= PUNTI_P85 and epa_min >= EPA_MIN_P85: stile = "attacchi scatenati da entrambe le parti"
    elif punti >= PUNTI_P85: stile = "tanti punti"
    elif punti <= PUNTI_P15 and lnqv >= LNQV_P50: stile = "battaglia difensiva"
    elif epa_tot <= EPA_TOT_P15: stile = "tanti errori, poca qualita'"
    elif epa_tot >= EPA_TOT_P85: stile = "giocata bene da entrambe"
    pezzi = [x for x in (forma, stile) if x]
    frase = liv + (": " + ", ".join(pezzi) if pezzi else "") + "."
    if lnqv < LNQV_P30: frase += " Bastano gli highlights."
    elif lnqv < LNQV_P50: frase += " Meglio la versione ridotta."
    return frase[0].upper() + frase[1:]


def suspense(wp, el):
    """Variazione quadratica attesa della win probability per giocata, dato punteggio (wp) e tempo."""
    w = np.clip((np.nan_to_num(wp, nan=0.5) * 20).astype(int), 0, 19)
    t = np.clip((el // 120).astype(int), 0, SUSPENSE.shape[1] - 1)
    return SUSPENSE[w, t]


def da_dove(lnqv, el, d2, sus):
    """Ultimo istante da cui resta almeno QUOTA della tensione (meta' sorpresa, meta' suspense)."""
    if lnqv < LNQV_P30: return "Solo highlights", ""
    tens = 0.5 * d2 / max(d2.sum(), 1e-9) + 0.5 * sus / max(sus.sum(), 1e-9)
    resto = np.cumsum(tens[::-1])[::-1]
    ok = np.where(resto >= QUOTA * tens.sum())[0]
    s = float(el[ok.max()]) if len(ok) else 0.0
    s = min(ULTIMO, 60 * np.floor(s / 60))
    if s < DALL_INIZIO: return "Dall'inizio", 0
    q = int(s // 900) + 1
    rest = 900 - (s - (q - 1) * 900)
    quando = f"Dall'inizio del {q}° quarto" if rest >= 900 else f"Dal {q}° quarto, {int(rest // 60)}:{int(rest % 60):02d} da giocare"
    return f"{quando} (minuto {int(s // 60)})", int(s // 60)


def misure(g):
    wp = g["vegas_home_wp"].astype(float).to_numpy()
    t = g["game_seconds_remaining"].astype(float).to_numpy()
    qt = g["qtr"].astype(float).to_numpy()
    ok = ~np.isnan(wp)
    wp, t, qt = wp[ok], t[ok], qt[ok]
    hs, as_ = g["home_score"].iloc[0], g["away_score"].iloc[0]
    fine = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
    percorso = np.concatenate([wp, [fine]])
    d = np.diff(percorso)
    tard = np.concatenate([t, [0.0]])[1:] <= 900
    qv = float((d ** 2).sum())
    gioc = g[g["play_type"].isin(["pass", "run"]) & g["epa"].notna()]
    epa = gioc.groupby("posteam")["epa"].sum()
    el = tempo_trascorso(np.concatenate([t, [0.0]])[1:], np.concatenate([qt, [qt[-1] if len(qt) else 4]])[1:])
    d2 = d ** 2
    lnqv = float(np.log(max(qv, 1e-6)))
    tardi = float(d2[el >= 3600 - 480].sum() / qv) if qv > 0 else 0.0
    primo = float(d2[el < 1800].sum() / qv) if qv > 0 else 0.0
    eh, ea = float(epa.get(g["home_team"].iloc[0], 0.0)), float(epa.get(g["away_team"].iloc[0], 0.0))
    dd, dmin = da_dove(lnqv, el, d2, suspense(wp, el))
    tutte = g[g["epa"].notna() & g["posteam"].notna() & (g["play_type"] != "no_play")].groupby("posteam")["epa"].sum()
    diff = float(tutte.get(g["home_team"].iloc[0], 0.0) - tutte.get(g["away_team"].iloc[0], 0.0))
    return pd.Series({
        "home_score": int(hs), "away_score": int(as_), "ot": int(g["qtr"].max() > 4),
        "qv": round(qv, 5), "ln_qv": round(float(np.log(max(qv, 1e-6))), 4),
        "ei": round(float(np.abs(d).sum()), 4), "ei_q4": round(float(np.abs(d[tard]).sum()), 4),
        "epa_home": round(float(epa.get(g["home_team"].iloc[0], np.nan)), 2),
        "epa_away": round(float(epa.get(g["away_team"].iloc[0], np.nan)), 2),
        "giocate": int(len(wp)),
        "quota_finale": round(tardi, 3), "cambi": cambi_inerzia(percorso),
        "perche": racconto(lnqv, tardi, primo, cambi_inerzia(percorso), hs + as_, eh + ea, min(eh, ea)),
        "da_dove": dd, "da_min": dmin,
        "pgwe_casa": round(1 / (1 + np.exp(-PGWE_B * diff)), 3),
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
