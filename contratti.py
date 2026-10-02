# Converte i contratti OverTheCap pubblicati da nflverse (parquet) in un CSV leggibile da Google Apps Script.
# Solo contratti attivi, una riga per giocatore. Eseguito ogni giorno da GitHub Actions.
import datetime as dt, sys
import pandas as pd

URL = 'https://github.com/nflverse/nflverse-data/releases/download/contracts/historical_contracts.parquet'
src = sys.argv[1] if len(sys.argv) > 1 else URL
oggi = dt.date.today()
stagione = oggi.year if oggi.month >= 3 else oggi.year - 1

d = pd.read_parquet(src)
d = d[d['is_active'] == True].copy()
d = d.sort_values(['year_signed', 'apy'], ascending=False).drop_duplicates('otc_id')

def anno(storia, campo):
    if storia is None:
        return None
    for s in storia:
        if str(s.get('year')) == str(stagione):
            return s.get(campo)
    return None

d['cap_hit_stagione'] = [anno(h, 'cap_number') for h in d['season_history']]

# Garantito residuo = dead money se tagliato ora (prima del 1° giugno), come nella tabella "Dead Money & Cap Savings" di OverTheCap:
# bonus ripartiti non ancora contabilizzati (stagione corrente e future) + stipendio garantito della stagione corrente e future.
def dead_money(storia):
    if storia is None:
        return None
    tot = 0.0
    for s in storia:
        y = str(s.get('year'))
        if y.isdigit() and int(y) >= stagione:
            tot += (s.get('prorated_bonus') or 0) + (s.get('guaranteed_salary') or 0)
    return round(tot, 6)

d['dead_money_taglio'] = [dead_money(h) for h in d['season_history']]
d['risparmio_taglio'] = d['cap_hit_stagione'] - d['dead_money_taglio']
d['cap_pct_stagione'] = [anno(h, 'cap_percent') for h in d['season_history']]
d['fine_contratto'] = d['year_signed'] + d['years'] - 1
cols = ['gsis_id', 'otc_id', 'player', 'position', 'team', 'year_signed', 'years', 'fine_contratto',
        'value', 'apy', 'guaranteed', 'apy_cap_pct', 'cap_hit_stagione', 'cap_pct_stagione', 'dead_money_taglio', 'risparmio_taglio']
d[cols].to_csv('contratti.csv', index=False)
print(len(d), 'contratti attivi, stagione', stagione)
