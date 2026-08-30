import os
import json
import math
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

ENDPOINT_SENSORS_ANAGRAFICA = "https://www.dati.lombardia.it/resource/nf78-nj6b.json"
ENDPOINT_METEO_DATA = "https://www.dati.lombardia.it/resource/647i-nhxk.json"
ID_STAZIONE = "1545"

def get_sensor_ids_for_station(id_stazione: str) -> Dict[str, str]:
    params = {"$where": f"idstazione = '{id_stazione}'", "$limit": 50}
    response = requests.get(ENDPOINT_SENSORS_ANAGRAFICA, params=params, timeout=20)
    response.raise_for_status()
    mappa = {}
    for s in response.json():
        tipo = s.get("tipologia", "").lower()
        ids = s.get("idsensore")
        if "precipitazione" in tipo: mappa["pioggia"] = ids
        elif "temperatura" in tipo: mappa["temperatura"] = ids
        elif "umidit" in tipo: mappa["umidita"] = ids
        elif "vento" in tipo or "velocit" in tipo: mappa["vento"] = ids
    return mappa

def download_weather_history(mappa_sensori: Dict[str, str], days: int = 45) -> pd.DataFrame:
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
    id_list = "','".join(mappa_sensori.values())
    params = {
        "$where": f"idsensore in ('{id_list}') AND data >= '{start_date}' AND valore != -9999",
        "$limit": 50000, "$order": "data ASC"
    }
    response = requests.get(ENDPOINT_METEO_DATA, params=params, timeout=30)
    response.raise_for_status()
    records = response.json()
    if not records: return pd.DataFrame()
    df = pd.DataFrame(records)
    df["data"] = pd.to_datetime(df["data"])
    df["valore"] = pd.to_numeric(df["valore"], errors="coerce")
    inv_map = {v: k for k, v in mappa_sensori.items()}
    df["tipo"] = df["idsensore"].map(inv_map)
    return df.pivot_table(index="data", columns="tipo", values="valore", aggfunc="mean").reset_index()

def aggregate_daily(df_hourly: pd.DataFrame) -> List[Dict[str, Any]]:
    if df_hourly.empty: return []
    
    df_hourly["giorno"] = df_hourly["data"].dt.strftime("%Y-%m-%d")
    oggi_str = datetime.now().strftime("%Y-%m-%d")
    df_hourly = df_hourly[df_hourly["giorno"] < oggi_str]
    
    if df_hourly.empty: return []

    agg_rules = {}
    if "pioggia" in df_hourly.columns: agg_rules["pioggia"] = "sum"
    if "umidita" in df_hourly.columns: agg_rules["umidita"] = "mean"
    if "vento" in df_hourly.columns: agg_rules["vento"] = "max"
    if "temperatura" in df_hourly.columns: agg_rules["temperatura"] = ["mean", "max", "min"]
    
    df_daily = df_hourly.groupby("giorno").agg(agg_rules)
    df_daily.columns = ['_'.join(col).strip() for col in df_daily.columns.values]
    df_daily = df_daily.reset_index()

    serie = []
    for _, row in df_daily.iterrows():
        vento_kmh = row.get("vento_max", 0.0) * 3.6 if pd.notna(row.get("vento_max")) else 10.0
        serie.append({
            "data": row["giorno"],
            "pioggia_mm": round(float(row.get("pioggia_sum", 0.0)), 1),
            "t_media": round(float(row.get("temperatura_mean", 15.0)), 1),
            "t_max": round(float(row.get("temperatura_max", 15.0)), 1),
            "t_min": round(float(row.get("temperatura_min", 15.0)), 1),
            "rh_media": round(float(row.get("umidita_mean", 70.0)), 1),
            "vento_max": round(float(vento_kmh), 1)
        })
    return serie

class AnalizzatoreSiccitaPorcini:
    def __init__(self, soglia_evento: float = 35.0):
        self.soglia_evento = soglia_evento

    def analizza(self, serie: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not serie:
            return {"evento_rilevato": False, "stato": "In attesa di dati ARPA"}

        n = len(serie)
        giorno_ieri = serie[-1]
        
        eventi_trovati = []
        i = 2
        while i < n:
            c3 = sum(serie[k]["pioggia_mm"] for k in range(i - 2, i + 1))
            if c3 >= self.soglia_evento:
                giorni_finestra = [serie[k] for k in range(i - 2, i + 1)]
                giorno_max = max(giorni_finestra, key=lambda x: x["pioggia_mm"])
                idx = serie.index(giorno_max)
                
                if not eventi_trovati or (idx - eventi_trovati[-1]["indice"]) >= 4:
                    p_pre_30 = sum(serie[k]["pioggia_mm"] for k in range(max(0, idx - 32), max(0, idx - 2)))
                    
                    # Calcolo biologico del potenziale post-siccità
                    if p_pre_30 < 15.0: ritardo, soglia, smorz = 12, 60.0, 0.30 
                    elif 15.0 <= p_pre_30 < 30.0: ritardo, soglia, smorz = 7, 50.0, 0.70
                    elif 30.0 <= p_pre_30 < 60.0: ritardo, soglia, smorz = 3, 40.0, 0.90
                    else: ritardo, soglia, smorz = 0, 35.0, 1.00

                    giorni_da_ev = (n - 1) - idx
                    
                    giorni_favonio = sum(1 for k in range(idx + 1, n) 
                                         if serie[k]["vento_max"] > 22.0 
                                         and serie[k]["rh_media"] < 60.0 
                                         and serie[k]["pioggia_mm"] < 1.0)
                    danno_favonio = max(0.1, 1.0 - (giorni_favonio * 0.25))
                    
                    eventi_trovati.append({
                        "indice": idx,
                        "data": giorno_max["data"],
                        "pioggia": c3,
                        "giorni_da_evento": giorni_da_ev,
                        "ritardo": ritardo,
                        "soglia": soglia,
                        "smorzamento": smorz * danno_favonio
                    })
                i += 3 
            else:
                i += 1

        eventi_trovati = [ev for ev in eventi_trovati if ev["giorni_da_evento"] <= 40]

        # Sensore Notti Tropicali su tutto l'archivio (fino a 90 gg)
        notti_tropicali = sum(1 for d in serie if d.get("t_min", 0) >= 19.0)
        rischio_senescenza = notti_tropicali >= 2

        diag = {
            "t_max_attuale": giorno_ieri["t_max"],
            "t_min_attuale": giorno_ieri["t_min"],
            "rh_media_attuale": giorno_ieri["rh_media"],
            "vento_max_attuale": giorno_ieri["vento_max"],
            "pioggia_oggi": giorno_ieri["pioggia_mm"], 
            "eventi": eventi_trovati,
            "rischio_senescenza": rischio_senescenza
        }
        
        if not eventi_trovati:
            diag.update({"evento_rilevato": False, "stato": "Nessuna pioggia rilevante"})
        else:
            ev_recente = eventi_trovati[-1]
            diag.update({
                "evento_rilevato": True,
                "data_evento": ev_recente["data"],
                "giorni_da_evento": ev_recente["giorni_da_evento"],
                "ritardo_siccita_applicato": ev_recente["ritardo"]
            })
        return diag

def calcola_microzone(diag: Dict[str, Any], quota_stazione: int = 1285) -> List[Dict[str, Any]]:
    zone_cfg = [
        {"nome": "Camnasco", "quota": 750, "essenza": "castagno", "esposizione": "SE", "giorni_base": 6},
        {"nome": "Betulle SE", "quota": 1222, "essenza": "betulla", "esposizione": "SE", "giorni_base": 9},
        {"nome": "Betulle NE", "quota": 1144, "essenza": "betulla", "esposizione": "NE", "giorni_base": 8},
        {"nome": "Faggi Ovest", "quota": 1561, "essenza": "faggio", "esposizione": "OVEST_OMBRA", "giorni_base": 12},
        {"nome": "Abeti Nord", "quota": 1478, "essenza": "pino", "esposizione": "NORD", "giorni_base": 13}
    ]

    if not diag.get("evento_rilevato"):
        return [{"zona": z["nome"], "indice_buttata": 0.0, "stato": "In attesa", "giorni_mancanti_al_picco": None, "onde": []} for z in zone_cfg]

    res = []
    for z in zone_cfg:
        gradiente = 0.0065 * (z["quota"] - quota_stazione)
        t_min_b = diag["t_min_attuale"] - gradiente
        t_max_b = diag["t_max_attuale"] - gradiente
        
        rh_eff = diag["rh_media_attuale"]
        if z["esposizione"] == "SE":
            t_max_eff, t_min_eff, rh_eff = t_max_b + 2.5, t_min_b, max(0.0, diag["rh_media_attuale"] - 10.0)
        elif z["esposizione"] == "NE":
            t_max_eff, t_min_eff, rh_eff = t_max_b - 1.0, t_min_b, min(100.0, diag["rh_media_attuale"] + 12.0)
        elif z["esposizione"] == "OVEST_OMBRA":
            t_max_eff, t_min_eff, rh_eff = t_max_b - 1.5, t_min_b - 0.5, min(100.0, diag["rh_media_attuale"] + 15.0)
        elif z["esposizione"] == "NORD":
            t_max_eff, t_min_eff, rh_eff = t_max_b - 2.5, t_min_b - 1.5, min(100.0, diag["rh_media_attuale"] + 20.0)
        else:
            t_max_eff, t_min_eff = t_max_b, t_min_b
            
        # Correzioni speciali ecosistemi
        if z["nome"] == "Camnasco": rh_eff = max(60.0, rh_eff) 
        if z["nome"] == "Faggi Ovest": t_min_eff += 1.0 

        t_opt = 16.5 if z["essenza"] not in ["pino", "faggio"] else 14.5
        t_media_eff = (t_max_eff + t_min_eff) / 2
        f_T_media = math.exp(- ((t_media_eff - t_opt) ** 2) / (2 * (3.5 ** 2)))
        f_T_freddo = 0.0 if t_min_eff < 3.0 else ((t_min_eff - 3.0) / 4.0 if t_min_eff < 7.0 else 1.0)
        
        # Grilletto termico
        if 8.0 <= t_min_eff <= 13.0: f_grilletto = 1.3
        elif t_min_eff > 17.0: f_grilletto = 0.7
        else: f_grilletto = 1.0

        f_H = 1.0 if rh_eff >= 85 else (0.0 if rh_eff < 40 else ((rh_eff - 40) / 45) ** 1.2)
        
        vento = diag["vento_max_attuale"]
        is_favonio = (vento > 20 and diag["rh_media_attuale"] < 60 and diag["pioggia_oggi"] < 1.0)
        
        # Sant'Amate è protetta dal catino roccioso
        if z["nome"] == "Faggi Ovest": phi_vento = 1.0
        else: phi_vento = max(0.1, 1.0 - 0.04 * (vento - 20)) if is_favonio else 1.0
        
        indice_totale = 0.0
        onde = []
        
        for ev in diag["eventi"]:
            f_R = 1.0 / (1.0 + math.exp(-0.12 * (ev["pioggia"] - ev["soglia"])))
            
            ritardo_applicato = ev["ritardo"]
            if z["nome"] == "Camnasco": ritardo_applicato = min(1, ritardo_applicato) 
                
            picco_eff = z["giorni_base"] + ritardo_applicato
            f_L = math.exp(- ((ev["giorni_da_evento"] - picco_eff) ** 2) / (2 * (2.2 ** 2)))
            
            ind_pieno = 100.0 * (f_R * (f_T_media * f_T_freddo) * 1.0 * f_H) * phi_vento * ev["smorzamento"] * f_grilletto
            ind_oggi = ind_pieno * f_L
            
            indice_totale += ind_oggi
            onde.append({
                "giorni_mancanti_da_ieri": round(picco_eff - ev["giorni_da_evento"], 1),
                "indice_picco": round(ind_pieno, 1)
            })

        indice_totale = min(100.0, indice_totale)
        
        futuri = [onda["giorni_mancanti_da_ieri"] for onda in onde if onda["giorni_mancanti_da_ieri"] > 0]
        giorni_mancanti = min(futuri) if futuri else (min([o["giorni_mancanti_da_ieri"] for o in onde]) if onde else 0)
        
        if f_T_freddo == 0.0: stato = "Blocco freddo"
        elif indice_totale > 65: stato = "Buttata in corso (Onde multiple)" if len(onde)>1 else "Buttata in corso"
        elif giorni_mancanti > 0: stato = f"Incubazione ({giorni_mancanti} gg)"
        else: stato = "In esaurimento"

        res.append({
            "zona": z["nome"],
            "indice_buttata": round(indice_totale, 1),
            "t_min_stimata": round(t_min_eff, 1),
            "t_max_stimata": round(t_max_eff, 1),
            "giorni_mancanti_al_picco": giorni_mancanti,
            "stato": stato,
            "onde": onde
        })
    return res

def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Connessione ARPA...")
    sensori = get_sensor_ids_for_station(ID_STAZIONE)
    df_orari = download_weather_history(sensori, days=45)
    nuovi_dati = aggregate_daily(df_orari)
    
    if not nuovi_dati:
        print("Errore: nessun dato scaricato da ARPA.")
        return

    # Carichiamo il vecchio storico (se esiste) per allungare la memoria dell'app
    storico_esistente = []
    try:
        if os.path.exists("data/storico.json"):
            with open("data/storico.json", "r") as f:
                storico_esistente = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fusione dei dati: sovrascrive i giorni duplicati aggiornandoli e tiene i vecchi
    storico_unito = {d["data"]: d for d in storico_esistente if "data" in d}
    for d in nuovi_dati:
        storico_unito[d["data"]] = d
        
    # Ordiniamo cronologicamente e teniamo gli ultimi 90 giorni
    storico_ordinato = sorted(storico_unito.values(), key=lambda x: x["data"])
    storico_finale = storico_ordinato[-90:]

    analizzatore = AnalizzatoreSiccitaPorcini()
    diagnosi = analizzatore.analizza(storico_finale)
    previsioni = calcola_microzone(diagnosi)

    output = {
        "ultimo_aggiornamento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stazione": {"id": ID_STAZIONE, "nome": "San Siro", "quota_m": 1285},
        "diagnosi_meteo": diagnosi,
        "zone": previsioni,
        "storico_completo": storico_finale
    }

    os.makedirs("data", exist_ok=True)
    with open("data/previsioni.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        
    with open("data/storico.json", "w", encoding="utf-8") as f:
        json.dump(storico_finale, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
