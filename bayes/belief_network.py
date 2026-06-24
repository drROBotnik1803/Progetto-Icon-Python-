"""
bayes/belief_network.py
=======================
Ragionamento sotto Incertezza — Rete Bayesiana
"""

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Soglia oltre la quale il robot si ferma per sicurezza
SOGLIA_STOP = 0.70


P_ANOMALIA_DATO_GUASTO = 0.85      # Sensore sensibile al guasto
P_ANOMALIA_DATO_NO_GUASTO = 0.15   # Rumore o falsi positivi del sensore


class BeliefNetwork:
    """
    Rete Bayesiana per il rilevamento guasti durante la navigazione autonoma.
    Calcola il posterior di guasto in base al modello di robot e all'evidenza dei sensori.
    """

    def __init__(self):
        # Mappa robot_model -> prior di guasto (calcolata basandosi sulla media della severity in vulnerability_data.csv)
        self.priors_robot: Dict[str, float] = {}

    def load_from_csv(self, path: Optional[Path] = None) -> None:
        """Carica i dati di vulnerabilità e calcola il prior di guasto per ciascun modello."""
        path = path or DATA_DIR / "vulnerability_data.csv"
        
        # Raggruppa i punteggi di gravità (severity scores) per modello
        modelli_severity: Dict[str, List[float]] = {}
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    model = row["Model"].strip()
                    try:
                        severity = float(row["Severity Score (1-5)"])
                        if model not in modelli_severity:
                            modelli_severity[model] = []
                        modelli_severity[model].append(severity)
                    except (ValueError, KeyError):
                        continue

       
        for model, scores in modelli_severity.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                prior_val = 0.05 + (avg_score - 1.0) / 4.0 * (0.60 - 0.05)
                self.priors_robot[model] = round(prior_val, 3)
            else:
                self.priors_robot[model] = 0.25

        if "rover" not in self.priors_robot:
            self.priors_robot["rover"] = 0.20
        if "drone" not in self.priors_robot:
            self.priors_robot["drone"] = 0.30

    def prior(self, robot_model: str) -> float:
        """P(G=1) = Prior di guasto a priori per il modello di robot selezionato."""
        # Se il modello specifico non è censito, restituisce un valore generico basato sul tipo
        if robot_model in self.priors_robot:
            return self.priors_robot[robot_model]
        if "drone" in robot_model.lower():
            return self.priors_robot.get("SwiftBot Elite", 0.30)
        return self.priors_robot.get("DeliverBot 100", 0.20)

    def aggiorna_credenza(self, robot_model: str, zona: str, anomalia_rilevata: bool) -> dict:
        """
        Inferenza esatta tramite Teorema di Bayes.
        P(G=1 | A) = P(A | G=1) * P(G=1) / P(A)
        """
        p_g = self.prior(robot_model)

        if anomalia_rilevata:
            p_a_g1 = P_ANOMALIA_DATO_GUASTO
            p_a_g0 = P_ANOMALIA_DATO_NO_GUASTO
        else:
            p_a_g1 = 1.0 - P_ANOMALIA_DATO_GUASTO
            p_a_g0 = 1.0 - P_ANOMALIA_DATO_NO_GUASTO

        # Probabilità totale dell'evidenza P(A)
        p_a = p_a_g1 * p_g + p_a_g0 * (1.0 - p_g)

        # Posterior
        p_guasto_posteriore = (p_a_g1 * p_g) / p_a if p_a > 0 else 0.0
        decisione = "STOP" if p_guasto_posteriore >= SOGLIA_STOP else "PROSEGUI"

        return {
            "zona": zona,
            "robot_model": robot_model,
            "anomalia_rilevata": anomalia_rilevata,
            "prior_guasto": round(p_g, 3),
            "posterior_guasto": round(p_guasto_posteriore, 3),
            "soglia_stop": SOGLIA_STOP,
            "decisione": decisione,
        }

    def valuta_percorso(self, percorso: List[str], robot_model: str, anomalie: Dict[str, bool]) -> List[dict]:
        """Esegue il monitoraggio in tempo reale lungo il percorso pianificato."""
        risultati = []
        for zona in percorso:
            anomalia = anomalie.get(zona, False)
            r = self.aggiorna_credenza(robot_model, zona, anomalia)
            risultati.append(r)
            if r["decisione"] == "STOP":
                break 
        return risultati
