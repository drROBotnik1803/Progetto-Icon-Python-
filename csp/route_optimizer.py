"""
csp/route_optimizer.py
======================
Ragionamento con Vincoli e Ottimizzazione — CSP
"""

import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from csp.naive_bayes import NaiveBayesClassifier
from csp.csp_solver import Variable, Constraint, SoftConstraint, CSP, CSPSolver

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Fattore di scala per convertire la probabilità di disapprovazione in penalità di costo (km equivalenti)
FASCIAPENALITA_MAX = 2.0  # Fino a +2.0 km di costo fittizio per zone ad alto conflitto


class RouteOptimizer:
    """
    Modella il percorso e l'allocazione flotta come problemi di ottimizzazione con vincoli (CSP).
    
    1. Vincoli RIGIDI (hard constraints):
        - Esclusione dei nodi non accessibili definiti dalla KB Datalog.
        - Capacità di carico dei robot coerenti con il peso dei pacchi.
        - Esclusività di assegnamento dei robot (un robot per ordine).
        
    2. Vincoli FLESSIBILI (soft constraints):
        - Penalità proporzionale all'indice di conflitto sociale stimato tramite Naive Bayes.
        - Penalità proporzionale alla probabilità di guasto intrinseca stimata dalla Rete Bayesiana.
    """

    def __init__(self):
        self.zona_profili: Dict[str, Dict[str, str]] = {}
        self.classifier = NaiveBayesClassifier()
        self.classifier.fit()  # Addestra Naive Bayes sui dati di sondaggio
        self.robot_specs: Dict[str, Dict[str, Any]] = {}
        self.load_robot_specs()

    def load_robot_specs(self) -> None:
        """Carica le specifiche e le capacità dei robot."""
        # Mappatura dei modelli presenti in vulnerability_data.csv
        # per associarli a tipologia e capacità di carico massimo.
        self.robot_specs = {
            "DeliverBot 100": {"type": "rover", "max_weight": 5.0, "price": "Budget"},
            "RoboDeliver X1": {"type": "rover", "max_weight": 4.0, "price": "Budget"},
            "CarryMaster Pro": {"type": "rover", "max_weight": 15.0, "price": "Mid-range"},
            "AutoCourier 2.0": {"type": "rover", "max_weight": 10.0, "price": "Mid-range"},
            "SwiftBot Elite": {"type": "drone", "max_weight": 4.0, "price": "Mid-range"},
            "NaviGo Delivery": {"type": "drone", "max_weight": 6.0, "price": "Premium"},
            "UrbanPorter 3": {"type": "rover", "max_weight": 20.0, "price": "Premium"},
            "LogiRobo Advanced": {"type": "rover", "max_weight": 25.0, "price": "Premium"},
        }

    def load_from_csv(self, path: Optional[Path] = None) -> None:
        """Mappa ogni zona a un profilo demografico basato su spatial_distribution_data.csv."""
        path = path or DATA_DIR / "spatial_distribution_data.csv"
        
        distretti_seattle = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["City"].strip() == "Seattle":
                        distretti_seattle.append(row)

        for i, row in enumerate(distretti_seattle[:9]):
            zid = f"Z{i+1:02d}"
            
            # 1. Living Area basata sulla densità di popolazione
            pop_density = float(row["Population Density (people/km²)"])
            if pop_density > 5000:
                living_area = "Urban"
            elif pop_density > 2000:
                living_area = "Suburban"
            else:
                living_area = "Rural"
                
            # 2. Prior ADR Exposure basata sulla Service Density di robot
            service_density = float(row["Service Density (robots/km²)"])
            if service_density > 3.0:
                exposure = "Interacted"
            elif service_density > 1.5:
                exposure = "Seen"
            else:
                exposure = "None"
                
            # 3. Education basata sul reddito medio
            income = float(row["Median Household Income (thousand USD)"])
            if income > 85.0:
                education = "Graduate Degree"
            elif income > 75.0:
                education = "Bachelor's"
            elif income > 60.0:
                education = "Some College"
            else:
                education = "High School"

            self.zona_profili[zid] = {
                "Living Area": living_area,
                "Prior ADR Exposure": exposure,
                "Education": education,
                "Neighborhood": row["Neighborhood"]
            }
            
        # Distretto Z10 di fallback
        self.zona_profili["Z10"] = {
            "Living Area": "Rural",
            "Prior ADR Exposure": "None",
            "Education": "High School",
            "Neighborhood": "Seattle-Boundary"
        }

    def penalita_zona(self, zona: str) -> float:
        """
        Soft constraint: Restituisce la penalità di attraversamento (km) basata 
        sulla probabilità che la zona abbia una percezione pubblica negativa.
        """
        if zona not in self.zona_profili:
            return 0.0
            
        profilo = self.zona_profili[zona]
        # Calcoliamo la probabilità che il giudizio sia negativo
        probs = self.classifier.predict_probability({
            "Living Area": profilo["Living Area"],
            "Prior ADR Exposure": profilo["Prior ADR Exposure"],
            "Education": profilo["Education"]
        })
        
        prob_negativo = probs.get("negativo", 0.0)
        # La penalità è proporzionale al rischio di conflitto sociale (probabilità negativo)
        return round(prob_negativo * FASCIAPENALITA_MAX, 3)

    def penalita_percorso(self, percorso: List[str]) -> dict:
        """Calcola la penalità cumulativa sul percorso e restituisce i dettagli sulle zone critiche."""
        totale = 0.0
        zone_critiche = []
        for zona in percorso:
            p = self.penalita_zona(zona)
            totale += p
            if p > 0.30:  # Soglia arbitraria per considerare una zona "critica"
                profilo = self.zona_profili.get(zona, {})
                zone_critiche.append({
                    "zona": zona,
                    "nome": profilo.get("Neighborhood", "Unknown"),
                    "penalita": p,
                    "prob_negativa_perc": f"{p / FASCIAPENALITA_MAX * 100:.1f}%"
                })
        return {
            "penalita_totale": round(totale, 3),
            "zone_critiche": zone_critiche,
            "conflitti": len(zone_critiche)
        }

    def mappa_penalita(self) -> Dict[str, float]:
        """Restituisce la mappa completa delle penalità per l'algoritmo A*."""
        return {z: self.penalita_zona(z) for z in self.zona_profili}

    def risolvi_allocazione_flotta(
        self, 
        ordini: List[Dict[str, Any]], 
        kb: Any, 
        bn: Any
    ) -> Dict[str, Any]:
        """
        Formula e risolve il problema di allocazione dei robot agli ordini tramite CSP.
        Ciascun ordine deve avere la struttura:
        {
            "id": "Ordine_1",
            "destination": "Z02",
            "weight": 3.5
        }
        """
        variabili: List[Variable] = []
        robot_models = list(self.robot_specs.keys())
        for ord_info in ordini:
            variabili.append(Variable(ord_info["id"], robot_models))
        vincoli_rigidi: List[Constraint] = []

        for ord_info in ordini:
            ord_id = ord_info["id"]
            pacco_peso = ord_info["weight"]

            def check_weight(robot_model, w=pacco_peso):
                return self.robot_specs[robot_model]["max_weight"] >= w

            vincoli_rigidi.append(
                Constraint([ord_id], check_weight, name=f"WeightCap_{ord_id}")
            )

        for ord_info in ordini:
            ord_id = ord_info["id"]
            dest = ord_info["destination"]
            pacco_peso = ord_info["weight"]

            def check_permission(robot_model, d=dest, w=pacco_peso):
                tipo = self.robot_specs[robot_model]["type"]
                # Interroga la base di conoscenza Datalog ( SLD )
                return kb.accessibile(d, tipo, w)

            vincoli_rigidi.append(
                Constraint([ord_id], check_permission, name=f"AccessKB_{ord_id}_{dest}")
            )

        for i in range(len(ordini)):
            for j in range(i + 1, len(ordini)):
                id1 = ordini[i]["id"]
                id2 = ordini[j]["id"]

                def check_diff(r1, r2):
                    return r1 != r2

                vincoli_rigidi.append(
                    Constraint([id1, id2], check_diff, name=f"Exclusivity_{id1}_{id2}")
                )

        vincoli_flessibili: List[SoftConstraint] = []

        for ord_info in ordini:
            ord_id = ord_info["id"]
            dest = ord_info["destination"]

            def cost_social(robot_model, d=dest):
                return self.penalita_zona(d)

            vincoli_flessibili.append(
                SoftConstraint([ord_id], cost_social, name=f"SocialCost_{ord_id}")
            )

        # Soft Constraint 2: Sicurezza / Affidabilità
        # Minimizza il prior di guasto del robot assegnato (Belief Network prior)
        for ord_info in ordini:
            ord_id = ord_info["id"]

            def cost_fault(robot_model):
                prior_fault = bn.prior(robot_model)
                # Scaliamo il prior di guasto per pesarlo analogamente al costo sociale
                return prior_fault * 3.0

            vincoli_flessibili.append(
                SoftConstraint([ord_id], cost_fault, name=f"FaultCost_{ord_id}")
            )

        # 4. Creazione ed esecuzione del CSP
        problem = CSP(variabili, vincoli_rigidi, vincoli_flessibili)
        solver = CSPSolver(problem)
        assegnazione, costo_ottimo = solver.solve()

        return {
            "successo": assegnazione is not None,
            "assegnazione": assegnazione,
            "costo": costo_ottimo,
            "backtracks": solver.backtracks_count,
            "vincoli_rigidi_totali": len(vincoli_rigidi),
            "vincoli_flessibili_totali": len(vincoli_flessibili)
        }

