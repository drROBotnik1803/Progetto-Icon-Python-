"""
pathfinding/astar.py
====================
Ricerca Informata su Grafo — Algoritmo A*
Elemento teorico 2: Pruning nello Spazio degli Stati (Cycle Pruning vs Multiple-Path Pruning).
Confronta l'efficienza della ricerca con e senza potature.
"""

import csv
import math
import heapq
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"


@dataclass
class Nodo:
    id: str
    nome: str
    lat: float
    lon: float


@dataclass(order=True)
class Stato:
    f: float
    g: float = field(compare=False)
    nodo_id: str = field(compare=False)
    path: List[str] = field(compare=False, default_factory=list)


class GrafoCitta:
    """Grafo pesato della città costruito basandosi sui distretti di Seattle."""

    def __init__(self):
        self.nodi: Dict[str, Nodo] = {}
        self.archi: Dict[str, List[Tuple[str, float]]] = {}  # nodo -> [(vicino, dist)]

    def load_from_csv(self, path: Optional[Path] = None) -> None:
        """Costruisce un grafo connesso basato sui 9 distretti di Seattle."""
        path = path or DATA_DIR / "spatial_distribution_data.csv"
        
        distretti = []
        if path.exists():
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["City"].strip() == "Seattle":
                        distretti.append(row)

        # Griglia 3x3 per coordinate spaziali
        # Row 0: Z01, Z02, Z03 (lat=47.60, 47.61, 47.62; lon=-122.33)
        # Row 1: Z04, Z05, Z06 (lat=47.60, 47.61, 47.62; lon=-122.34)
        # Row 2: Z07, Z08, Z09 (lat=47.60, 47.61, 47.62; lon=-122.35)
        coords = {
            "Z01": (47.60, -122.33), "Z02": (47.61, -122.33), "Z03": (47.62, -122.33),
            "Z04": (47.60, -122.34), "Z05": (47.61, -122.34), "Z06": (47.62, -122.34),
            "Z07": (47.60, -122.35), "Z08": (47.61, -122.35), "Z09": (47.62, -122.35)
        }

        # Asseriamo i nodi
        for i, row in enumerate(distretti[:9]):
            zid = f"Z{i+1:02d}"
            lat, lon = coords[zid]
            self.nodi[zid] = Nodo(id=zid, nome=row["Neighborhood"], lat=lat, lon=lon)
            self.archi[zid] = []

        # Aggiungiamo Z10 in caso di fallback
        if "Z10" not in self.nodi:
            self.nodi["Z10"] = Nodo(id="Z10", nome="Seattle-Boundary", lat=47.63, lon=-122.36)
            self.archi["Z10"] = []

        # Adjacencies (connessioni) nella griglia 3x3
        connessioni = [
            ("Z01", "Z02", 1.2), ("Z01", "Z04", 1.5),
            ("Z02", "Z03", 1.1), ("Z02", "Z05", 1.3),
            ("Z03", "Z06", 1.4),
            ("Z04", "Z05", 1.2), ("Z04", "Z07", 1.6),
            ("Z05", "Z06", 1.1), ("Z05", "Z08", 1.4),
            ("Z06", "Z09", 1.5),
            ("Z07", "Z08", 1.3),
            ("Z08", "Z09", 1.2),
            # Connessioni con Z10
            ("Z09", "Z10", 2.0), ("Z06", "Z10", 2.2)
        ]

        # Rendi il grafo non orientato (bidirezionale)
        for u, v, d in connessioni:
            if u in self.nodi and v in self.nodi:
                self.archi[u].append((v, d))
                self.archi[v].append((u, d))

    def distanza_euclidea(self, a: str, b: str) -> float:
        """Euristica ammissibile h(n): distanza in linea d'aria."""
        if a not in self.nodi or b not in self.nodi:
            return 0.0
        na, nb = self.nodi[a], self.nodi[b]
        dlat = (na.lat - nb.lat) * 111.0
        dlon = (na.lon - nb.lon) * 111.0 * math.cos(math.radians(na.lat))
        return math.sqrt(dlat ** 2 + dlon ** 2)


class AStarPlanner:
    """
    Pianificatore di percorso A*.
    Fornisce la pianificazione con 3 modalità di potatura per confronto didattico.
    """

    def __init__(self, grafo: GrafoCitta):
        self.grafo = grafo

    def pianifica(
        self,
        start: str,
        goal: str,
        nodi_accessibili: Set[str],
        penalita: Optional[Dict[str, float]] = None,
        pruning_mode: str = "MPP"  # "NONE", "CP" (Cycle Pruning), "MPP" (Multiple Path Pruning)
    ) -> Dict[str, Any]:
        """
        Pianifica il percorso da start a goal.
        pruning_mode: determina la tecnica di potatura.
        """
        penalita = penalita or {}
        g = self.grafo

        # Nodi accessibili: start e goal devono essere inclusi
        accessibili = nodi_accessibili | {start, goal}

        frontiera: List[Stato] = []
        heapq.heappush(frontiera, Stato(
            f=g.distanza_euclidea(start, goal),
            g=0.0,
            nodo_id=start,
            path=[start],
        ))

        # Strutture dati per le statistiche e potature
        visitati: Dict[str, float] = {}  # Per MPP: closed list (nodo_id -> costo g minimo)
        nodi_esplosi = 0
        stati_generati = 1
        limite_esplosione = 5000  # Evita loop infiniti in modalità "NONE"

        while frontiera:
            if nodi_esplosi >= limite_esplosione:
                break  # Taglio di sicurezza

            stato = heapq.heappop(frontiera)
            nid, costo, path = stato.nodo_id, stato.g, stato.path
            nodi_esplosi += 1

            # --- 1. MULTIPLE-PATH PRUNING (MPP) ---
            if pruning_mode == "MPP":
                if nid in visitati and visitati[nid] <= costo:
                    continue
                visitati[nid] = costo

            # Verifica obiettivo raggiunto
            if nid == goal:
                return {
                    "trovato": True,
                    "percorso": path,
                    "costo_km": round(costo, 3),
                    "nodi_esplosi": nodi_esplosi,
                    "stati_generati": stati_generati,
                    "lunghezza_percorso": len(path),
                }

            # Esplorazione vicini
            for vicino, dist in g.archi.get(nid, []):
                if vicino not in accessibili:
                    continue

                # --- 2. CYCLE PRUNING (CP) ---
                if pruning_mode == "CP":
                    if vicino in path:
                        continue  # Evita i cicli diretti nel percorso corrente

                costo_arco = dist + penalita.get(vicino, 0.0)
                nuovo_g = costo + costo_arco
                h = g.distanza_euclidea(vicino, goal)

                heapq.heappush(frontiera, Stato(
                    f=nuovo_g + h,
                    g=nuovo_g,
                    nodo_id=vicino,
                    path=path + [vicino],
                ))
                stati_generati += 1

        return {
            "trovato": False,
            "percorso": [],
            "costo_km": float("inf"),
            "nodi_esplosi": nodi_esplosi,
            "stati_generati": stati_generati,
            "lunghezza_percorso": 0
        }

