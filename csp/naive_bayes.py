"""
csp/naive_bayes.py
===================
Modello di Apprendimento Supervisionato — Naive Bayes Classifier
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"


class NaiveBayesClassifier:
    """
    Classificatore Naive Bayes discreto implementato da zero.
    Ottimizzato con lisciamento di Laplace (Laplace smoothing) per evitare probabilità nulle.
    """

    def __init__(self):
        self.classes: Set[str] = set()
        self.class_priors: Dict[str, float] = {}
        self.likelihoods: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.feature_domains: Dict[str, Set[str]] = {}

    def fit(self, dataset_path: Optional[Path] = None) -> Dict[str, Any]:
        """Addestra il modello calcolando a priori e verosimiglianze dai dati reali."""
        dataset_path = dataset_path or DATA_DIR / "public_perception_data.csv"
        
        features = ["Living Area", "Prior ADR Exposure", "Education"]
        
        rows: List[Dict[str, str]] = []
        class_counts: Dict[str, int] = {}
        feature_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        
        # Inizializza strutture conteggio
        for f in features:
            feature_counts[f] = {}
            self.feature_domains[f] = set()

        total_samples = 0
        if dataset_path.exists():
            with open(dataset_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    try:
                        attitude_val = float(row["General ADR Attitude (1-5)"])
                        label = "positivo" if attitude_val >= 3.0 else "negativo"
                        
                        self.classes.add(label)
                        class_counts[label] = class_counts.get(label, 0) + 1
                        total_samples += 1
                        
                        # Memorizza valori feature per i domini
                        item = {}
                        for feat in features:
                            val = row[feat].strip()
                            item[feat] = val
                            self.feature_domains[feat].add(val)
                        
                        item["label"] = label
                        rows.append(item)
                    except (ValueError, KeyError):
                        continue

        # Inizializza conteggi condizionati
        for feat in features:
            for label in self.classes:
                feature_counts[feat][label] = {val: 0 for val in self.feature_domains[feat]}

        # Esegui conteggi
        for item in rows:
            label = item["label"]
            for feat in features:
                val = item[feat]
                feature_counts[feat][label][val] += 1

        # 2. Calcolo dei Priori con lisciamento
        for label in self.classes:
            # P(C) = (count(C) + 1) / (total + |Classes|)
            self.class_priors[label] = (class_counts[label] + 1) / (total_samples + len(self.classes))

        # 3. Calcolo delle Likelihoods con lisciamento di Laplace
        # P(Xi = x | C = c) = (count(Xi = x e C = c) + 1) / (count(C = c) + |Dom(Xi)|)
        for feat in features:
            self.likelihoods[feat] = {}
            dom_size = len(self.feature_domains[feat])
            for label in self.classes:
                self.likelihoods[feat][label] = {}
                c_count = class_counts[label]
                for val in self.feature_domains[feat]:
                    val_count = feature_counts[feat][label][val]
                    # Applica formula di Laplace
                    prob = (val_count + 1) / (c_count + dom_size)
                    self.likelihoods[feat][label][val] = prob

        # Ritorna metriche di training
        return {
            "samples": total_samples,
            "classes": list(self.classes),
            "priors": self.class_priors,
            "features": features
        }

    def predict_probability(self, features_dict: Dict[str, str]) -> Dict[str, float]:
        """Calcola la distribuzione di probabilità a posteriori P(C | X) per le classi."""
        scores: Dict[str, float] = {}
        
        for label in self.class_priors:
            score = self.class_priors[label]
            for feat, val in features_dict.items():
                if feat in self.likelihoods:
                    # Se il valore non è nel training set, usiamo il lisciamento di Laplace generico
                    dom_size = len(self.feature_domains[feat])
                    prob = self.likelihoods[feat][label].get(val, 1.0 / (dom_size + 1))
                    score *= prob
            scores[label] = score

        # Normalizzazione dei punteggi per ottenere una vera distribuzione di probabilità (somma = 1)
        total_score = sum(scores.values())
        if total_score > 0:
            return {label: val / total_score for label, val in scores.items()}
        return {label: 1.0 / len(self.class_priors) for label in self.class_priors}

    def predict(self, features_dict: Dict[str, str]) -> str:
        """Predice la classe a massima verosimiglianza (MAP)."""
        probs = self.predict_probability(features_dict)
        return max(probs, key=probs.get)

