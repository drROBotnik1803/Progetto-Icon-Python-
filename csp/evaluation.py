"""
csp/evaluation.py
=================
Valutazione del Modello di Apprendimento Supervisionato.
Esegue uno split Train/Test (80/20) e calcola:
- Accuracy Globale
- Precision, Recall e F1-Score per le classi "positivo" e "negativo"
- Matrice di Confusione
"""

import csv
import random
from pathlib import Path
from typing import List, Dict, Any
from csp.naive_bayes import NaiveBayesClassifier


def valuta_naive_bayes():
    # Definiamo i percorsi
    base_dir = Path(__file__).resolve().parent.parent
    dataset_path = base_dir / "dataset" / "public_perception_data.csv"
    train_path = base_dir / "dataset" / "public_perception_train_temp.csv"
    test_path = base_dir / "dataset" / "public_perception_test_temp.csv"

    if not dataset_path.exists():
        print(f"Errore: Dataset {dataset_path} non trovato.")
        return

    # Lettura delle righe del dataset
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        rows = list(reader)

    # Split deterministico (seed=42) per riproducibilità
    random.seed(42)
    random.shuffle(rows)
    split_idx = int(len(rows) * 0.8)
    train_rows = rows[:split_idx]
    test_rows = rows[split_idx:]

    # Salvataggio temporaneo del train set
    with open(train_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(train_rows)

    # Salvataggio temporaneo del test set
    with open(test_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(test_rows)

    try:
        # Istanziamo e addestriamo il classificatore sul train set
        clf = NaiveBayesClassifier()
        clf.fit(dataset_path=train_path)

        # Contatori per la matrice di confusione
        tp = 0  # True Positives
        fp = 0  # False Positives
        tn = 0  # True Negatives
        fn = 0  # False Negatives

        for row in test_rows:
            try:
                attitude_val = float(row["General ADR Attitude (1-5)"])
                actual = "positivo" if attitude_val >= 3.0 else "negativo"

                features = {
                    "Living Area": row["Living Area"].strip(),
                    "Prior ADR Exposure": row["Prior ADR Exposure"].strip(),
                    "Education": row["Education"].strip()
                }

                predicted = clf.predict(features)

                if actual == "positivo" and predicted == "positivo":
                    tp += 1
                elif actual == "negativo" and predicted == "positivo":
                    fp += 1
                elif actual == "negativo" and predicted == "negativo":
                    tn += 1
                elif actual == "positivo" and predicted == "negativo":
                    fn += 1
            except (ValueError, KeyError):
                continue

        total = tp + fp + tn + fn
        accuracy = (tp + tn) / total if total > 0 else 0

        # Metodologia di calcolo metriche
        prec_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_pos = 2 * prec_pos * rec_pos / (prec_pos + rec_pos) if (prec_pos + rec_pos) > 0 else 0

        prec_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) > 0 else 0

        print("\n" + "="*65)
        print("    VALUTAZIONE PRESTAZIONI: CLASSFICATORE NAIVE BAYES (80/20 Split) ")
        print("="*65)
        print(f"  Dimensioni Train Set (80%): {len(train_rows)} campioni")
        print(f"  Dimensioni Test Set  (20%): {len(test_rows)} campioni")
        print(f"  Campioni di test valutati:  {total}")
        print("-"*65)
        print(f"  Accuracy Globale:           {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("-"*65)
        print("  Metriche Dettagliate per Classe:")
        print(f"    Classe 'POSITIVO' (Attaccamento favorevole ai robot):")
        print(f"      Precision:              {prec_pos:.4f}")
        print(f"      Recall (Sensitivity):   {rec_pos:.4f}")
        print(f"      F1-Score:               {f1_pos:.4f}")
        print(f"    Classe 'NEGATIVO' (Attaccamento sfavorevole ai robot):")
        print(f"      Precision:              {prec_neg:.4f}")
        print(f"      Recall (Specificity):   {rec_neg:.4f}")
        print(f"      F1-Score:               {f1_neg:.4f}")
        print("-"*65)
        print("  Matrice di Confusione:")
        print(f"                       Predetto POSITIVO  |  Predetto NEGATIVO")
        print(f"    Attuale POSITIVO   {tp:<17}  |  {fn:<17}")
        print(f"    Attuale NEGATIVO   {fp:<17}  |  {tn:<17}")
        print("="*65 + "\n")

    finally:
        # Pulizia dei file temporanei
        if train_path.exists():
            train_path.unlink()
        if test_path.exists():
            test_path.unlink()

