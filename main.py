"""
main.py
=======
Pipeline integrata con interfaccia a riga di comando (CLI) interattiva.
Consente di esplorare e coordinare tutti i moduli teorici del progetto:
1. KB (SLD Datalog) per l'accessibilità normativa
2. CSP (Backtracking Solver) per l'allocazione flotta
3. A* (Cycle Pruning vs MPP) per la pianificazione di percorsi
4. Rete Bayesiana (Inferenza Esatta) per il monitoraggio guasti sensori
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any

# Aggiunge la cartella corrente al path di ricerca dei moduli
sys.path.insert(0, str(Path(__file__).parent))

from kb.knowledge_base import KnowledgeBase
from pathfinding.astar import GrafoCitta, AStarPlanner
from csp.route_optimizer import RouteOptimizer
from bayes.belief_network import BeliefNetwork
from csp.evaluation import valuta_naive_bayes


def stampa_intestazione(titolo: str):
    """Utility per stampare titoli formattati in console."""
    print("\n" + "=" * 75)
    print(f" {titolo.upper():^73}")
    print("=" * 75)


def leggi_scelta(min_val: int, max_val: int, prompt: str = "Seleziona un'opzione: ") -> int:
    """Utility per leggere e validare una scelta numerica dall'utente."""
    while True:
        try:
            scelta = int(input(prompt).strip())
            if min_val <= scelta <= max_val:
                return scelta
            print(f"Valore non valido. Inserire un numero compreso tra {min_val} e {max_val}.")
        except ValueError:
            print("Input non valido. Inserire un valore numerico.")


def seleziona_robot(csp: RouteOptimizer) -> str:
    """Consente all'utente di selezionare interattivamente un modello di robot."""
    print("\nModelli di Robot Disponibili:")
    robot_list = list(csp.robot_specs.keys())
    for idx, name in enumerate(robot_list):
        specs = csp.robot_specs[name]
        print(f"  {idx + 1}. {name:<18} | Tipo: {specs['type']:<6} | Carico Max: {specs['max_weight']:>4.1f} kg | Fascia: {specs['price']}")
    scelta = leggi_scelta(1, len(robot_list), "Scegli il modello di robot (numero): ")
    return robot_list[scelta - 1]


def seleziona_zona(prompt: str) -> str:
    """Consente all'utente di selezionare una zona (Z01-Z10)."""
    while True:
        zona = input(prompt).strip().upper()
        # Aggiunge uno zero se l'utente inserisce es. Z1
        if len(zona) == 2 and zona.startswith("Z"):
            zona = f"Z0{zona[1]}"
        if zona in [f"Z{i:02d}" for i in range(1, 11)]:
            return zona
        print("Zona non valida. Inserire una sigla corretta da Z01 a Z10.")


def visualizza_flotta(csp: RouteOptimizer, bn: BeliefNetwork):
    """Opzione 1: Mostra le specifiche dettagliate di tutti i robot della flotta."""
    stampa_intestazione("Specifiche Flotta Robotica")
    print(f"  {'Modello':<18} | {'Tipo':<6} | {'Portata Max':<11} | {'Fascia':<9} | {'P(Guasto Prior)'}")
    print(f"  {'-'*18}-|-{'-'*6}-|-{'-'*11}-|-{'-'*9}-|-{'-'*15}")
    for name, specs in sorted(csp.robot_specs.items()):
        prior_fault = bn.prior(name)
        print(f"  {name:<18} | {specs['type']:<6} | {specs['max_weight']:>7.1f} kg | {specs['price']:<9} | {prior_fault:>10.3f}")
    print("\n  [Nota] P(Guasto Prior) è calcolato normalizzando i dati di severità vulnerabilità (CVSS)")
    print("  e rappresenta la fragilità intrinseca stimata per quel modello.")


def visualizza_distretti(kb: KnowledgeBase, csp: RouteOptimizer):
    """Opzione 2: Elenca i distretti di Seattle estratti dalla KB Datalog e profili CSP."""
    stampa_intestazione("Elenco Distretti e Regolamentazioni (KB)")
    print(f"  {'Zona':<4} | {'Distretto (Seattle)':<24} | {'Droni':<6} | {'Rover':<6} | {'Limite Peso':<11} | {'Penalità Sociale'}")
    print(f"  {'-'*4}-|-{'-'*24}-|-{'-'*6}-|-{'-'*6}-|-{'-'*11}-|-{'-'*17}")
    
    # Estraiamo i fatti per ciascuna zona
    for i in range(1, 11):
        zid = f"Z{i:02d}"
        profilo = csp.zona_profili.get(zid, {})
        nome = profilo.get("Neighborhood", "Sconosciuto")
        
        # Interroga fatti di permesso nella KB
        drone_ok = "SI" if any(c.head.args[0] == zid for c in kb.clauses if not c.body and c.head.name == "permesso_droni") else "NO"
        rover_ok = "SI" if any(c.head.args[0] == zid for c in kb.clauses if not c.body and c.head.name == "permesso_rover") else "NO"
        
        # Interroga limite di peso
        limite = 0.0
        for c in kb.clauses:
            if not c.body and c.head.name == "limite_peso" and c.head.args[0] == zid:
                limite = float(c.head.args[1])
                break
                
        # Calcola penalità sociale con Naive Bayes
        penalita = csp.penalita_zona(zid)
        
        print(f"  {zid:<4} | {nome:<24} | {drone_ok:<6} | {rover_ok:<6} | {limite:>7.1f} kg | {penalita:>10.3f} km")


def esegui_query_sld(kb: KnowledgeBase, csp: RouteOptimizer):
    """Sub-Opzione 1: Interrogazione manuale Datalog con albero di prova SLD."""
    stampa_intestazione("Interrogazione Base di Conoscenza (SLD Datalog)")
    robot_model = seleziona_robot(csp)
    tipo = csp.robot_specs[robot_model]["type"]
    peso = float(input("Inserire il peso del pacco (kg): "))
    zona = seleziona_zona("Inserire la zona da interrogare (Z01-Z10): ")
    
    print("\nEsecuzione della query logica in corso...")
    risultato = kb.query(zona, tipo, peso)
    
    print("\n--- RISULTATO QUERY ---")
    if risultato["accessibile"]:
        print(f"  >>> ACCESSO CONSENTITO! Il robot {robot_model} ({tipo}) può accedere alla zona {zona}.")
    else:
        print(f"  >>> ACCESSO NEGATO! Il robot {robot_model} ({tipo}) NON ha i permessi o supera il limite di peso.")
    
    print("\n--- TRACCIA DELLA DIMOSTRAZIONE SLD (PROOF TREE) ---")
    print(risultato["proof_tree_trace"])


def esegui_csp_flotta(kb: KnowledgeBase, csp: RouteOptimizer, bn: BeliefNetwork):
    """Sub-Opzione 2: Modella e risolve il CSP di allocazione flotta per più ordini."""
    stampa_intestazione("CSP: Allocazione Ottima della Flotta Consegne")
    print("1. Usa un lotto di ordini predefinito (didattico)")
    print("2. Inserisci ordini personalizzati")
    scelta = leggi_scelta(1, 2)
    
    ordini: List[Dict[str, Any]] = []
    if scelta == 1:
        ordini = [
            {"id": "Ordine_A (Urgente)", "destination": "Z01", "weight": 3.0},
            {"id": "Ordine_B (Pesante)", "destination": "Z02", "weight": 12.0},
            {"id": "Ordine_C (Standard)", "destination": "Z05", "weight": 1.5},
            {"id": "Ordine_D (Leggero)", "destination": "Z06", "weight": 2.2},
        ]
        print("\nLotto di ordini predefinito caricato:")
        for o in ordini:
            print(f"  - {o['id']}: Destinazione {o['destination']}, Peso {o['weight']} kg")
    else:
        num_ordini = leggi_scelta(1, 4, "Quanti ordini vuoi pianificare? (Max 4 per esclusività robot): ")
        for idx in range(num_ordini):
            print(f"\nInserimento Ordine #{idx+1}:")
            ord_id = f"Ordine_{chr(65 + idx)} (Custom)"
            dest = seleziona_zona("  Zona di destinazione (Z01-Z10): ")
            peso = float(input("  Peso del pacco (kg): "))
            ordini.append({"id": ord_id, "destination": dest, "weight": peso})

    print("\nFormulazione del problema CSP ed esecuzione del Backtracking Solver...")
    risultato = csp.risolvi_allocazione_flotta(ordini, kb, bn)
    
    print("\n--- RISULTATO RISOLUTORE CSP ---")
    if risultato["successo"]:
        print(f"  Assegnazione ottima trovata con successo!")
        print(f"  Costo/Penalità Totale dell'assegnazione: {risultato['costo']:.3f}")
        print(f"  Numero di nodi/backtrack esplorati: {risultato['backtracks']}")
        print(f"  Vincoli rigidi verificati: {risultato['vincoli_rigidi_totali']}")
        print(f"  Vincoli flessibili valutati: {risultato['vincoli_flessibili_totali']}")
        print("\n  Accoppiamenti Ottimizzati:")
        for ord_id, robot in risultato["assegnazione"].items():
            specs = csp.robot_specs[robot]
            prior_fail = bn.prior(robot)
            dest_zona = next(o["destination"] for o in ordini if o["id"] == ord_id)
            penalita_soc = csp.penalita_zona(dest_zona)
            
            print(f"    * {ord_id:<20} -> {robot:<18} ({specs['type']:<5}, max {specs['max_weight']:>4.1f}kg)")
            print(f"      [Analisi costi] Sicurezza (P_Guasto Prior={prior_fail:.3f}) | Accettazione Sociale (Penalità={penalita_soc:.3f} km)")
    else:
        print("  ❌ IMPOSSIBILE TROVARE UNA SOLUZIONE COERENTE!")
        print("  I vincoli rigidi (esclusività dei robot, limiti di portata o autorizzazioni Datalog) non sono soddisfatti.")
        print(f"  Nodi esplorati prima del fallimento: {risultato['backtracks']}")


def esegui_navigazione_simulata(
    kb: KnowledgeBase, 
    grafo: GrafoCitta, 
    csp: RouteOptimizer, 
    bn: BeliefNetwork
):
    """Sub-Opzione 3: Pianificazione A* e simulazione interattiva della rete Bayesiana."""
    stampa_intestazione("Pianificazione A* e Monitoraggio Bayesiano")
    
    robot_model = seleziona_robot(csp)
    tipo = csp.robot_specs[robot_model]["type"]
    peso = float(input("Inserire il peso del carico (kg): "))
    start = seleziona_zona("Inserire la zona di partenza (Z01-Z10): ")
    goal = seleziona_zona("Inserire la zona di arrivo (Z01-Z10): ")
    
    if start == goal:
        print("Errore: Partenza e arrivo coincidono.")
        return

    # 1. Risoluzione Datalog per determinare l'accessibilità complessiva delle zone
    accessibili: Set[str] = set()
    for z in grafo.nodi:
        if kb.accessibile(z, tipo, peso):
            accessibili.add(z)
            
    print(f"\n[Datalog SLD] Zone accessibili per {robot_model} ({peso}kg): {sorted(list(accessibles := list(accessibili)))}")
    
    if start not in accessibili:
        print(f"Errore: La zona di partenza {start} non è accessibile per le specifiche correnti.")
        return
    if goal not in accessibili:
        print(f"Errore: La zona di arrivo {goal} non è accessibile per le specifiche correnti.")
        return

    # 2. Pianificazione A* con le tre tecniche di potatura per confronto didattico
    print("\nEsecuzione pianificazione A* (Confronto tecniche di potatura)...")
    penalita = csp.mappa_penalita()
    planner = AStarPlanner(grafo)
    
    risultati_pruning: Dict[str, dict] = {}
    for mode in ["NONE", "CP", "MPP"]:
        risultati_pruning[mode] = planner.pianifica(start, goal, accessibili, penalita, pruning_mode=mode)

    # Stampa la tabella di confronto
    print(f"\n  {'Pruning Mode':<15} | {'Trovato':<8} | {'Costo (km)':<12} | {'Nodi Esplosi':<13} | {'Stati Gen':<10} | {'Percorso'}")
    print(f"  {'-'*15}-|-{'-'*8}-|-{'-'*12}-|-{'-'*13}-|-{'-'*10}-|-{'-'*30}")
    for mode in ["NONE", "CP", "MPP"]:
        r = risultati_pruning[mode]
        costo_str = f"{r['costo_km']:.3f}" if r["trovato"] else "INF"
        percorso_str = " -> ".join(r["percorso"]) if r["trovato"] else "N/A"
        print(f"  {mode:<15} | {str(r['trovato']):<8} | {costo_str:<12} | {r['nodi_esplosi']:<13} | {r['stati_generati']:<10} | {percorso_str}")

    mpp_res = risultati_pruning["MPP"]
    if not mpp_res["trovato"]:
        print(f"\n[Errore] Nessun percorso accessibile trovato con MPP.")
        return

    percorso = mpp_res["percorso"]
    
    # 3. CSP: Analisi finale del percorso (social penalties)
    analisi_csp = csp.penalita_percorso(percorso)
    print(f"\n--- DETTAGLIO SOCIALE DEL PERCORSO ---")
    print(f"  Costo base geometrico: {mpp_res['costo_km'] - analisi_csp['penalita_totale']:.3f} km")
    print(f"  Penalità di accettazione sociale totale: +{analisi_csp['penalita_totale']:.3f} km")
    if analisi_csp["zone_critiche"]:
        print("  Zone a rischio di protesta pubblica attraversate:")
        for z in analisi_csp["zone_critiche"]:
            print(f"    - Zona {z['zona']} ({z['nome']}): penalità +{z['penalita']:.3f} km (P(Disapprovazione)={z['prob_negativa_perc']})")

    # 4. Rete Bayesiana: Monitoraggio real-time con inserimento interattivo di anomalie
    stampa_intestazione("Navigazione e Diagnostica in Tempo Reale (Bayes)")
    print(f"  Robot impiegato: {robot_model} | P(Guasto Intrinseco Prior) = {bn.prior(robot_model):.3f}")
    print("  Sarà richiesto per ogni tappa se simulare un'anomalia ai sensori.")
    
    for idx, zona in enumerate(percorso):
        nome_distretto = grafo.nodi[zona].nome
        print(f"\nTappa #{idx+1}: Ingresso nella zona {zona} ({nome_distretto})")
        
        scelta_anomalia = input("  Rilevare anomalia sensore per questa tappa? (y/N): ").strip().lower()
        anomalia = scelta_anomalia == "y"
        
        # Calcolo posterior con Rete Bayesiana
        r = bn.aggiorna_credenza(robot_model, zona, anomalia)
        
        anomalia_str = "⚠️ RILEVATA!" if anomalia else "REGOLARE"
        print(f"    Stato sensori: {anomalia_str}")
        print(f"    P(Guasto | Sensore) = {r['posterior_guasto']:.3f}  [Soglia di arresto: {r['soglia_stop']:.2f}]")
        
        if r["decisione"] == "STOP":
            print(f"\n🚨 [NAVIGAZIONE INTERROTTA] Rischio sicurezza critico a {zona}!")
            print(f"   Il robot si arresta in sicurezza per anomalie persistenti o elevata fragilità.")
            return

    print(f"\n🎉 [MISSIONE COMPLETATA] Consegna effettuata con successo a {goal}!")


def menu_operativo(
    kb: KnowledgeBase, 
    grafo: GrafoCitta, 
    csp: RouteOptimizer, 
    bn: BeliefNetwork
):
    """Sotto-menu per le operazioni di simulazione e ragionamento."""
    while True:
        stampa_intestazione("Menu Operazioni e Simulazione")
        print("1. Interroga Accessibilità Zona (Datalog SLD)")
        print("2. Risolvi Allocazione Flotta Consegne (CSP Solver)")
        print("3. Pianifica e Simula Consegna (A* & Rete Bayesiana)")
        print("4. Ritorna al Menu Principale")
        
        scelta = leggi_scelta(1, 4)
        if scelta == 1:
            esegui_query_sld(kb, csp)
        elif scelta == 2:
            esegui_csp_flotta(kb, csp, bn)
        elif scelta == 3:
            esegui_navigazione_simulata(kb, grafo, csp, bn)
        elif scelta == 4:
            break


def main():
    # Caricamento e inizializzazione dei dati
    print("Inizializzazione dei moduli del sistema in corso...")
    
    kb = KnowledgeBase()
    kb.load_from_csv()
    
    grafo = GrafoCitta()
    grafo.load_from_csv()
    
    csp = RouteOptimizer()
    csp.load_from_csv()
    
    bn = BeliefNetwork()
    bn.load_from_csv()

    while True:
        stampa_intestazione("ROBOT DELIVERY SYSTEM - SEATTLE PLANNER")
        print("1. Visualizza Specifiche Flotta Robotica")
        print("2. Visualizza Elenco Distretti e Accessibilità (KB Datalog)")
        print("3. Valuta Classificatore Naive Bayes (Supervised Learning)")
        print("4. Apri Menu Operativo e Simulatore Consegne")
        print("5. Esci")
        
        scelta = leggi_scelta(1, 5)
        
        if scelta == 1:
            visualizza_flotta(csp, bn)
        elif scelta == 2:
            visualizza_distretti(kb, csp)
        elif scelta == 3:
            valuta_naive_bayes()
        elif scelta == 4:
            menu_operativo(kb, grafo, csp, bn)
        elif scelta == 5:
            print("\nSpegnimento del sistema di pianificazione. Arrivederci!\n")
            break


if __name__ == "__main__":
    main()
