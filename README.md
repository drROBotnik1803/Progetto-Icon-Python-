# Progetto ICON: Sistema di Pianificazione per Consegne Autonome

Questo progetto è un pianificatore intelligente per la gestione e la sicurezza di una flotta di robot per le consegne a domicilio a Seattle. Il sistema combina logica deduttiva, ottimizzazione a vincoli, apprendimento automatico e ragionamento probabilistico per calcolare percorsi sicuri ed efficienti.

---

## Struttura del Progetto

Il codice è suddiviso in moduli indipendenti e puliti per tenere separate le diverse parti teoriche:

* **`main.py`**: Il punto di ingresso del programma che avvia l'interfaccia interattiva a riga di comando.
* **`kb/knowledge_base.py`**: La base di conoscenza relazionale in Datalog con motore di risoluzione SLD.
* **`csp/`**: Contiene il codice per l'ottimizzazione a vincoli e la classificazione demografica:
  * `csp_solver.py`: Risolutore CSP generico basato su backtracking ed euristica MRV.
  * `route_optimizer.py`: Modellazione del problema di allocazione flotta (CSP) e calcolo delle penalità sociali.
  * `naive_bayes.py`: Il classificatore probabilistico Naive Bayes.
  * `evaluation.py`: Script per calcolare le prestazioni del classificatore (accuracy, precision, recall, F1).
* **`pathfinding/astar.py`**: Algoritmo di ricerca A* con confronto tra le modalità di potatura degli stati.
* **`bayes/belief_network.py`**: Rete Bayesiana per stimare in tempo reale il rischio di guasto ai sensori.
* **`dataset/`**: I dati in formato CSV relativi a demografia, percorsi, vulnerabilità dei robot e feedback degli utenti.

---

## Come funzionano i vari moduli

### 1. KB e Risoluzione SLD (`kb/knowledge_base.py`)
Per gestire le regole amministrative di transito nei distretti di Seattle, abbiamo implementato un motore Datalog da zero. Le regole logiche verificano se un certo tipo di robot (drone o rover) può attraversare una zona in base al peso del pacco e alle restrizioni locali.

Effettuando una query, il motore esegue l'unificazione dei termini e il backtracking logico (Risoluzione SLD), stampando l'albero di prova (*proof tree*) con tutti i passaggi logici che hanno portato alla decisione.

### 2. Classificatore Naive Bayes (`csp/naive_bayes.py`)
Utilizziamo un classificatore Naive Bayes discreto (con lisciamento di Laplace per evitare probabilità nulle) per prevedere se gli abitanti di un distretto avranno una reazione positiva o negativa al passaggio dei robot. Le previsioni si basano su tre fattori demografici: densità della zona, precedente esposizione ai robot e livello di istruzione medio.

Nel file `csp/evaluation.py` è presente una suite di test che divide il dataset (80% train, 20% test) e calcola accuratezza, precisione, recall e la matrice di confusione del modello.

### 3. Risolutore CSP e Allocazione Flotta (`csp/csp_solver.py` & `csp/route_optimizer.py`)
Invece di limitarci a sommare i costi, abbiamo implementato un vero motore CSP generico. Il problema è l'**Allocazione della Flotta**: dato un lotto di consegne, dobbiamo associare ad ognuna il robot ottimale.

* **Vincoli Rigidi (Hard)**: Il robot deve avere portata sufficiente, deve essere autorizzato ad accedere alla zona (controllo tramite Datalog KB) e ogni robot può gestire una sola consegna alla volta (vincolo All-Different).

* **Vincoli Flessibili (Soft)**: Minimizziamo sia la disapprovazione sociale (stimata con Naive Bayes) sia la probabilità a priori di guasto del modello di robot (derivata dai dati di vulnerabilità).

### 4. Ricerca del percorso A* (`pathfinding/astar.py`)
La pianificazione stradale avviene su un grafo che unisce i distretti di Seattle. Per mostrare l'efficienza della ricerca nello spazio degli stati, il codice confronta tre tecniche di potatura:

* Nessuna potatura (`NONE`): Espansione cieca di ogni stato.
* Potatura dei cicli (`CP`): Evita di riattraversare nodi già presenti nel percorso corrente.
* Potatura a cammini multipli (`MPP`): Memorizza i nodi già visitati e scarta i percorsi con costo geometrico maggiore, riducendo drasticamente il numero di nodi esplosi.

### 5. Monitoraggio in tempo reale (`bayes/belief_network.py`)
Durante il viaggio del robot lungo il percorso calcolato da A*, il sistema simula una diagnostica attiva. Utilizzando il Teorema di Bayes (inferenza esatta), aggiorna la probabilità che il robot abbia un guasto reale in base ai segnali di anomalia ricevuti dai sensori. 
Se la probabilità a posteriori di guasto supera la soglia del 70%, il pianificatore interrompe il viaggio del robot per sicurezza.

---

## Come avviare il programma

Per lanciare l'applicazione in modalità interattiva, basta posizionarsi nella cartella principale ed eseguire:

```bash
python main.py
```

L'interfaccia a riga di comando guiderà l'utente attraverso la flotta di robot, l'interrogazione delle regole Datalog, il calcolo dell'allocazione ottima tramite CSP e la simulazione di un viaggio con inserimento manuale delle anomalie.
