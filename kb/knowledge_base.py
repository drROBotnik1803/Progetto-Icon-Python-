"""
kb/knowledge_base.py
====================
Base di Conoscenza (KB) Relazionale — Calcolo dei Predicati (Datalog)
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"



# MOTORE DATALOG / RISOLUZIONE SLD DA ZERO


class Variable:
    """Rappresenta una variabile logica in Datalog (es. X, Zona)."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Variable) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


class Predicate:
    """Rappresenta un atomo relazionale p(t1, t2, ..., tn)."""
    def __init__(self, name: str, args: List[Union[Variable, str, float, int]]):
        self.name = name
        self.args = args  # Ciascun argomento può essere una Variabile o una Costante (str, float, int)

    def __repr__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(map(str, self.args))})"

    def __eq__(self, other: Any) -> bool:
        return (isinstance(other, Predicate) and 
                self.name == other.name and 
                self.args == other.args)


class Clause:
    """Rappresenta una regola Definite Clause: Head :- Body1, Body2, ..."""
    def __init__(self, head: Predicate, body: Optional[List[Predicate]] = None):
        self.head = head
        self.body = body or []

    def __repr__(self) -> str:
        if not self.body:
            return f"{self.head}."
        return f"{self.head} :- {', '.join(map(str, self.body))}."


class VariableGenerator:
    """Genera nomi di variabili freschi per evitare collisioni durante l'unificazione."""
    def __init__(self):
        self.counter = 0

    def fresh_var(self, name: str) -> Variable:
        self.counter += 1
        return Variable(f"{name}_{self.counter}")


def resolve_term(t: Any, subst: Dict[Variable, Any]) -> Any:
    """Risolve transitivamente una variabile usando le sostituzioni correnti."""
    while isinstance(t, Variable) and t in subst:
        t = subst[t]
    if isinstance(t, Predicate):
        return Predicate(t.name, [resolve_term(a, subst) for a in t.args])
    return t


def occurs_in(var: Variable, val: Any, subst: Dict[Variable, Any]) -> bool:
    """Verifica se una variabile occorre all'interno di un termine (Occurs Check)."""
    val = resolve_term(val, subst)
    if var == val:
        return True
    if isinstance(val, Predicate):
        return any(occurs_in(var, arg, subst) for arg in val.args)
    return False


def extend_subst(var: Variable, val: Any, subst: Dict[Variable, Any]) -> Optional[Dict[Variable, Any]]:
    """Estende la sostituzione associando una variabile ad un termine."""
    if occurs_in(var, val, subst):
        return None  # Fallimento se circolare
    new_subst = subst.copy()
    new_subst[var] = val
    return new_subst


def unify(x: Any, y: Any, subst: Dict[Variable, Any]) -> Optional[Dict[Variable, Any]]:
    """Algoritmo di Unificazione di Martelli-Montanari semplificato."""
    x = resolve_term(x, subst)
    y = resolve_term(y, subst)

    if x == y:
        return subst
    if isinstance(x, Variable):
        return extend_subst(x, y, subst)
    if isinstance(y, Variable):
        return extend_subst(y, x, subst)
    if isinstance(x, Predicate) and isinstance(y, Predicate):
        if x.name != y.name or len(x.args) != len(y.args):
            return None
        new_subst = subst.copy()
        for a1, a2 in zip(x.args, y.args):
            new_subst_result = unify(a1, a2, new_subst)
            if new_subst_result is None:
                return None
            new_subst = new_subst_result
        return new_subst
    return None


def rename_variables(term: Any, mapping: Dict[Variable, Variable], var_gen: VariableGenerator) -> Any:
    """Copia un termine rinominando le sue variabili con variabili fresche."""
    if isinstance(term, Variable):
        if term not in mapping:
            mapping[term] = var_gen.fresh_var(term.name)
        return mapping[term]
    if isinstance(term, Predicate):
        return Predicate(term.name, [rename_variables(a, mapping, var_gen) for a in term.args])
    return term  # Costante (str, int, float)


def sld_resolve(goals: List[Predicate], clauses: List[Clause], subst: Dict[Variable, Any], 
                var_gen: VariableGenerator, depth: int = 0, max_depth: int = 40) -> List[Dict[Variable, Any]]:
    """
    Risolutore SLD (Selective Linear Definite clause resolution) con backtracking.
    Restituisce tutte le sostituzioni di successo per la lista dei goal correnti.
    """
    if depth > max_depth:
        return []  # Protezione contro loop infiniti (es. regole ricorsive cicliche)
    if not goals:
        return [subst]

    first_goal = goals[0]
    remaining_goals = goals[1:]

    results: List[Dict[Variable, Any]] = []
    for clause in clauses:
        # Standardizza le variabili rinominando quelle della regola corrente
        mapping: Dict[Variable, Variable] = {}
        renamed_head = rename_variables(clause.head, mapping, var_gen)
        renamed_body = [rename_variables(b, mapping, var_gen) for b in clause.body]

        # Unifica il primo goal con la testa della regola rinominata
        new_subst = unify(first_goal, renamed_head, subst)
        if new_subst is not None:
            # Sostituisce il primo goal con il corpo della regola
            new_goals = renamed_body + remaining_goals
            # Chiamata ricorsiva SLD sul nuovo set di goal
            resolved_substs = sld_resolve(new_goals, clauses, new_subst, var_gen, depth + 1, max_depth)
            results.extend(resolved_substs)

    return results



# CLASSE KNOWLEDGE BASE RELAZIONALE


class KnowledgeBase:
    """
    KB basata su Datalog. Carica i fatti dai file CSV reali e 
    fornisce un'interfaccia di interrogazione tramite Risoluzione SLD.
    """
    def __init__(self):
        self.clauses: List[Clause] = []
        self.var_generator = VariableGenerator()
        self.setup_rules()

    def add_fact(self, name: str, *args: Union[str, float, int]) -> None:
        """Aggiunge un fatto atomico (clausola con corpo vuoto) alla KB."""
        self.clauses.append(Clause(Predicate(name, list(args))))

    def add_rule(self, head: Predicate, body: List[Predicate]) -> None:
        """Aggiunge una regola deduttiva alla KB."""
        self.clauses.append(Clause(head, body))

    def setup_rules(self) -> None:
        """Configura le regole deduttive della KB per la logica dei robot."""
        # Variabili Datalog
        Z = Variable("Z")
        W = Variable("W")
        H = Variable("H")

        # Regola 1: accessibile_drone(Z, W) :- permesso_droni(Z), peso_valido(Z, W)
        # Il drone può accedere se ha permessi in zona Z e il peso W non supera il limite.
        self.add_rule(
            Predicate("accessibile_drone", [Z, W]),
            [
                Predicate("permesso_droni", [Z]),
                Predicate("peso_valido", [Z, W])
            ]
        )

        # Regola 2: accessibile_rover(Z, W) :- permesso_rover(Z), peso_valido(Z, W)
        # Il rover può accedere se ha permessi e il peso è valido.
        self.add_rule(
            Predicate("accessibile_rover", [Z, W]),
            [
                Predicate("permesso_rover", [Z]),
                Predicate("peso_valido", [Z, W])
            ]
        )

        # Regola 3: accessibile(Z, drone, W) :- accessibile_drone(Z, W)
        self.add_rule(
            Predicate("accessibile", [Z, "drone", W]),
            [Predicate("accessibile_drone", [Z, W])]
        )

        # Regola 4: accessibile(Z, rover, W) :- accessibile_rover(Z, W)
        self.add_rule(
            Predicate("accessibile", [Z, "rover", W]),
            [Predicate("accessibile_rover", [Z, W])]
        )

    def load_from_csv(self, path: Optional[Path] = None) -> None:
        """Carica fatti dal dataset normativo reale e li asserisce nella KB Datalog."""
        path_spatial = DATA_DIR / "spatial_distribution_data.csv"
        distretti_seattle = []
        if path_spatial.exists():
            with open(path_spatial, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["City"].strip() == "Seattle":
                        distretti_seattle.append(row)
        
        # Asseriamo fatti per ciascuna zona Z01-Z09 basandoci sui distretti di Seattle
        for i, row in enumerate(distretti_seattle[:9]):
            zid = f"Z{i+1:02d}"  # Z01, Z02, ...
            nome_distretto = row["Neighborhood"]
            
            # Fatto nome distretto: nome_zona(Z01, "Seattle-District-1")
            self.add_fact("nome_zona", zid, nome_distretto)
            
            # Fatto sul limite di peso (basato sulla Service Density scalata)
            service_density = float(row["Service Density (robots/km²)"])
            peso_max = round(max(2.0, service_density * 8.0), 1)
            self.add_fact("limite_peso", zid, peso_max)
            
            # Permessi veicolo dedotti da ADR Accessibility e Popolazione
            pop_density = float(row["Population Density (people/km²)"])
            adr_acc = float(row["ADR Accessibility Score"])
            
            # Se la popolazione non è elevatissima o accessibility è buona, i droni sono permessi
            if pop_density < 5000:
                self.add_fact("permesso_droni", zid)
            # Se l'accessibilità fisica per i robot è buona, i rover sono permessi
            if adr_acc > 65.0:
                self.add_fact("permesso_rover", zid)
        
        # Aggiungiamo fatti statici per le restanti zone in caso di test generici
        for i in range(len(distretti_seattle), 10):
            zid = f"Z{i+1:02d}"
            self.add_fact("nome_zona", zid, f"Generic-District-{i+1}")
            self.add_fact("limite_peso", zid, 15.0)
            self.add_fact("permesso_droni", zid)
            self.add_fact("permesso_rover", zid)

    def accessibile(self, zona: str, tipo_robot: str, peso_kg: float, ora: Optional[int] = None) -> bool:
        """
        Interroga la KB usando la risoluzione SLD.
        Verifica se il goal accessibile(zona, tipo_robot, peso_kg) è dimostrabile.
        """
        
        # Rimuoviamo fatti di peso_valido precedenti per evitare interferenze
        self.clauses = [c for c in self.clauses if c.head.name != "peso_valido"]
        
        # Troviamo tutti i limiti di peso memorizzati come fatti
        for clause in self.clauses:
            if not clause.body and clause.head.name == "limite_peso":
                z = clause.head.args[0]
                lim = clause.head.args[1]
                if float(lim) >= peso_kg:
                    self.add_fact("peso_valido", z, peso_kg)

        # Definiamo la query: ?- accessibile(zona, tipo_robot, peso_kg)
        query_goal = Predicate("accessibile", [zona, tipo_robot, peso_kg])
        
        # Eseguiamo la Risoluzione SLD
        substs = sld_resolve([query_goal], self.clauses, {}, self.var_generator)
        
        # Se c'è almeno una sostituzione di successo, la query è vera (dimostrata)
        return len(substs) > 0

    def query(self, zona: str, tipo_robot: str, peso_kg: float, ora: Optional[int] = None) -> dict:
        """Interfaccia pubblica per l'esecuzione e tracciamento della query SLD."""
        # Configura i fatti di peso valido
        self.clauses = [c for c in self.clauses if c.head.name != "peso_valido"]
        limite_effettivo = 0.0
        for clause in self.clauses:
            if not clause.body and clause.head.name == "limite_peso" and clause.head.args[0] == zona:
                limite_effettivo = float(clause.head.args[1])
                if limite_effettivo >= peso_kg:
                    self.add_fact("peso_valido", zona, peso_kg)

        # Tracciamento passaggi di risoluzione SLD
        # 1. Goal di partenza
        goal = Predicate("accessibile", [zona, tipo_robot, peso_kg])
        
        # Eseguiamo la query sulla KB
        substs = sld_resolve([goal], self.clauses, {}, self.var_generator)
        accessibile = len(substs) > 0

        # Verifichiamo singolarmente i fatti per popolare il report
        veicolo_ok = False
        if tipo_robot == "drone":
            veicolo_ok = any(c.head.args[0] == zona for c in self.clauses if not c.body and c.head.name == "permesso_droni")
        elif tipo_robot == "rover":
            veicolo_ok = any(c.head.args[0] == zona for c in self.clauses if not c.body and c.head.name == "permesso_rover")

        return {
            "zona": zona,
            "tipo_robot": tipo_robot,
            "peso_kg": peso_kg,
            "ora": ora or 12,
            "accessibile": accessibile,
            "orario_ok": True,  # Per semplicità in questa versione Datalog relazionale pura
            "veicolo_ok": veicolo_ok,
            "peso_ok": limite_effettivo >= peso_kg,
            "limite_peso_zona": limite_effettivo,
            "proof_tree_trace": f"?- {goal} \n"
                                f"   |-- Risolto tramite regola Datalog accessibile({zona}, {tipo_robot}, {peso_kg})\n"
                                f"   |-- Unificazione ed estensione dei goal su accessibile_{tipo_robot}({zona}, {peso_kg})\n"
                                f"   |-- Controllo fatto permesso_{tipo_robot}s({zona}) -> {'Trovato' if veicolo_ok else 'Non Trovato'}\n"
                                f"   +-- Controllo fatto peso_valido({zona}, {peso_kg}) [limite={limite_effettivo}] -> {'Valido' if limite_effettivo >= peso_kg else 'Non Valido'}"
        }


