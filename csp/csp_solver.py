"""
csp/csp_solver.py

"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable

class Variable:
    """Rappresenta una variabile CSP con un nome e un dominio di valori possibili."""
    def __init__(self, name: str, domain: List[Any]):
        self.name = name
        self.domain = domain

    def __repr__(self) -> str:
        return f"Variable({self.name}, domain_size={len(self.domain)})"


class Constraint:
    """Rappresenta un vincolo rigido (Hard Constraint) che deve essere soddisfatto (True)."""
    def __init__(self, scope: List[str], condition: Callable[..., bool], name: str = ""):
        self.scope = scope  # Nomi delle variabili coinvolte nel vincolo
        self.condition = condition  # Funzione condizionale
        self.name = name or f"HardConstraint({', '.join(scope)})"

    def is_satisfied(self, assignment: Dict[str, Any]) -> bool:
        """Verifica se il vincolo è soddisfatto dall'assegnazione corrente."""
        # Se non tutte le variabili dello scope sono assegnate, consideriamo il vincolo provvisoriamente valido
        if not all(var in assignment for var in self.scope):
            return True
        args = [assignment[var] for var in self.scope]
        return self.condition(*args)


class SoftConstraint:
    """Rappresenta un vincolo flessibile (Soft Constraint) che attribuisce una penalità di costo."""
    def __init__(self, scope: List[str], cost_fn: Callable[..., float], name: str = ""):
        self.scope = scope  # Nomi delle variabili coinvolte nel vincolo
        self.cost_fn = cost_fn  # Funzione che restituisce il costo (float)
        self.name = name or f"SoftConstraint({', '.join(scope)})"

    def get_cost(self, assignment: Dict[str, Any]) -> float:
        """Restituisce il costo del vincolo per l'assegnazione corrente."""
        if not all(var in assignment for var in self.scope):
            return 0.0
        args = [assignment[var] for var in self.scope]
        return self.cost_fn(*args)


class CSP:
    """Modella un problema di soddisfazione dei vincoli con vincoli rigidi e flessibili."""
    def __init__(self, variables: List[Variable], hard_constraints: List[Constraint], soft_constraints: List[SoftConstraint]):
        self.variables = {v.name: v for v in variables}
        self.hard_constraints = hard_constraints
        self.soft_constraints = soft_constraints

    def is_consistent(self, var: str, value: Any, assignment: Dict[str, Any]) -> bool:
        """Verifica se l'assegnazione temporanea viola qualche vincolo rigido."""
        temp_assignment = assignment.copy()
        temp_assignment[var] = value
        
        for constraint in self.hard_constraints:
            if not constraint.is_satisfied(temp_assignment):
                return False
        return True

    def evaluate_cost(self, assignment: Dict[str, Any]) -> float:
        """Calcola la penalità totale accumulata dalle soft constraints per l'assegnazione completa."""
        total_cost = 0.0
        for constraint in self.soft_constraints:
            total_cost += constraint.get_cost(assignment)
        return round(total_cost, 3)


class CSPSolver:
    """Risolutore CSP basato su Backtracking con euristica MRV e ottimizzazione delle soft constraints."""
    def __init__(self, csp: CSP):
        self.csp = csp
        self.best_assignment: Optional[Dict[str, Any]] = None
        self.min_cost = float('inf')
        self.backtracks_count = 0

    def solve(self) -> Tuple[Optional[Dict[str, Any]], float]:
        """Avvia la ricerca dell'assegnazione ottima a costo minimo. Ritorna (assegnazione, costo)."""
        self.best_assignment = None
        self.min_cost = float('inf')
        self.backtracks_count = 0
        self._backtrack({})
        return self.best_assignment, (self.min_cost if self.best_assignment is not None else float('inf'))

    def _backtrack(self, assignment: Dict[str, Any]) -> None:
        self.backtracks_count += 1
        
        # Caso base: tutte le variabili sono state assegnate
        if len(assignment) == len(self.csp.variables):
            cost = self.csp.evaluate_cost(assignment)
            if cost < self.min_cost:
                self.min_cost = cost
                self.best_assignment = assignment.copy()
            return

        # Selezione della prossima variabile con euristica MRV (Minimum Remaining Values)
        unassigned = [v for v in self.csp.variables if v not in assignment]
        # Ordiniamo in base al numero di valori ancora validi nel dominio
        unassigned.sort(key=lambda var_name: len([
            val for val in self.csp.variables[var_name].domain 
            if self.csp.is_consistent(var_name, val, assignment)
        ]))
        
        var = unassigned[0]

        for value in self.csp.variables[var].domain:
            if self.csp.is_consistent(var, value, assignment):
                assignment[var] = value
                self._backtrack(assignment)
                del assignment[var]
