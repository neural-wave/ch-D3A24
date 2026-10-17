#!/usr/bin/env python3
"""
Budget Guard per fal.ai - Modulo di Sicurezza e Tracciamento Spesa
==================================================================
Modulo centralizzato per il controllo e il blocco rigido della spesa API su fal.ai.
Tiene traccia di ogni centesimo speso e impedisce qualsiasi chiamata che superi
il limite configurato (default: $3.00, modificabile via variabile d'ambiente).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Percorso predefinito del registro transazioni
SPENDING_LOG_PATH = Path("/workspaces/ch-D3A24/data/fal_spending_log.json")
DEFAULT_MAX_BUDGET = 3.00


def get_configured_max_budget() -> float:
    """Legge il limite massimo da variabile d'ambiente o fallback al default."""
    val = os.environ.get("FAL_MAX_BUDGET_USD")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return DEFAULT_MAX_BUDGET


def load_spending_log(log_path: Path = SPENDING_LOG_PATH) -> dict[str, Any]:
    """Carica lo storico della spesa dal file JSON."""
    if log_path.is_file():
        try:
            return json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "max_budget_usd": get_configured_max_budget(),
        "total_spent_usd": 0.0,
        "transactions": [],
    }


def save_spending_log(log_data: dict[str, Any], log_path: Path = SPENDING_LOG_PATH) -> None:
    """Salva lo storico della spesa su disco."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(log_data, indent=2), encoding="utf-8")


def get_total_spent(log_path: Path = SPENDING_LOG_PATH) -> float:
    """Restituisce il totale cumulativo speso finora."""
    log_data = load_spending_log(log_path)
    return float(log_data.get("total_spent_usd", 0.0))


def check_budget_or_abort(
    estimated_cost: float,
    operation_name: str,
    max_budget: float | None = None,
    log_path: Path = SPENDING_LOG_PATH,
) -> None:
    """
    Verifica se l'operazione proposta rientra nel budget.
    Se supera la soglia, interrompe immediatamente il programma impedendo la chiamata API.
    """
    if max_budget is None:
        max_budget = get_configured_max_budget()

    current_spent = get_total_spent(log_path)
    projected = current_spent + estimated_cost

    print(f"[BUDGET CHECK] Spesa attuale: ${current_spent:.4f} | Costo stimato '{operation_name}': ${estimated_cost:.4f} | Limite: ${max_budget:.2f}")

    if projected > max_budget:
        msg = (
            f"\n" + "!" * 65 + "\n"
            f"[BLOCCO DI SICUREZZA BUDGET ATTIVO]\n"
            f"L'operazione '{operation_name}' (costo stimato: ${estimated_cost:.4f}) "
            f"porterebbe la spesa a ${projected:.4f}, superando il limite rigido di ${max_budget:.2f}!\n"
            f"La chiamata a fal.ai è stata ANNULLATA in automatico.\n"
            f"Nessun credito è stato consumato.\n"
            + "!" * 65 + "\n"
        )
        sys.stderr.write(msg)
        raise SystemExit(1)


def record_spending(
    cost_usd: float,
    operation: str,
    details: dict[str, Any] | None = None,
    log_path: Path = SPENDING_LOG_PATH,
) -> float:
    """Registra una transazione di spesa effettiva nel ledger."""
    log_data = load_spending_log(log_path)
    max_budget = get_configured_max_budget()
    log_data["max_budget_usd"] = max_budget

    new_total = round(log_data.get("total_spent_usd", 0.0) + cost_usd, 4)
    log_data["total_spent_usd"] = new_total

    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "operation": operation,
        "cost_usd": cost_usd,
        "total_after_usd": new_total,
        "details": details or {},
    }
    log_data["transactions"].append(record)
    save_spending_log(log_data, log_path)

    remaining = max(0.0, max_budget - new_total)
    print(f"[BUDGET AGGIORNATO] Speso: +${cost_usd:.4f} | Totale accumulato: ${new_total:.4f} / ${max_budget:.2f} (Rimanenti: ${remaining:.4f})")
    return new_total


def print_budget_summary(log_path: Path = SPENDING_LOG_PATH) -> None:
    log_data = load_spending_log(log_path)
    max_budget = get_configured_max_budget()
    total_spent = float(log_data.get("total_spent_usd", 0.0))
    transactions = log_data.get("transactions", [])
    remaining = max(0.0, max_budget - total_spent)

    print("\n" + "=" * 60)
    print(" RIEPILOGO SPESA FAL.AI (BUDGET GUARD)")
    print("=" * 60)
    print(f"File di log     : {log_path}")
    print(f"Limite Massimo  : ${max_budget:.2f}")
    print(f"Totale Speso    : ${total_spent:.4f}")
    print(f"Budget Residuo  : ${remaining:.4f}")
    print(f"N° Transazioni  : {len(transactions)}")
    print("-" * 60)

    if transactions:
        print("Ultime transazioni:")
        for t in transactions[-5:]:
            print(f" - [{t.get('timestamp')}] {t.get('operation')}: +${t.get('cost_usd', 0.0):.4f} (Totale: ${t.get('total_after_usd', 0.0):.4f})")
    else:
        print("Nessuna transazione effettuata finora (Spesa = $0.00).")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_budget_summary()
