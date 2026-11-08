"""src/agent/refund_agent.py
Autonomous ReAct refund agent for Instamart customer support.
Implements five domain tools, tool mapping, ReAct execution loop,
safe stop conditions (iterations, token budget, wall-clock timeout),
and resumable execution with persistent checkpointing.

Strengthened with tool-boundary validation guards and transaction idempotency.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mock_llm
from src.rag.retriever import PolicyRetriever

# Default paths
DEFAULT_LEDGER_PATH: Path = PROJECT_ROOT / "outputs" / "refund_ledger.jsonl"
DEFAULT_DB_PATH: Path = PROJECT_ROOT / "data" / "instamart.db"

# ---------------------------------------------------------------------------
# Global / Lazy Retriever for search_policy
# ---------------------------------------------------------------------------
_default_retriever: PolicyRetriever | None = None


def get_policy_retriever() -> PolicyRetriever:
    """Lazily load and cache the TF-IDF PolicyRetriever."""
    global _default_retriever
    if _default_retriever is None:
        policies_dir = PROJECT_ROOT / "data" / "policies"
        _default_retriever = PolicyRetriever(policies_dir=policies_dir, chunk_size=40, overlap=10)
    return _default_retriever


# ---------------------------------------------------------------------------
# TASK 5: Tool Implementations with Tool-Boundary Security Guards
# ---------------------------------------------------------------------------
def get_order(order_id: str, db_path: str | Path | None = None) -> dict[str, Any]:
    """Query data/instamart.db for order details.

    Returns:
        dict with order_id, status, order_amount, delivery_minutes,
        or {"error": "order not found"} if missing.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        )
        row = cursor.fetchone()
        if not row:
            return {"error": "order not found"}
        return {
            "order_id": str(row[0]),
            "status": str(row[1]),
            "order_amount": float(row[2]) if row[2] is not None else None,
            "delivery_minutes": int(row[3]) if row[3] is not None else None,
        }
    finally:
        conn.close()


def search_policy(query: str) -> dict[str, Any]:
    """Reuse the existing Part D TF-IDF retriever to find relevant policy passages.

    Returns:
        dict with key "results" containing top-k retrieved chunks.
    """
    retriever = get_policy_retriever()
    results = retriever.retrieve(query, k=3)
    return {"results": results}


def calculate_refund(
    order_amount: float,
    delivery_minutes: float | int | None,
) -> dict[str, Any]:
    """Calculate refund amount in INR based on late delivery rules.

    Boundary Validation:
      - Validates order_amount is a valid non-negative, finite number (rejects NaN, +/-Inf, OverflowError).
      - Safely rejects non-numeric, NaN, infinite, negative, overflow, or missing delivery_minutes.
      - Returns 0.0 when delivery time is <= 45 minutes or invalid.
      - Calculates 100% of order_amount when delivery_minutes > 45.
      - Explicitly marks requires_escalation = True for amounts > INR 1000.
      - Pure calculation tool: does NOT automatically issue any refunds.

    Returns:
        {"refund_inr": amount, "eligible": bool, "requires_escalation": bool}
    """
    if order_amount is None:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Order amount is missing",
        }

    try:
        order_amount_val = float(order_amount)
    except (TypeError, ValueError, OverflowError):
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Invalid order amount format or numeric overflow",
        }

    # Reject NaN and +/- Infinity
    if math.isnan(order_amount_val) or math.isinf(order_amount_val):
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Order amount cannot be NaN or infinite",
        }

    if order_amount_val < 0.0:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Order amount cannot be negative",
        }

    # Validate delivery_minutes
    if delivery_minutes is None:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Delivery minutes is missing",
        }

    try:
        delivery_minutes_val = float(delivery_minutes)
    except (TypeError, ValueError, OverflowError):
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Delivery minutes must be numeric and finite",
        }

    if math.isnan(delivery_minutes_val) or math.isinf(delivery_minutes_val):
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Delivery minutes cannot be NaN or infinite",
        }

    if delivery_minutes_val < 0.0:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
            "error": "Delivery minutes cannot be negative",
        }

    if delivery_minutes_val <= 45.0:
        return {
            "refund_inr": 0.0,
            "eligible": False,
            "requires_escalation": False,
        }

    refund = round(order_amount_val, 2)
    requires_escalation = refund > 1000.0

    return {
        "refund_inr": refund,
        "eligible": True,
        "requires_escalation": requires_escalation,
    }


def issue_refund(
    order_id: str,
    refund_inr: float,
    ledger_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Issue an automated refund with independent database validation and idempotency protection.

    Tool-Boundary Guardrails:
      1. Validates refund_inr is a positive finite numeric amount.
      2. Rejects any refund > INR 1000.00 (must be escalated to human).
      3. Independently queries data/instamart.db to verify order exists.
      4. Rejects orders whose status != 'completed'.
      5. Rejects orders whose delivery_minutes <= 45.
      6. Verifies requested refund_inr matches the actual order_amount within INR 0.01.
      7. Idempotency Safeguard: Scans existing ledger to ensure no duplicate disbursement is written.
         (Note: This check-before-append safeguard is designed for single-process offline demonstration;
          production environments require transactional database constraints or table locks).

    Returns:
        {"refund_inr": amount, "status": "issued"} on success,
        {"refund_inr": amount, "status": "already_issued", "idempotent": True} if already in ledger,
        or {"error": "<reason>", "refund_inr": 0.0, "status": "rejected"} on validation failure.
    """
    try:
        refund_inr_val = float(refund_inr)
    except (TypeError, ValueError, OverflowError):
        return {"error": "Invalid refund amount format or numeric overflow", "refund_inr": 0.0, "status": "rejected"}

    if math.isnan(refund_inr_val) or math.isinf(refund_inr_val):
        return {"error": "Refund amount cannot be NaN or infinite", "refund_inr": 0.0, "status": "rejected"}

    if refund_inr_val <= 0:
        return {"error": "Refund amount must be strictly positive", "refund_inr": 0.0, "status": "rejected"}

    refund_inr_val = round(refund_inr_val, 2)

    # Automatic issuance ceiling
    if refund_inr_val > 1000.0:
        return {
            "error": f"Refund amount INR {refund_inr_val:.2f} exceeds automatic limit of INR 1000; must be escalated to human lead",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    # Independent database verification
    order_info = get_order(order_id, db_path=db_path)
    if "error" in order_info:
        return {"error": f"Order {order_id} not found in database", "refund_inr": 0.0, "status": "rejected"}

    status = order_info.get("status")
    if status != "completed":
        return {
            "error": f"Order {order_id} has status '{status}'; only completed orders can be refunded",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    delivery_min = order_info.get("delivery_minutes")
    if delivery_min is None or delivery_min <= 45:
        return {
            "error": f"Order {order_id} delivery time was {delivery_min} minutes (eligible only if > 45 minutes)",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    order_amount = order_info.get("order_amount")
    if order_amount is None or not math.isclose(refund_inr_val, order_amount, abs_tol=0.01):
        return {
            "error": f"Requested refund INR {refund_inr_val:.2f} does not match eligible order amount INR {order_amount}",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    # Idempotency Safeguard
    if ledger_path is None:
        ledger_path = DEFAULT_LEDGER_PATH
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("order_id") == order_id and record.get("tool") == "issue_refund":
                        return {
                            "refund_inr": refund_inr_val,
                            "status": "already_issued",
                            "idempotent": True,
                        }
                except json.JSONDecodeError:
                    continue

    # Commit write to ledger
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "issue_refund",
        "order_id": str(order_id),
        "refund_inr": refund_inr_val,
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {"refund_inr": refund_inr_val, "status": "issued"}


def escalate_to_human(
    order_id: str,
    refund_inr: float,
    ledger_path: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Escalate a high-value refund (> INR 1000) with independent database verification.

    Tool-Boundary Guardrails:
      1. Validates refund_inr is a positive finite numeric amount > INR 1000.00.
      2. Independently queries data/instamart.db to verify order exists.
      3. Rejects orders whose status != 'completed'.
      4. Rejects orders whose delivery_minutes <= 45.
      5. Verifies refund_inr matches the actual order_amount within INR 0.01.
      6. Idempotency Safeguard: Scans existing ledger to ensure no duplicate escalation is written.
         (Note: This check-before-append safeguard is designed for single-process offline demonstration;
          production environments require transactional database constraints or table locks).

    Returns:
        {"refund_inr": amount, "status": "escalated"} on success,
        {"refund_inr": amount, "status": "already_escalated", "idempotent": True} if already in ledger,
        or {"error": "<reason>", "refund_inr": 0.0, "status": "rejected"} on validation failure.
    """
    try:
        refund_inr_val = float(refund_inr)
    except (TypeError, ValueError, OverflowError):
        return {"error": "Invalid refund amount format or numeric overflow", "refund_inr": 0.0, "status": "rejected"}

    if math.isnan(refund_inr_val) or math.isinf(refund_inr_val):
        return {"error": "Refund amount cannot be NaN or infinite", "refund_inr": 0.0, "status": "rejected"}

    refund_inr_val = round(refund_inr_val, 2)

    if refund_inr_val <= 1000.0:
        return {
            "error": f"Refund amount INR {refund_inr_val:.2f} does not exceed INR 1000; escalation is reserved for high-value claims",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    # Independent database verification
    order_info = get_order(order_id, db_path=db_path)
    if "error" in order_info:
        return {"error": f"Order {order_id} not found in database", "refund_inr": 0.0, "status": "rejected"}

    status = order_info.get("status")
    if status != "completed":
        return {
            "error": f"Order {order_id} has status '{status}'; only completed orders can be escalated for refund",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    delivery_min = order_info.get("delivery_minutes")
    if delivery_min is None or delivery_min <= 45:
        return {
            "error": f"Order {order_id} delivery time was {delivery_min} minutes (eligible only if > 45 minutes)",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    order_amount = order_info.get("order_amount")
    if order_amount is None or not math.isclose(refund_inr_val, order_amount, abs_tol=0.01):
        return {
            "error": f"Requested refund INR {refund_inr_val:.2f} does not match eligible order amount INR {order_amount}",
            "refund_inr": 0.0,
            "status": "rejected",
        }

    # Idempotency Safeguard
    if ledger_path is None:
        ledger_path = DEFAULT_LEDGER_PATH
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    if ledger_path.exists():
        with open(ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if record.get("order_id") == order_id and record.get("tool") == "escalate_to_human":
                        return {
                            "refund_inr": refund_inr_val,
                            "status": "already_escalated",
                            "idempotent": True,
                        }
                except json.JSONDecodeError:
                    continue

    # Commit write to ledger
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": "escalate_to_human",
        "order_id": str(order_id),
        "refund_inr": refund_inr_val,
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {"refund_inr": refund_inr_val, "status": "escalated"}


# ---------------------------------------------------------------------------
# TOOLS_SPEC & TOOL_MAP
# ---------------------------------------------------------------------------
TOOLS_SPEC: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Look up order status, order amount, and delivery time from the Instamart database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique identifier of the order, e.g. ORD-100048.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search Instamart policy documents using TF-IDF cosine similarity retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The policy search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund amount in INR based on order delivery time: refunds 100% of order_amount if delivery_minutes exceeds 45, otherwise INR 0.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {
                        "type": "number",
                        "description": "The total order amount in INR.",
                    },
                    "delivery_minutes": {
                        "type": "number",
                        "description": "The delivery time in minutes.",
                    },
                },
                "required": ["order_amount", "delivery_minutes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Issue an automated refund for eligible late delivery and append to the refund ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID receiving the refund.",
                    },
                    "refund_inr": {
                        "type": "number",
                        "description": "The refund amount in INR to be issued.",
                    },
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Escalate high-value refund requests (> INR 1000) to a human support lead and record in the ledger.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID being escalated.",
                    },
                    "refund_inr": {
                        "type": "number",
                        "description": "The requested refund amount in INR requiring human authorization.",
                    },
                },
                "required": ["order_id", "refund_inr"],
            },
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ---------------------------------------------------------------------------
# ReAct Execution Loop & Safe Stop Logic
# ---------------------------------------------------------------------------
def _execute_agent_loop(
    scenario_id: str,
    run_id: str,
    messages: list[dict[str, Any]],
    llm_calls: int,
    tokens_used: int,
    steps: list[dict[str, Any]],
    tools_executed: list[str],
    max_iterations: int,
    token_limit: int,
    time_limit_seconds: float,
    start_time: float,
    checkpoint_file: Path,
    simulate_crash_after_tool: int | None = None,
) -> dict[str, Any]:
    """Execute or continue the ReAct loop across iterations."""
    for iteration in range(llm_calls + 1, max_iterations + 1):
        # Stop Condition C: Time limit checked at top of iteration
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TIMEOUT",
                "final_answer": f"Execution timeout exceeded ({elapsed:.2f}s >= {time_limit_seconds}s). Handoff to human support lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        # Model invocation
        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += response.usage.total_tokens

        # Stop Condition B: Token limit checked immediately after model call
        if tokens_used > token_limit:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_TOKEN_LIMIT",
                "final_answer": f"Token budget exceeded ({tokens_used} > {token_limit}). Handoff to human support lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        choice = response.choices[0]
        msg = choice.message

        # Final answer received (no tool calls)
        if not msg.tool_calls:
            final_answer = msg.content or ""
            status = final_answer.split(":")[0].strip() if ":" in final_answer else final_answer.strip()
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        # Stop Condition A: Maximum iterations reached without final answer
        if iteration == max_iterations:
            return {
                "scenario_id": scenario_id,
                "run_id": run_id,
                "status": "FALLBACK_MAX_ITERATIONS",
                "final_answer": f"Maximum iterations reached ({max_iterations} calls without resolution). Handoff to human support lead.",
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used,
            }

        # Tool calls present: append assistant message
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in msg.tool_calls
            ],
        }
        messages.append(assistant_msg)

        # Execute requested tools using TOOL_MAP
        for tc in msg.tool_calls:
            tool_name = tc.name
            tool_args = tc.arguments
            if tool_name not in TOOL_MAP:
                raise ValueError(f"Unknown tool requested: {tool_name}")

            tool_fn = TOOL_MAP[tool_name]
            observation = tool_fn(**tool_args)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": json.dumps(observation),
            }
            messages.append(tool_msg)

            step_record = {
                "step": len(steps) + 1,
                "tool": tool_name,
                "arguments": tool_args,
                "observation": observation,
            }
            steps.append(step_record)
            tools_executed.append(tool_name)

            # Checkpoint saved after EVERY tool observation is appended
            checkpoint_data = {
                "run_id": run_id,
                "scenario_id": scenario_id,
                "messages": messages,
                "tokens_used": tokens_used,
                "llm_calls": llm_calls,
                "step_number": len(steps),
                "tools_already_executed": list(tools_executed),
                "steps": steps,
                "elapsed_before_pause": time.monotonic() - start_time,
            }
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint_data, f, indent=2)

            # Simulated crash condition
            if simulate_crash_after_tool is not None and len(tools_executed) == simulate_crash_after_tool:
                raise RuntimeError(
                    f"Simulated crash triggered after {simulate_crash_after_tool} tool executions ({tools_executed})."
                )

    # Catch-all if max_iterations exceeded
    return {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "status": "FALLBACK_MAX_ITERATIONS",
        "final_answer": "Maximum iterations reached. Handoff to human support lead.",
        "steps": steps,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
    }


def run_agent(
    user_message: str,
    scenario_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    simulate_crash_after_tool: int | None = None,
    run_id: str | None = None,
    state_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run the ReAct agent on a customer message.

    Returns:
        dict with status, final_answer, steps, llm_calls, tokens_used.
    """
    if run_id is None:
        run_id = f"{scenario_id}_{int(time.time() * 1000)}"

    if state_dir is None:
        state_dir = PROJECT_ROOT / "state"
    state_path = Path(state_dir)
    checkpoint_file = state_path / f"{run_id}.json"

    messages = [
        {"role": "system", "content": "You are the Instamart AGENT for customer support and refunds."},
        {"role": "user", "content": user_message},
    ]

    llm_calls = 0
    tokens_used = 0
    steps: list[dict[str, Any]] = []
    tools_executed: list[str] = []
    start_time = time.monotonic()

    return _execute_agent_loop(
        scenario_id=scenario_id,
        run_id=run_id,
        messages=messages,
        llm_calls=llm_calls,
        tokens_used=tokens_used,
        steps=steps,
        tools_executed=tools_executed,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        start_time=start_time,
        checkpoint_file=checkpoint_file,
        simulate_crash_after_tool=simulate_crash_after_tool,
    )


def resume_agent(
    run_id: str,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10,
    state_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Resume execution of an interrupted agent run from its state checkpoint file.

    Loads the checkpoint, restores messages, step counter, tokens, and continues.
    """
    if state_dir is None:
        state_dir = PROJECT_ROOT / "state"
    checkpoint_file = Path(state_dir) / f"{run_id}.json"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    with open(checkpoint_file, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    scenario_id = checkpoint["scenario_id"]
    messages = checkpoint["messages"]
    tokens_used = checkpoint["tokens_used"]
    llm_calls = checkpoint["llm_calls"]
    steps = checkpoint["steps"]
    tools_executed = list(checkpoint["tools_already_executed"])
    start_time = time.monotonic() - checkpoint.get("elapsed_before_pause", 0.0)

    return _execute_agent_loop(
        scenario_id=scenario_id,
        run_id=run_id,
        messages=messages,
        llm_calls=llm_calls,
        tokens_used=tokens_used,
        steps=steps,
        tools_executed=tools_executed,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        start_time=start_time,
        checkpoint_file=checkpoint_file,
    )
