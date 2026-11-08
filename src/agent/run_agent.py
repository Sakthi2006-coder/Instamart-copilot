"""src/agent/run_agent.py
Driver and validation script for Part D autonomous refund agent.
Runs all 7 benchmark scenarios, validates results against reference figures,
executes safe-stop demonstrations (token limit, wall-clock timeout),
demonstrates checkpointing with simulated crash recovery,
verifies financial idempotency under crash replay,
and runs focused tool-boundary safety validation tests.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import mock_llm
import src.agent.refund_agent as refund_agent_mod
from src.agent.refund_agent import (
    TOOLS_SPEC,
    TOOL_MAP,
    DEFAULT_LEDGER_PATH,
    get_order,
    search_policy,
    calculate_refund,
    issue_refund,
    escalate_to_human,
    run_agent,
    resume_agent,
)


def get_step_refund_amount(steps: list[dict]) -> float:
    """Extract final executed refund amount from steps observation."""
    for step in reversed(steps):
        if step["tool"] in ("issue_refund", "escalate_to_human"):
            obs = step.get("observation", {})
            if "refund_inr" in obs:
                return round(float(obs["refund_inr"]), 2)
    return 0.0


def main():
    print("=" * 85)
    print("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT")
    print("PART D: AUTONOMOUS REFUND AGENT EVALUATION & SAFETY VERIFICATION")
    print("=" * 85)

    scenarios_path = PROJECT_ROOT / "data" / "raw" / "agent_scenarios.csv"
    runs_log_path = PROJECT_ROOT / "outputs" / "agent_runs.jsonl"
    ledger_path = PROJECT_ROOT / "outputs" / "refund_ledger.jsonl"
    report_path = PROJECT_ROOT / "outputs" / "agent_evaluation_report.txt"
    state_dir = PROJECT_ROOT / "state"

    runs_log_path.parent.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    # Clean existing ledger and runs log before benchmark run
    if runs_log_path.exists():
        runs_log_path.unlink()
    if ledger_path.exists():
        ledger_path.unlink()

    scenarios_df = pd.read_csv(scenarios_path)
    print(f"Loaded {len(scenarios_df)} benchmark scenarios from {scenarios_path.name}\n")

    scenario_results = []
    print("-" * 85)
    print("1. RUNNING SEVEN BENCHMARK SCENARIOS (max_iterations=6, token_limit=6000, time_limit=10s)")
    print("-" * 85)
    print(f"{'ID':<4} {'Expected Status':<15} {'Actual Status':<25} {'Refund (INR)':<14} {'Calls':<7} {'Tokens':<8} {'Status'}")
    print("-" * 85)

    for _, row in scenarios_df.iterrows():
        s_id = str(row["scenario_id"])
        c_id = str(row["customer_id"])
        user_msg = str(row["message"])
        exp_outcome = str(row["expected_outcome"])
        exp_refund = float(row["expected_refund_inr"])

        mock_llm.reset()
        run_res = run_agent(
            user_message=user_msg,
            scenario_id=s_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10.0,
            run_id=f"benchmark_{s_id}",
            state_dir=state_dir,
        )

        status = run_res["status"]
        final_answer = run_res["final_answer"]
        llm_calls = run_res["llm_calls"]
        tokens_used = run_res["tokens_used"]
        steps = run_res["steps"]

        refund_amt = get_step_refund_amount(steps)

        # Write to outputs/agent_runs.jsonl
        log_entry = {
            "scenario_id": s_id,
            "status": status,
            "final_answer": final_answer,
            "refund_inr": refund_amt,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used,
            "steps": steps,
        }
        with open(runs_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Verify expectations
        if s_id == "S7":
            status_match = (status == "FALLBACK_MAX_ITERATIONS")
        else:
            status_match = (status == exp_outcome)

        refund_match = math.isclose(refund_amt, exp_refund, abs_tol=0.01)
        passed = status_match and refund_match

        if s_id == "S1":
            assert llm_calls == 5, f"S1 expected exactly 5 model calls, got {llm_calls}"
        if s_id == "S7":
            assert llm_calls == 6, f"S7 expected exactly 6 model calls, got {llm_calls}"
            assert 2700 <= tokens_used <= 5500, f"S7 expected tokens in [2700, 5500], got {tokens_used}"

        scenario_results.append({
            "scenario_id": s_id,
            "expected_outcome": exp_outcome,
            "status": status,
            "refund_inr": refund_amt,
            "expected_refund_inr": exp_refund,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used,
            "passed": passed,
            "final_answer": final_answer,
        })

        verdict = "PASS" if passed else "FAIL"
        print(f"{s_id:<4} {exp_outcome:<15} {status:<25} {refund_amt:<14.2f} {llm_calls:<7} {tokens_used:<8} {verdict}")

    # Mandatory Quality-Control Assertion immediately after scenario evaluation loop
    assert all(r["passed"] for r in scenario_results), (
        "One or more benchmark scenarios failed"
    )

    print("-" * 85)

    # ---------------------------------------------------------------------------
    # TASK 7: Safe Stop Demonstrations
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("2. TASK 7: SAFE STOP CONDITION DEMONSTRATIONS")
    print("-" * 85)

    # Demonstration 7B: S1 with token_limit=200
    mock_llm.reset()
    s1_msg = scenarios_df[scenarios_df["scenario_id"] == "S1"]["message"].values[0]
    s1_token_demo = run_agent(
        user_message=s1_msg,
        scenario_id="S1_token_demo",
        max_iterations=6,
        token_limit=200,
        time_limit_seconds=10.0,
        run_id="demo_s1_token_limit_200",
        state_dir=state_dir,
    )
    print(f"  [7B] Token Limit Demo (S1, token_limit=200):")
    print(f"       - Status       : {s1_token_demo['status']}")
    print(f"       - LLM Calls    : {s1_token_demo['llm_calls']}")
    print(f"       - Tokens Used  : {s1_token_demo['tokens_used']} (Exceeded 200 limit)")
    print(f"       - Steps Taken  : {len(s1_token_demo['steps'])} (Tool requested at step 3 halted without execution)")
    print(f"       - Final Answer : {s1_token_demo['final_answer']}")
    assert s1_token_demo["status"] == "FALLBACK_TOKEN_LIMIT", f"Expected FALLBACK_TOKEN_LIMIT, got {s1_token_demo['status']}"
    assert len(s1_token_demo["steps"]) == 2, f"Expected 2 completed steps before token stop, got {len(s1_token_demo['steps'])}"
    print("       -> Result: PASS (Halted immediately when cumulative tokens exceeded 200)")

    # Demonstration 7C: S1 with time_limit_seconds=0
    mock_llm.reset()
    s1_timeout_demo = run_agent(
        user_message=s1_msg,
        scenario_id="S1_timeout_demo",
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=0.0,
        run_id="demo_s1_timeout_zero",
        state_dir=state_dir,
    )
    print(f"\n  [7C] Wall-Clock Timeout Demo (S1, time_limit_seconds=0):")
    print(f"       - Status       : {s1_timeout_demo['status']}")
    print(f"       - LLM Calls    : {s1_timeout_demo['llm_calls']} (Stopped before call 1)")
    print(f"       - Tokens Used  : {s1_timeout_demo['tokens_used']}")
    print(f"       - Steps Taken  : {len(s1_timeout_demo['steps'])}")
    print(f"       - Final Answer : {s1_timeout_demo['final_answer']}")
    assert s1_timeout_demo["status"] == "FALLBACK_TIMEOUT", f"Expected FALLBACK_TIMEOUT, got {s1_timeout_demo['status']}"
    assert s1_timeout_demo["llm_calls"] == 0, f"Expected 0 calls, got {s1_timeout_demo['llm_calls']}"
    print("       -> Result: PASS (Stopped at top of loop before first model call)")

    # Demonstration 7A: S7 Max Iterations Check
    s7_res = next(r for r in scenario_results if r["scenario_id"] == "S7")
    print(f"\n  [7A] Max Iterations Demo (S7, max_iterations=6):")
    print(f"       - Status       : {s7_res['status']}")
    print(f"       - LLM Calls    : {s7_res['llm_calls']} (Completed all 6 calls under 6000 token limit)")
    print(f"       - Tokens Used  : {s7_res['tokens_used']}")
    print(f"       - Final Answer : {s7_res['final_answer']}")
    assert s7_res["status"] == "FALLBACK_MAX_ITERATIONS"
    print("       -> Result: PASS (Clean fallback triggered after 6 iterations without resolution)")

    # ---------------------------------------------------------------------------
    # TASK 8: Checkpointing & Crash Recovery Demonstration
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("3. TASK 8: CHECKPOINTING & CRASH RECOVERY DEMONSTRATION")
    print("-" * 85)

    crash_run_id = "s1_crash_recovery_demo"
    crash_ckpt_path = state_dir / f"{crash_run_id}.json"
    if crash_ckpt_path.exists():
        crash_ckpt_path.unlink()

    mock_llm.reset()
    crash_caught = False
    try:
        run_agent(
            user_message=s1_msg,
            scenario_id="S1_crash_demo",
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10.0,
            simulate_crash_after_tool=2,
            run_id=crash_run_id,
            state_dir=state_dir,
        )
    except RuntimeError as e:
        crash_caught = True
        print(f"  [Crash Phase] Simulated crash caught: {e}")

    assert crash_caught, "Simulated crash was not raised!"
    assert crash_ckpt_path.exists(), f"Checkpoint file missing at {crash_ckpt_path}"

    with open(crash_ckpt_path, "r", encoding="utf-8") as f:
        pre_crash_ckpt = json.load(f)

    tools_before_crash = pre_crash_ckpt["tools_already_executed"]
    print(f"  [Checkpoint Inspection] Pre-crash checkpoint saved at: state/{crash_run_id}.json")
    print(f"       - Step Number            : {pre_crash_ckpt['step_number']}")
    print(f"       - Tools Already Executed : {tools_before_crash}")
    print(f"       - LLM Calls Completed    : {pre_crash_ckpt['llm_calls']}")
    print(f"       - Tokens Logged          : {pre_crash_ckpt['tokens_used']}")
    assert tools_before_crash == ["get_order", "search_policy"], f"Unexpected pre-crash tools: {tools_before_crash}"

    # Resume from checkpoint
    print(f"\n  [Resume Phase] Calling resume_agent('{crash_run_id}')...")
    resumed_res = resume_agent(
        run_id=crash_run_id,
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10.0,
        state_dir=state_dir,
    )

    with open(crash_ckpt_path, "r", encoding="utf-8") as f:
        post_resume_ckpt = json.load(f)

    all_executed_tools = post_resume_ckpt["tools_already_executed"]
    print(f"  [Resume Completed]:")
    print(f"       - Final Status            : {resumed_res['status']}")
    print(f"       - Final Answer            : {resumed_res['final_answer']}")
    print(f"       - Total Tools Executed    : {all_executed_tools}")
    print(f"       - Total Model Calls       : {resumed_res['llm_calls']}")
    print(f"       - Total Tokens Used       : {resumed_res['tokens_used']}")

    expected_tool_sequence = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert all_executed_tools == expected_tool_sequence, (
        f"Expected tool sequence {expected_tool_sequence}, got {all_executed_tools}"
    )
    assert resumed_res["status"] == "REFUND", f"Expected status REFUND, got {resumed_res['status']}"
    assert resumed_res["llm_calls"] == 5, f"Expected 5 total LLM calls, got {resumed_res['llm_calls']}"
    print("       -> Result: PASS (Exact 4 tools executed without repetition; correct refund issued)")

    # ---------------------------------------------------------------------------
    # TASK 8B: FOCUSED CRASH-REPLAY FINANCIAL IDEMPOTENCY TEST
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("3B. CRASH-REPLAY FINANCIAL IDEMPOTENCY VERIFICATION")
    print("-" * 85)

    replay_run_id = "s1_crash_replay_idempotency_demo"
    replay_ledger_path = PROJECT_ROOT / "outputs" / "test_replay_ledger.jsonl"
    replay_ckpt_path = state_dir / f"{replay_run_id}.json"

    if replay_ledger_path.exists():
        replay_ledger_path.unlink()
    if replay_ckpt_path.exists():
        replay_ckpt_path.unlink()

    # Redirect default ledger path temporarily to isolated test ledger
    orig_ledger_path = refund_agent_mod.DEFAULT_LEDGER_PATH
    refund_agent_mod.DEFAULT_LEDGER_PATH = replay_ledger_path

    try:
        mock_llm.reset()
        # Step 1: Run agent until tool 3 (calculate_refund) completes and checkpoint is saved
        try:
            run_agent(
                user_message=s1_msg,
                scenario_id="S1_replay_demo",
                max_iterations=6,
                token_limit=6000,
                time_limit_seconds=10.0,
                simulate_crash_after_tool=3,
                run_id=replay_run_id,
                state_dir=state_dir,
            )
        except RuntimeError:
            pass

        assert replay_ckpt_path.exists(), "Replay checkpoint was not saved after tool 3"

        # Step 2: Simulate financial write (issue_refund) occurring before checkpoint update
        pre_crash_tx = issue_refund("ORD-100048", 128.25, ledger_path=replay_ledger_path)
        assert pre_crash_tx.get("status") == "issued", f"Failed to issue initial refund: {pre_crash_tx}"

        # Verify ledger has exactly 1 entry now, while checkpoint is still at step 3
        with open(replay_ledger_path, "r", encoding="utf-8") as f:
            lines_before_resume = [json.loads(line) for line in f if line.strip()]
        assert len(lines_before_resume) == 1, "Expected exactly 1 pre-crash ledger entry"
        orig_timestamp = lines_before_resume[0]["timestamp"]
        print(f"  [Crash-Replay Simulation] Financial write committed to ledger at: {orig_timestamp}")
        print(f"       Checkpoint state on disk deliberately halted at step 3 (prior to step 4 checkpoint).")

        # Step 3: Resume from step 3 checkpoint (agent will attempt to execute issue_refund again)
        print("  [Crash-Replay Simulation] Calling resume_agent to replay step 4...")
        replay_resumed = resume_agent(
            run_id=replay_run_id,
            max_iterations=6,
            token_limit=6000,
            time_limit_seconds=10.0,
            state_dir=state_dir,
        )

        # Step 4: Verify idempotency protection
        with open(replay_ledger_path, "r", encoding="utf-8") as f:
            lines_after_resume = [json.loads(line) for line in f if line.strip()]

        print(f"  [Ledger Audit After Replay]:")
        print(f"       - Total Ledger Records   : {len(lines_after_resume)} (Expected: exactly 1, no duplicates)")
        print(f"       - Original Timestamp     : {lines_after_resume[0]['timestamp']} (Intact)")
        print(f"       - Resumed Agent Status   : {replay_resumed['status']}")
        print(f"       - Resumed Final Answer   : {replay_resumed['final_answer']}")

        assert len(lines_after_resume) == 1, (
            f"Financial idempotency failed! Ledger contains {len(lines_after_resume)} records instead of 1."
        )
        assert lines_after_resume[0]["timestamp"] == orig_timestamp, "Original transaction was altered."
        assert lines_after_resume[0]["order_id"] == "ORD-100048"
        assert lines_after_resume[0]["refund_inr"] == 128.25
        assert replay_resumed["status"] == "REFUND"
        print("       -> Result: PASS (Financial transaction was not duplicated; original record remains intact)")
    finally:
        # Restore default ledger path and clean test artifacts
        refund_agent_mod.DEFAULT_LEDGER_PATH = orig_ledger_path
        if replay_ledger_path.exists():
            replay_ledger_path.unlink()
        if replay_ckpt_path.exists():
            replay_ckpt_path.unlink()

    # ---------------------------------------------------------------------------
    # TASK 5: Tool Map & Refund Ledger Verification
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("4. TASK 5: TOOL MAP & REFUND LEDGER VERIFICATION")
    print("-" * 85)
    with open(ledger_path, "r", encoding="utf-8") as f:
        ledger_lines = [json.loads(line) for line in f if line.strip()]

    print(f"  - Total Ledger Records Appended: {len(ledger_lines)}")
    for i, entry in enumerate(ledger_lines, 1):
        print(f"    {i}. [{entry['timestamp']}] Tool: {entry['tool']:<18} Order: {entry['order_id']} Amount: INR {entry['refund_inr']:.2f}")

    assert len(ledger_lines) == 3, f"Expected exactly 3 ledger entries (S1, S2, S3), got {len(ledger_lines)}"
    assert any(e["tool"] == "issue_refund" and e["order_id"] == "ORD-100048" for e in ledger_lines)
    assert any(e["tool"] == "issue_refund" and e["order_id"] == "ORD-100067" for e in ledger_lines)
    assert any(e["tool"] == "escalate_to_human" and e["order_id"] == "ORD-100145" for e in ledger_lines)
    print("  -> Ledger Verification: PASS")

    # ---------------------------------------------------------------------------
    # TASK 5B: FOCUSED TOOL-BOUNDARY SAFETY & IDEMPOTENCY TESTS
    # ---------------------------------------------------------------------------
    print("\n" + "-" * 85)
    print("5. FOCUSED TOOL-BOUNDARY SAFETY & IDEMPOTENCY VALIDATION TESTS")
    print("-" * 85)

    test_ledger_path = PROJECT_ROOT / "outputs" / "test_safety_ledger.jsonl"
    if test_ledger_path.exists():
        test_ledger_path.unlink()

    safety_tests = []

    # Test 1: Cancelled order refund attempt
    t1_res = issue_refund(order_id="ORD-100025", refund_inr=45.0, ledger_path=test_ledger_path)
    t1_passed = (t1_res.get("status") == "rejected" and "cancelled" in t1_res.get("error", "").lower())
    safety_tests.append(("1. Cancelled order refund attempt (ORD-100025)", t1_passed, t1_res))
    print(f"  [Safety Test 1] Cancelled order rejection: {t1_res.get('error')} -> {'PASS' if t1_passed else 'FAIL'}")

    # Test 2: Unknown order refund attempt
    t2_res = issue_refund(order_id="ORD-999999", refund_inr=100.0, ledger_path=test_ledger_path)
    t2_passed = (t2_res.get("status") == "rejected" and "not found" in t2_res.get("error", "").lower())
    safety_tests.append(("2. Unknown order refund attempt (ORD-999999)", t2_passed, t2_res))
    print(f"  [Safety Test 2] Unknown order rejection: {t2_res.get('error')} -> {'PASS' if t2_passed else 'FAIL'}")

    # Test 3: Refund of INR 1000 or less
    t3_calc = calculate_refund(order_amount=128.25, delivery_minutes=65)
    t3_passed = (t3_calc.get("refund_inr") == 128.25 and t3_calc.get("requires_escalation") is False and t3_calc.get("eligible") is True)
    safety_tests.append(("3. Refund of INR 1000 or less (calculate_refund)", t3_passed, t3_calc))
    print(f"  [Safety Test 3] Refund <= INR 1000 eligibility: refund={t3_calc.get('refund_inr')}, escalation={t3_calc.get('requires_escalation')} -> {'PASS' if t3_passed else 'FAIL'}")

    # Test 4: Refund above INR 1000 through automatic issuance
    t4_res = issue_refund(order_id="ORD-100145", refund_inr=1999.0, ledger_path=test_ledger_path)
    t4_passed = (t4_res.get("status") == "rejected" and "exceeds automatic limit" in t4_res.get("error", "").lower())
    safety_tests.append(("4. Refund above INR 1000 auto-issuance block", t4_passed, t4_res))
    print(f"  [Safety Test 4] High-value auto-issuance rejection: {t4_res.get('error')} -> {'PASS' if t4_passed else 'FAIL'}")

    # Test 5: Valid high-value escalation
    t5_res = escalate_to_human(order_id="ORD-100145", refund_inr=1999.0, ledger_path=test_ledger_path)
    t5_passed = (t5_res.get("status") == "escalated" and t5_res.get("refund_inr") == 1999.0)
    safety_tests.append(("5. Valid high-value escalation (ORD-100145)", t5_passed, t5_res))
    print(f"  [Safety Test 5] Valid high-value escalation: status={t5_res.get('status')}, refund=INR {t5_res.get('refund_inr')} -> {'PASS' if t5_passed else 'FAIL'}")

    # Test 6: Duplicate refund attempt (idempotency safeguard)
    t6_first = issue_refund(order_id="ORD-100048", refund_inr=128.25, ledger_path=test_ledger_path)
    assert t6_first.get("status") == "issued"
    t6_second = issue_refund(order_id="ORD-100048", refund_inr=128.25, ledger_path=test_ledger_path)
    t6_passed = (t6_second.get("status") == "already_issued" and t6_second.get("idempotent") is True)
    safety_tests.append(("6. Duplicate refund attempt idempotency", t6_passed, t6_second))
    print(f"  [Safety Test 6] Duplicate refund idempotency safeguard: status={t6_second.get('status')}, idempotent={t6_second.get('idempotent')} -> {'PASS' if t6_passed else 'FAIL'}")

    # Test 7: Valid ordinary refund
    t7_res = issue_refund(order_id="ORD-100067", refund_inr=135.0, ledger_path=test_ledger_path)
    t7_passed = (t7_res.get("status") == "issued" and t7_res.get("refund_inr") == 135.0)
    safety_tests.append(("7. Valid ordinary refund (ORD-100067)", t7_passed, t7_res))
    print(f"  [Safety Test 7] Valid ordinary refund issuance: status={t7_res.get('status')}, refund=INR {t7_res.get('refund_inr')} -> {'PASS' if t7_passed else 'FAIL'}")

    # Test 8: Comprehensive Input Hardening & Overflow Protection
    t8_nan = calculate_refund(order_amount=float("nan"), delivery_minutes=60)
    t8_inf = calculate_refund(order_amount=float("inf"), delivery_minutes=60)
    t8_neginf = calculate_refund(order_amount=-float("inf"), delivery_minutes=60)
    t8_nonnum_amt = calculate_refund(order_amount="not_a_number", delivery_minutes=60)
    t8_missing_amt = calculate_refund(order_amount=None, delivery_minutes=60)
    t8_overflow_amt = calculate_refund(order_amount=10**1000, delivery_minutes=60)

    t8_nan_del = calculate_refund(order_amount=100.0, delivery_minutes=float("nan"))
    t8_inf_del = calculate_refund(order_amount=100.0, delivery_minutes=float("inf"))
    t8_neginf_del = calculate_refund(order_amount=100.0, delivery_minutes=-float("inf"))
    t8_nonnum_del = calculate_refund(order_amount=100.0, delivery_minutes="thirty_mins")
    t8_missing_del = calculate_refund(order_amount=100.0, delivery_minutes=None)
    t8_overflow_del = calculate_refund(order_amount=100.0, delivery_minutes=10**1000)
    t8_neg_del = calculate_refund(order_amount=100.0, delivery_minutes=-10)

    # Tool execution rejections for invalid inputs (ensuring zero ledger writes)
    t8_issue_nan = issue_refund("ORD-100048", float("nan"), ledger_path=test_ledger_path)
    t8_issue_neginf = issue_refund("ORD-100048", -float("inf"), ledger_path=test_ledger_path)
    t8_issue_nonnum = issue_refund("ORD-100048", "invalid", ledger_path=test_ledger_path)
    t8_issue_none = issue_refund("ORD-100048", None, ledger_path=test_ledger_path)
    t8_issue_overflow = issue_refund("ORD-100048", 10**1000, ledger_path=test_ledger_path)
    t8_esc_overflow = escalate_to_human("ORD-100145", 10**1000, ledger_path=test_ledger_path)

    t8_calc_cases = [
        t8_nan, t8_inf, t8_neginf, t8_nonnum_amt, t8_missing_amt, t8_overflow_amt,
        t8_nan_del, t8_inf_del, t8_neginf_del, t8_nonnum_del, t8_missing_del, t8_overflow_del, t8_neg_del
    ]
    t8_issue_cases = [t8_issue_nan, t8_issue_neginf, t8_issue_nonnum, t8_issue_none, t8_issue_overflow, t8_esc_overflow]

    t8_passed = (
        all(res.get("refund_inr") == 0.0 and res.get("eligible") is False for res in t8_calc_cases)
        and all(res.get("status") == "rejected" and res.get("refund_inr") == 0.0 for res in t8_issue_cases)
    )
    safety_tests.append(("8. Input hardening (NaN/Inf/-Inf/Non-numeric/Missing/Overflow)", t8_passed, t8_nan))
    print(f"  [Safety Test 8] Input hardening (NaN/Inf/-Inf/Non-numeric/Missing/Overflow): all safe 0.0 & rejected -> {'PASS' if t8_passed else 'FAIL'}")

    # Verify test ledger row count (exactly 3 valid writes: t5 escalate ORD-100145, t6 first issue ORD-100048, t7 issue ORD-100067; t6 duplicate was blocked)
    with open(test_ledger_path, "r", encoding="utf-8") as f:
        test_ledger_count = sum(1 for line in f if line.strip())
    assert test_ledger_count == 3, f"Expected 3 ledger entries in test ledger, got {test_ledger_count}"

    # Clean up test ledger
    test_ledger_path.unlink()

    all_safety_passed = all(p for _, p, _ in safety_tests)
    assert all_safety_passed, "One or more safety tests failed!"
    print(f"\n  -> All 8 Focused Tool-Boundary Safety Tests Passed!")

    # ---------------------------------------------------------------------------
    # Write Evaluation Report
    # ---------------------------------------------------------------------------
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("INSTAMART CUSTOMER INTELLIGENCE & SUPPORT COPILOT\n")
        f.write("PART D, TASKS 5-8: REFUND AGENT EVALUATION & SAFETY REPORT\n")
        f.write("=" * 85 + "\n\n")

        f.write("1. BENCHMARK SCENARIO RESULTS (outputs/agent_runs.jsonl)\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'ID':<4} {'Expected Status':<16} {'Actual Status':<25} {'Refund (INR)':<14} {'Calls':<7} {'Tokens':<8}\n")
        f.write("-" * 85 + "\n")
        for res in scenario_results:
            f.write(
                f"{res['scenario_id']:<4} {res['expected_outcome']:<16} {res['status']:<25} "
                f"{res['refund_inr']:<14.2f} {res['llm_calls']:<7} {res['tokens_used']:<8}\n"
            )
        f.write("-" * 85 + "\n\n")

        f.write("2. SAFE STOP CONDITION DEMONSTRATIONS (TASK 7)\n")
        f.write("-" * 85 + "\n")
        f.write(f"  [7A] Maximum Iterations (S7) : status={s7_res['status']}, calls={s7_res['llm_calls']}, tokens={s7_res['tokens_used']} -> PASS\n")
        f.write(f"  [7B] Token Limit Demo (S1)   : status={s1_token_demo['status']}, calls={s1_token_demo['llm_calls']}, tokens={s1_token_demo['tokens_used']} -> PASS\n")
        f.write(f"  [7C] Timeout Demo (S1)       : status={s1_timeout_demo['status']}, calls={s1_timeout_demo['llm_calls']}, tokens={s1_timeout_demo['tokens_used']} -> PASS\n\n")

        f.write("3. CHECKPOINTING & CRASH RECOVERY DEMONSTRATION (TASK 8)\n")
        f.write("-" * 85 + "\n")
        f.write(f"  Pre-crash tools executed    : {tools_before_crash}\n")
        f.write(f"  Full execution tool sequence: {all_executed_tools}\n")
        f.write(f"  Sequence Check              : Exactly 4 distinct tools without repetition -> PASS\n")
        f.write(f"  Resumed Final Status        : {resumed_res['status']} -> PASS\n")
        f.write(f"  Resumed Final Answer        : {resumed_res['final_answer']} -> PASS\n\n")

        f.write("3B. CRASH-REPLAY FINANCIAL IDEMPOTENCY TEST\n")
        f.write("-" * 85 + "\n")
        f.write(f"  Financial transaction replay: issue_refund attempted after un-checkpointed transaction -> BLOCKED DUPLICATE -> PASS\n")
        f.write(f"  Ledger Integrity            : Exactly 1 record preserved; original timestamp intact -> PASS\n\n")

        f.write("4. REFUND LEDGER SUMMARY (outputs/refund_ledger.jsonl)\n")
        f.write("-" * 85 + "\n")
        for entry in ledger_lines:
            f.write(f"  - [{entry['timestamp']}] {entry['tool']:<18} {entry['order_id']} INR {entry['refund_inr']:.2f}\n")
        f.write("-" * 85 + "\n\n")

        f.write("5. FOCUSED TOOL-BOUNDARY SAFETY & IDEMPOTENCY TESTS\n")
        f.write("-" * 85 + "\n")
        for name, passed, detail in safety_tests:
            f.write(f"  - {name:<55}: {'PASS' if passed else 'FAIL'}\n")
            f.write(f"    Observation: {detail}\n")
        f.write("=" * 85 + "\n")

    print(f"\nSaved evaluation report to {report_path.relative_to(PROJECT_ROOT)}")
    print("\nALL TASKS 5-8 ASSERTIONS & SAFETY TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
