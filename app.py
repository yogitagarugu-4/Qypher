"""Streamlit dashboard for the Qypher prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from data_loader import REQUIRED_COLUMNS, validate_required_columns
from detect.comparison import compare_detectors
from detect.conventional_model import detect_conventional_anomalies
from detect.quantum_model import detect_quantum_anomalies
from history.store import analyze_transaction, load_history, submit_new_transaction
from protect.signature_simulator import simulate_signature_for_transaction
from protect.verifier import verify_transaction_signature
from respond.alerting import handle_alert
from respond.policy_engine import decide_action
from respond.trust_score import calculate_trust_score


DEFAULT_DATA_PATH = Path("data/transactions.csv")
DEFAULT_HISTORY_PATH = Path("data/history.csv")
GROUND_TRUTH_PATH = Path("data/ground_truth.csv")


@st.cache_data
def analyze_transactions(csv_path: str) -> Dict[str, Any]:
    """Run the project backend on the selected transaction CSV.

    Each row is treated as an individual transaction event and evaluated through the
    same PROTECT -> DETECT -> RESPOND flow as a live transaction. This prevents the
    whole CSV from being collapsed into a single aggregated score.
    """
    transactions_df = pd.read_csv(csv_path)
    validate_required_columns(transactions_df)

    conventional_df = detect_conventional_anomalies(transactions_df)
    quantum_df = detect_quantum_anomalies(transactions_df)

    comparison = compare_detectors(csv_path, str(GROUND_TRUTH_PATH))

    row_results = []
    for _, row in transactions_df.iterrows():
        tx = row.to_dict()
        tx["transaction_id"] = str(tx["transaction_id"])
        tx["timestamp"] = tx.get("timestamp", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))

        result = analyze_transaction(tx, persist_history=True)

        row_results.append(
            {
                "transaction_id": str(result["transaction_id"]),
                "amount": float(tx.get("amount", 0.0)),
                "timestamp": str(result["timestamp"]),
                "location": tx.get("location", ""),
                "device_id": tx.get("device_id", ""),
                "anomaly_score": float(result["anomaly_score"]),
                "anomaly_status": str(result["anomaly_status"]).upper(),
                "ml_result": str(result["anomaly_status"]).upper(),
                "quantum_result": str(result.get("quantum_label", "UNKNOWN")).upper(),
                "quantum_anomaly_score": float(result.get("quantum_score", 0.0)),
                "risk_score": float(result["trust_score"]),
                "signature_valid": bool(result["signature_valid"]),
                "action": str(result["action"]).upper(),
                "alert": bool(result.get("alert", False)),
            }
        )

    monitor_df = pd.DataFrame(row_results)
    monitor_df["transaction_id"] = monitor_df["transaction_id"].astype(str)
    total_transactions = len(transactions_df)
    normal_transactions = int(((monitor_df["ml_result"] == "NORMAL") & (monitor_df["quantum_result"] == "NORMAL")).sum())
    anomalous_transactions = int(total_transactions - normal_transactions)
    blocked_transactions = int(monitor_df["action"].isin(["DENY", "QUARANTINE"]).sum())
    monitored_transactions = int((monitor_df["action"] == "MONITOR").sum())

    transaction_preview = transactions_df.iloc[0].to_dict() if not transactions_df.empty else {}
    preview_signature = simulate_signature_for_transaction(transaction_preview) if transaction_preview else {"signature": ""}
    preview_valid = verify_transaction_signature(transaction_preview, preview_signature["signature"]) if transaction_preview else False
    preview_label = str(conventional_df.iloc[0]["prediction_label"]).upper() if not conventional_df.empty else "UNKNOWN"
    preview_q_label = str(quantum_df.iloc[0]["quantum_prediction_label"]).upper() if not quantum_df.empty else "UNKNOWN"
    preview_action = decide_action(
        calculate_trust_score(
            signature_valid=preview_valid,
            conventional_label=preview_label,
            quantum_label=preview_q_label,
            amount=float(transaction_preview.get("amount", 0.0)),
            distance_from_usual=float(transaction_preview.get("distance_from_usual", 0.0)),
            failed_attempts=int(transaction_preview.get("failed_attempts", 0)),
            is_new_device=bool(transaction_preview.get("is_new_device", False)),
        )
    ) if transaction_preview else "ALLOW"

    transaction_level_results = monitor_df[["transaction_id", "anomaly_score", "anomaly_status", "action"]].copy()
    transaction_level_results["anomaly_status"] = transaction_level_results["anomaly_status"].astype(str).str.upper()
    transaction_level_results["action"] = transaction_level_results["action"].astype(str).str.upper()

    return {
        "transactions_df": transactions_df,
        "conventional_df": conventional_df,
        "quantum_df": quantum_df,
        "comparison": comparison,
        "monitor_df": monitor_df,
        "transaction_level_results": transaction_level_results,
        "total_transactions": total_transactions,
        "normal_transactions": normal_transactions,
        "anomalous_transactions": anomalous_transactions,
        "blocked_transactions": blocked_transactions,
        "monitored_transactions": monitored_transactions,
        "preview_signature_valid": preview_valid,
        "preview_action": preview_action,
        "preview_transaction": transaction_preview,
    }


def safe_load_transactions(uploaded_file=None) -> tuple[Path | str, pd.DataFrame | None]:
    """Resolve the path to the active CSV and validate its structure."""
    if uploaded_file is not None:
        target = Path("data/uploaded_transactions.csv")
        target.parent.mkdir(exist_ok=True)
        with open(target, "wb") as file:
            file.write(uploaded_file.getvalue())
        csv_path = str(target)
    else:
        csv_path = str(DEFAULT_DATA_PATH)

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"The file could not be found: {csv_path}")
        return csv_path, None
    except Exception as exc:  # pragma: no cover - UI safety
        st.error(f"The file could not be read: {exc}")
        return csv_path, None

    try:
        validate_required_columns(df)
    except ValueError as exc:
        st.error(f"The uploaded CSV is missing required columns. {exc}")
        return csv_path, None

    return csv_path, df


def render_pill(label: str, color: str) -> None:
    """Show a small status chip."""
    st.markdown(
        f"<span style='display:inline-block;padding:6px 10px;border-radius:999px;background:{color};color:white;font-weight:600'>{label}</span>",
        unsafe_allow_html=True,
    )


def render_protect_section(transaction_sample: Dict[str, Any], status_valid: bool) -> None:
    st.markdown("### 🛡️ 1. PROTECT")
    col_signature, col_details = st.columns([2, 1])

    with col_signature:
        signature = simulate_signature_for_transaction(transaction_sample)
        valid = verify_transaction_signature(transaction_sample, signature["signature"])
        if valid:
            st.success("Signature: VALID")
        else:
            st.error("Signature: INVALID")
        st.caption("Protect checks whether the transaction has a valid digital signature.")

    with col_details:
        st.markdown("**Public/Private Key concept**")
        st.write("- Public key: used to verify a signature.")
        st.write("- Private key: kept secure and never shown in the UI.")
        st.write("- Digital Signature: " + ("VALID" if valid else "INVALID"))


def render_detect_section(analysis: Dict[str, Any]) -> None:
    st.markdown("### 🔍 2. DETECT")

    conventional = analysis["conventional_df"].iloc[0].copy()
    quantum = analysis["quantum_df"].iloc[0].copy()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### CONVENTIONAL ML")
        st.caption("Isolation Forest")
        st.metric("Prediction", str(conventional["prediction_label"]))
        st.metric("Anomaly Score", f"{float(conventional['anomaly_score']):.4f}")

    with c2:
        st.markdown("#### QUANTUM-INSPIRED")
        st.caption("Quantum-inspired detector")
        st.metric("Prediction", str(quantum["quantum_prediction_label"]))
        st.metric("Anomaly Score", f"{float(quantum['quantum_anomaly_score']):.4f}")

    metrics = analysis["comparison"]
    metric_rows = [
        ["Precision", f"{metrics['conventional_metrics']['precision']:.4f}", f"{metrics['quantum_metrics']['precision']:.4f}"],
        ["Recall", f"{metrics['conventional_metrics']['recall']:.4f}", f"{metrics['quantum_metrics']['recall']:.4f}"],
        ["F1 Score", f"{metrics['conventional_metrics']['f1_score']:.4f}", f"{metrics['quantum_metrics']['f1_score']:.4f}"],
        ["False Positives", str(metrics['conventional_metrics']['false_positives']), str(metrics['quantum_metrics']['false_positives'])],
        ["False Negatives", str(metrics['conventional_metrics']['false_negatives']), str(metrics['quantum_metrics']['false_negatives'])],
        ["Processing Time", f"{metrics['conventional_summary']['execution_time_seconds']:.6f}s", f"{metrics['quantum_summary']['execution_time_seconds']:.6f}s"],
    ]

    comparison_df = pd.DataFrame(metric_rows, columns=["Metric", "Conventional ML", "Quantum-Inspired"])
    st.markdown("#### Comparison")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def render_respond_section(analysis: Dict[str, Any]) -> None:
    st.markdown("### 🚨 3. RESPOND")

    selected_transaction = analysis["preview_transaction"]
    signature_valid = analysis["preview_signature_valid"]
    conventional_label = str(analysis["conventional_df"].iloc[0]["prediction_label"]).upper()
    quantum_label = str(analysis["quantum_df"].iloc[0]["quantum_prediction_label"]).upper()
    trust_score = calculate_trust_score(
        signature_valid=signature_valid,
        conventional_label=conventional_label,
        quantum_label=quantum_label,
        amount=float(selected_transaction.get("amount", 0.0)),
        distance_from_usual=float(selected_transaction.get("distance_from_usual", 0.0)),
        failed_attempts=int(selected_transaction.get("failed_attempts", 0)),
        is_new_device=bool(selected_transaction.get("is_new_device", False)),
    )
    action = decide_action(trust_score)
    explanation_map = {
        "ALLOW": "Normal transaction → allowed.",
        "MONITOR": "Unusual behavior detected → transaction monitored.",
        "DENY": "Unusual behavior detected → transaction denied.",
        "QUARANTINE": "Unusual behavior detected → transaction quarantined.",
    }

    st.metric("Trust Score", f"{trust_score:.1f}/100")
    st.metric("Final Action", action)
    st.caption(explanation_map.get(action, "Transaction reviewed by the Qypher policy engine."))

    if action in {"DENY", "QUARANTINE"}:
        handle_alert(action, "Transaction evaluated by the Qypher response policy.")
        st.warning("Alert triggered for a risky transaction.")


def render_monitor_section(analysis: Dict[str, Any]) -> None:
    st.markdown("### 📊 TRANSACTION MONITOR")

    total = analysis["total_transactions"]
    normal = analysis["normal_transactions"]
    anomalous = analysis["anomalous_transactions"]
    blocked = analysis["blocked_transactions"]
    monitored = analysis["monitored_transactions"]

    cols = st.columns(5)
    cols[0].metric("Total Transactions", total)
    cols[1].metric("Normal Transactions", normal)
    cols[2].metric("Anomalous Transactions", anomalous)
    cols[3].metric("Transactions Blocked", blocked)
    cols[4].metric("Transactions Monitored", monitored)

    tx_table = analysis["transaction_level_results"].copy()
    tx_table = tx_table.rename(columns={
        "transaction_id": "Transaction ID",
        "anomaly_score": "Anomaly Score",
        "anomaly_status": "Anomaly Status",
        "action": "Action",
    })
    tx_table = tx_table.sort_values(by="Transaction ID").reset_index(drop=True)
    st.dataframe(tx_table, use_container_width=True, hide_index=True)


def render_history_section() -> None:
    st.markdown("### 📜 TRANSACTION HISTORY")
    try:
        history = load_history()
    except Exception as exc:  # pragma: no cover - UI safety
        st.error(f"The history file could not be loaded: {exc}")
        return

    if history.empty:
        st.info("No transaction history has been saved yet.")
        return

    filter_options = ["All", "Normal", "Anomaly", "Allowed", "Monitored", "Denied", "Quarantined"]
    selected_filter = st.selectbox("Filter by status", filter_options, index=0)
    if selected_filter == "All":
        filtered = history
    elif selected_filter == "Normal":
        filtered = history[(history["conventional_label"].str.upper() == "NORMAL") & (history["quantum_label"].str.upper() == "NORMAL")]
    elif selected_filter == "Anomaly":
        filtered = history[(history["conventional_label"].str.upper() == "ANOMALY") | (history["quantum_label"].str.upper() == "ANOMALY")]
    elif selected_filter == "Allowed":
        filtered = history[history["action"].str.upper() == "ALLOW"]
    elif selected_filter == "Monitored":
        filtered = history[history["action"].str.upper() == "MONITOR"]
    elif selected_filter == "Denied":
        filtered = history[history["action"].str.upper() == "DENY"]
    else:
        filtered = history[history["action"].str.upper() == "QUARANTINE"]

    st.dataframe(filtered, use_container_width=True, hide_index=True)


def render_how_it_works() -> None:
    st.markdown("### 🧒 How Qypher works")
    st.write("🛡️ Protect: Is this transaction properly signed?")
    st.write("🔍 Detect: Does this transaction look unusual?")
    st.write("🚨 Respond: What should we do about it?")


def main() -> None:
    st.set_page_config(page_title="QYPHER", page_icon="🛡️", layout="wide")

    st.title("QYPHER")
    st.subheader("Quantum-Inspired Security for Digital Transactions")
    st.markdown("## 🛡️ PROTECT  →  🔍 DETECT  →  🚨 RESPOND")
    st.caption("Qypher checks the transaction, looks for unusual behavior, and decides what action to take.")

    with st.sidebar:
        st.header("Qypher Controls")
        uploaded_file = st.file_uploader("Upload Transactions CSV", type=["csv"])
        st.caption("If no file is uploaded, the system uses data/transactions.csv.")
        st.sidebar.button("Run Analysis", use_container_width=True)
        st.sidebar.button("Make New Transaction", use_container_width=True)
        st.sidebar.button("View Transaction History", use_container_width=True)

    csv_path, transactions_df = safe_load_transactions(uploaded_file)

    if transactions_df is None:
        st.stop()

    analysis = analyze_transactions(csv_path)

    render_protect_section(analysis["preview_transaction"], analysis["preview_signature_valid"])
    render_detect_section(analysis)
    render_respond_section(analysis)

    st.markdown("---")

    with st.form("new_transaction_form", clear_on_submit=True):
        st.markdown("### 🧾 Make New Transaction")
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Amount", min_value=0.0, value=120.0, step=1.0)
            location = st.text_input("Location", value="Boston")
            merchant_category = st.text_input("Merchant Category", value="Retail")
            payment_method = st.text_input("Payment Method", value="Credit Card")
            device_id = st.text_input("Device ID", value="DEV-NEW-01")
        with c2:
            transaction_type = st.text_input("Transaction Type", value="purchase")
            hour = st.number_input("Hour", min_value=0, max_value=23, value=14)
            is_new_device = st.checkbox("Is New Device", value=False)
            distance_from_usual = st.number_input("Distance From Usual", min_value=0.0, value=15.0, step=1.0)
            failed_attempts = st.number_input("Failed Attempts", min_value=0, max_value=50, value=0)

        submitted = st.form_submit_button("SUBMIT TRANSACTION")

    if submitted:
        new_transaction = {
            "amount": float(amount),
            "location": location,
            "merchant_category": merchant_category,
            "payment_method": payment_method,
            "device_id": device_id,
            "transaction_type": transaction_type,
            "hour": int(hour),
            "is_new_device": bool(is_new_device),
            "distance_from_usual": float(distance_from_usual),
            "failed_attempts": int(failed_attempts),
        }
        try:
            result = submit_new_transaction(new_transaction)
            st.success("Transaction submitted successfully.")
            st.json(result)
        except Exception as exc:
            st.error(f"The transaction could not be submitted: {exc}")

    render_monitor_section(analysis)
    render_history_section()
    render_how_it_works()

    st.markdown("---")
    st.caption("Qypher Prototype — Quantum-Inspired Security for Digital Transactions")


if __name__ == "__main__":
    main()
