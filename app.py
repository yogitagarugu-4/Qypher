"""Streamlit dashboard for transaction-level Qypher analysis."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import streamlit as st

from data_loader import (
    REQUIRED_COLUMNS,
    get_features_for_anomaly_detection,
    normalize_boolean_value,
    validate_required_columns,
)
from detect.comparison import compare_detectors
from detect.conventional_model import detect_conventional_anomalies
from detect.quantum_model import detect_quantum_anomalies
from history.store import (
    append_history_records,
    create_history_record,
    load_history,
)
from protect.signature_simulator import simulate_signature_for_transaction
from protect.verifier import verify_transaction_signature
from respond.alerting import trigger_sound_alert
from respond.policy_engine import evaluate_transaction
from respond.trust_score import assess_transaction_risk


DEFAULT_DATA_PATH = Path("data/transactions.csv")
GROUND_TRUTH_PATH = Path("data/ground_truth.csv")
ACCOUNT_ID = "QYP-ACCT-1042"
MODEL_OPTIONS = {
    "isolation_forest": {
        "label": "Isolation Forest",
        "short_label": "ISOLATION FOREST",
        "detector": detect_conventional_anomalies,
    },
    "qsvm": {
        "label": "Quantum-Inspired QSVM",
        "short_label": "QUANTUM-INSPIRED QSVM",
        "detector": detect_quantum_anomalies,
    },
}
ACTION_OPTIONS = ["ALLOW", "MONITOR", "DENY", "QUARANTINE"]


def analyze_transactions(
    csv_path: str,
    model: str,
    cache_token: str = "",
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the selected detector and create one response for every CSV row."""
    if model not in MODEL_OPTIONS:
        raise ValueError("Choose either Isolation Forest or Quantum-Inspired QSVM.")

    transactions = pd.read_csv(csv_path)
    validate_required_columns(transactions)
    if transactions.empty:
        raise ValueError("The transaction CSV contains no transaction rows.")
    transaction_ids = transactions["transaction_id"]
    if transaction_ids.isna().any() or transaction_ids.astype(str).str.strip().eq("").any():
        raise ValueError("Every transaction must have a non-empty transaction_id.")
    if transaction_ids.astype(str).duplicated().any():
        raise ValueError("transaction_id values must be unique.")

    method = MODEL_OPTIONS[model]
    predictions = method["detector"](transactions)
    if len(predictions) != len(transactions):
        raise ValueError(
            f"{method['label']} returned {len(predictions)} results for "
            f"{len(transactions)} input transactions."
        )

    prediction_id_column = "transaction_id"
    predictions[prediction_id_column] = predictions[prediction_id_column].astype(str)
    transaction_ids = transaction_ids.astype(str)
    predictions = predictions.set_index(prediction_id_column).reindex(transaction_ids)
    if predictions["prediction_label" if model == "isolation_forest" else "quantum_prediction_label"].isna().any():
        raise ValueError("Detector results could not be aligned to every transaction.")

    features = get_features_for_anomaly_detection(transactions)
    prepared_transactions = features["transaction_data"].reset_index(drop=True)
    amount_values = pd.to_numeric(transactions["amount"], errors="coerce").dropna()
    reference_amount_median = float(amount_values.median()) if not amount_values.empty else 0.0
    reference_amount_mad = (
        float((amount_values - reference_amount_median).abs().median())
        if not amount_values.empty
        else 0.0
    )

    results = transactions.reset_index(drop=True).copy()
    results["transaction_id"] = transaction_ids.reset_index(drop=True)
    results["timestamp"] = prepared_transactions["timestamp"].astype(str).to_numpy()
    results["amount"] = pd.to_numeric(prepared_transactions["amount"], errors="coerce").to_numpy()
    results["detection_method"] = method["label"]
    results["model_key"] = model
    label_column = "prediction_label" if model == "isolation_forest" else "quantum_prediction_label"
    score_column = "anomaly_score" if model == "isolation_forest" else "quantum_anomaly_score"
    results["detection"] = predictions[label_column].astype(str).str.upper().to_numpy()
    results["model_score"] = pd.to_numeric(predictions[score_column], errors="coerce").to_numpy()

    trust_scores: list[float] = []
    risk_levels: list[str] = []
    risk_factors_list: list[list[dict[str, Any]]] = []
    explanations: list[list[str]] = []
    decisions: list[str] = []
    reasons: list[str] = []
    signatures: list[bool] = []

    for position, row in results.iterrows():
        transaction = transactions.iloc[position].to_dict()
        transaction["transaction_id"] = str(transaction["transaction_id"])
        signature = simulate_signature_for_transaction(transaction)
        signature_valid = verify_transaction_signature(transaction, signature["signature"])

        prepared = prepared_transactions.iloc[position].to_dict()
        detection = str(row["detection"])
        conventional_label = detection if model == "isolation_forest" else "NORMAL"
        quantum_label = detection if model == "qsvm" else "NORMAL"
        risk = assess_transaction_risk(
            prepared,
            conventional_label,
            quantum_label,
            signature_valid,
            reference_amount_median=reference_amount_median,
            reference_amount_mad=reference_amount_mad,
            transactions_last_hour=int(
                pd.to_numeric(
                    pd.Series([prepared.get("transactions_last_hour", 0)]),
                    errors="coerce",
                ).fillna(0).iloc[0]
            ),
        )
        decision = evaluate_transaction(
            trust_score=risk["trust_score"],
            signature_valid=signature_valid,
            conventional_label=conventional_label,
            quantum_label=quantum_label,
            amount=float(prepared["amount"]),
            distance_from_usual=float(prepared.get("distance_from_usual", 0.0)),
            failed_attempts=int(prepared.get("failed_attempts", 0)),
            is_new_device=normalize_boolean_value(prepared.get("is_new_device", False)),
            risk_factors=risk["risk_factors"],
        )

        trust_scores.append(float(risk["trust_score"]))
        risk_levels.append(str(risk["risk_level"]))
        risk_factors_list.append(risk["risk_factors"])
        explanations.append(risk["explanation"])
        decisions.append(str(decision["action"]))
        reasons.append(str(decision["reason"]))
        signatures.append(signature_valid)
        if progress_callback is not None:
            progress_callback(position + 1, len(results))

    results["trust_score"] = trust_scores
    results["risk_level"] = risk_levels
    results["risk_factors"] = risk_factors_list
    results["explanation"] = explanations
    results["decision"] = decisions
    results["reason"] = reasons
    results["signature_valid"] = signatures

    result_columns = [
        "transaction_id",
        "model_key",
        "detection_method",
        "detection",
        "model_score",
        "trust_score",
        "risk_level",
        "decision",
        "risk_factors",
        "explanation",
    ]
    return {
        "transactions_df": transactions,
        "monitor_df": results,
        "transaction_level_results": results[result_columns].copy(),
        "model": model,
        "method_label": method["label"],
        "input_row_count": len(transactions),
        "result_row_count": len(results),
        "total_transactions": len(results),
        "normal_transactions": int(results["detection"].eq("NORMAL").sum()),
        "anomalous_transactions": int(results["detection"].eq("ANOMALY").sum()),
        "high_risk_transactions": int(results["trust_score"].le(30).sum()),
        "cache_token": cache_token,
    }


def safe_load_transactions(uploaded_file=None) -> tuple[Path, pd.DataFrame | None]:
    """Save and validate the uploaded CSV, or use the bundled sample dataset."""
    if uploaded_file is not None:
        target = Path("data/uploaded_transactions.csv")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(uploaded_file.getvalue())
        csv_path = target
    else:
        csv_path = DEFAULT_DATA_PATH

    try:
        transactions = pd.read_csv(csv_path)
        validate_required_columns(transactions)
        if transactions.empty:
            raise ValueError("The transaction CSV contains no transaction rows.")
    except FileNotFoundError:
        st.error(f"The transaction file could not be found: {csv_path}")
        return csv_path, None
    except pd.errors.EmptyDataError:
        st.error("The transaction CSV is empty or has no header row.")
        return csv_path, None
    except (OSError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"The transaction CSV could not be used: {exc}")
        return csv_path, None

    return csv_path, transactions


def render_styles() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        html, body, [data-testid="stAppViewContainer"], .stApp { background: #F4F7FB; color: #0F172A; }
        [data-testid="stHeader"] { background: rgba(244, 247, 251, .96); }
        .block-container { max-width: 1420px; padding-top: 1.2rem; padding-bottom: 3rem; }
        h1, h2, h3, h4 { color: #172554; }
        p, label, li, [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"] { color: #0F172A; }
        h1 { font-size: clamp(2.5rem, 5vw, 3.7rem) !important; font-weight: 800 !important; }
        [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1rem; }
        [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color: #64748B !important; }
        [data-testid="stMetricValue"], [data-testid="stMetricValue"] div { color: #172554 !important; font-weight: 750; }
        div[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 16px; }
        .qy-hero { display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:1.4rem 1.6rem;
          border:1px solid #E2E8F0; border-radius:18px; background:#FFFFFF; margin-bottom:1rem; }
        .qy-kicker,.qy-label { color:#64748B; font-size:.75rem; font-weight:750; letter-spacing:.13em; text-transform:uppercase; }
        .qy-subtitle { color:#64748B !important; margin:.3rem 0; }
        .qy-live { color:#16A34A; background:#F0FDF4; border:1px solid #BBF7D0; border-radius:999px;
          padding:.65rem .9rem; font-size:.78rem; font-weight:750; white-space:nowrap; }
        .qy-account { height:100%; padding:1rem 1.1rem; border-radius:14px; border:1px solid #E2E8F0; background:#FFFFFF; }
        .qy-name { color:#172554; font-size:1.25rem; font-weight:750; margin-top:.3rem; }
        .qy-card { height:100%; padding:1rem; border-radius:14px; border:1px solid #E2E8F0; background:#FFFFFF; }
        .qy-card h3 { margin:.4rem 0; }
        .qy-card p { color:#64748B; font-size:.93rem; }
        .stButton > button[kind="primary"], .stFormSubmitButton > button { background:#2563EB; color:white;
          border:0; border-radius:10px; font-weight:750; min-height:2.8rem; }
        .stButton > button[kind="primary"]:hover { background:#1D4ED8; }
        [data-testid="stCaptionContainer"] p { color:#64748B; }
        [data-testid="stDataFrame"] { border:1px solid #E2E8F0; border-radius:12px; }
        [data-testid="stAlert"] p { color:#0F172A; }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stFileUploader"] section { background:#FFFFFF; color:#0F172A; border-color:#E2E8F0; }
        .qy-decision-label { color:#64748B; font-size:.75rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
        .qy-decision-badge { display:inline-block; margin-top:.25rem; padding:.38rem .7rem;
          border-radius:999px; font-weight:800; font-size:.9rem; border:1px solid transparent; }
        .qy-decision-allow { color:#16A34A; background:#F0FDF4; border-color:#BBF7D0; }
        .qy-decision-monitor { color:#D97706; background:#FFFBEB; border-color:#FDE68A; }
        .qy-decision-deny { color:#DC2626; background:#FEF2F2; border-color:#FECACA; }
        .qy-decision-quarantine { color:#991B1B; background:#FEF2F2; border-color:#FCA5A5; }
        @media(max-width:700px) { .block-container { padding-left:1rem; padding-right:1rem; }
          .qy-hero { align-items:flex-start; flex-direction:column; padding:1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


DECISION_STYLES = {
    "ALLOW": ("#16A34A", "#F0FDF4", "#BBF7D0"),
    "MONITOR": ("#D97706", "#FFFBEB", "#FDE68A"),
    "DENY": ("#DC2626", "#FEF2F2", "#FECACA"),
    "QUARANTINE": ("#991B1B", "#FEF2F2", "#FCA5A5"),
}


def _decision_cell_style(value: Any) -> str:
    foreground, background, border = DECISION_STYLES.get(
        str(value).upper(),
        ("#0F172A", "#FFFFFF", "#E2E8F0"),
    )
    return (
        f"color: {foreground}; background-color: {background}; "
        f"font-weight: 750; border: 1px solid {border};"
    )


def _render_decision_badge(container: Any, decision: str) -> None:
    normalized = str(decision).upper()
    if normalized not in DECISION_STYLES:
        normalized = "DENY"
    container.markdown(
        f'<div class="qy-decision-label">Decision</div>'
        f'<div class="qy-decision-badge qy-decision-{normalized.lower()}">{normalized}</div>',
        unsafe_allow_html=True,
    )


def render_header(analysis: dict[str, Any] | None) -> None:
    st.markdown(
        '<div class="qy-hero"><div><div class="qy-kicker">ACCOUNT TRANSACTION MONITORING</div>'
        '<h1>QYPHER</h1><p class="qy-subtitle">Quantum-Inspired Security for Digital Transactions</p>'
        '<p class="qy-subtitle">Monitor account activity, detect abnormal behaviour, calculate trust, and respond.</p>'
        '</div><div class="qy-live">● MONITORING ACTIVE</div></div>',
        unsafe_allow_html=True,
    )
    account_columns = st.columns([1.35, 1, 1])
    account_columns[0].markdown(
        '<div class="qy-account"><div class="qy-label">Logged in as</div>'
        '<div class="qy-name">Alex Morgan</div><div>Treasury Analyst</div></div>',
        unsafe_allow_html=True,
    )
    account_columns[1].markdown(
        f'<div class="qy-account"><div class="qy-label">Account</div>'
        f'<div class="qy-name">{ACCOUNT_ID}</div><div>Account status · Active</div></div>',
        unsafe_allow_html=True,
    )
    account_columns[2].markdown(
        '<div class="qy-account"><div class="qy-label">Monitoring status</div>'
        '<div class="qy-name" style="color:#16A34A">● ACTIVE</div><div>Transactions monitored individually</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qy-card" style="text-align:center;margin:1rem 0">'
        '<b>TRANSACTION</b>　→　<b>BEHAVIOUR ANALYSIS</b>　→　<b>TRUST SCORE</b>　→　'
        '<b>ALLOW · MONITOR · DENY · QUARANTINE</b></div>',
        unsafe_allow_html=True,
    )

    if analysis is None:
        try:
            history = load_history()
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            st.error(f"Transaction history could not be loaded: {exc}")
            history = pd.DataFrame()
        if not history.empty:
            detection = _history_detection(history)
            totals = (
                len(history),
                int(detection.eq("NORMAL").sum()),
                int(detection.eq("ANOMALY").sum()),
                int(pd.to_numeric(
                    history.get("trust_score", pd.Series(100, index=history.index)),
                    errors="coerce",
                ).le(30).sum()),
            )
        else:
            totals = (0, 0, 0, 0)
    else:
        totals = (
            analysis["total_transactions"],
            analysis["normal_transactions"],
            analysis["anomalous_transactions"],
            analysis["high_risk_transactions"],
        )
    metric_columns = st.columns(4)
    for column, label, value in zip(
        metric_columns,
        ("Transactions Monitored", "Normal", "Anomalies", "High Risk"),
        totals,
    ):
        column.metric(label, f"{value:,}")


def render_method_choices() -> str | None:
    st.markdown('<div id="choose-detection-method"></div>', unsafe_allow_html=True)
    st.subheader("Choose Detection Method")
    st.caption("Select one detector to analyze every transaction in the uploaded file.")
    left, right = st.columns(2)
    with left:
        st.markdown(
            '<div class="qy-card"><div class="qy-label">Classical baseline</div>'
            '<h3>Isolation Forest</h3><p>Classical anomaly detection for observations that are relatively easy to isolate.</p></div>',
            unsafe_allow_html=True,
        )
        isolation_selected = st.button(
            "ISOLATION FOREST",
            key="choose_isolation_forest",
            type="primary",
            use_container_width=True,
        )
    with right:
        st.markdown(
            '<div class="qy-card"><div class="qy-label">Qypher proposed approach</div>'
            '<h3>Quantum-Inspired QSVM</h3><p>Prototype feature-similarity detector. Its output is not guaranteed to outperform the baseline.</p></div>',
            unsafe_allow_html=True,
        )
        qsvm_selected = st.button(
            "QUANTUM-INSPIRED QSVM",
            key="choose_qsvm",
            type="primary",
            use_container_width=True,
        )
    st.info(
        "Why QSVM? Qypher explores a quantum-inspired feature representation to identify relationships "
        "between behavioural features. This prototype currently uses a classical cosine-similarity model; "
        "it is not a true QSVM or quantum hardware implementation."
    )
    if isolation_selected:
        return "isolation_forest"
    if qsvm_selected:
        return "qsvm"
    return None


def _persist_analysis(analysis: dict[str, Any], token: str) -> None:
    model = analysis["model"]
    saved_key = f"{token}:{model}"
    if st.session_state.get("history_saved_key") == saved_key:
        return

    records = []
    for _, row in analysis["monitor_df"].iterrows():
        records.append(create_history_record(
            transaction_id=str(row["transaction_id"]),
            timestamp=str(row["timestamp"]),
            amount=float(row["amount"]),
            signature_valid=bool(row["signature_valid"]),
            conventional_label=str(row["detection"]) if model == "isolation_forest" else "NOT RUN",
            quantum_label=str(row["detection"]) if model == "qsvm" else "NOT RUN",
            conventional_score=float(row["model_score"]) if model == "isolation_forest" else 0.0,
            quantum_score=float(row["model_score"]) if model == "qsvm" else 0.0,
            trust_score=float(row["trust_score"]),
            action=str(row["decision"]),
            alert_message=(
                f"ALERT: {row['decision']} - {row['reason']}"
                if row["decision"] in {"DENY", "QUARANTINE"}
                else "No alert required."
            ),
            location=str(row.get("location", "")),
            device_id=str(row.get("device_id", "")),
            anomaly_status=str(row["detection"]),
            reason=str(row["reason"]),
            risk_level=str(row["risk_level"]),
            risk_factors=row["risk_factors"],
            detection_method=str(row["detection_method"]),
            account=ACCOUNT_ID,
        ))
    append_history_records(records)
    st.session_state["history_saved_key"] = saved_key


def render_analysis_results(analysis: dict[str, Any]) -> None:
    st.markdown('<div id="live-transaction-analysis"></div>', unsafe_allow_html=True)
    st.subheader("Transaction Analysis")
    st.caption(
        f"Detection Method: {analysis['method_label']} · "
        f"{analysis['result_row_count']:,} individual transaction results"
    )
    result = analysis["monitor_df"]
    anomalies = result[result["detection"].eq("ANOMALY")]
    if not anomalies.empty:
        first = anomalies.iloc[0]
        st.warning(
            f"⚠ ANOMALY DETECTED · Transaction {first['transaction_id']} · "
            f"Trust {first['trust_score']:.0f}/100 · Decision {first['decision']}. "
            f"{len(anomalies):,} anomalous transaction(s) are available for review."
        )

    display = pd.DataFrame({
        "Transaction": result["transaction_id"].astype(str),
        "Time": result["timestamp"].astype(str),
        "Amount": result["amount"].map(lambda value: f"₹{float(value):,.2f}"),
        "Detection": result["detection"],
        "Model score": result["model_score"].map(lambda value: f"{float(value):.4f}"),
        "Trust Score": result["trust_score"].map(lambda value: f"{float(value):.0f} / 100"),
        "Decision": result["decision"],
    })
    st.caption(f"Input rows: {analysis['input_row_count']:,} · Results: {analysis['result_row_count']:,}. No rows are truncated.")
    st.dataframe(
        display.style.map(_decision_cell_style, subset=["Decision"]),
        use_container_width=True,
        hide_index=True,
        height=470,
    )

    details_pool = anomalies if not anomalies.empty else result
    selected_id = st.selectbox(
        "Choose a transaction to inspect",
        details_pool["transaction_id"].astype(str).tolist(),
        key=f"details_{analysis['cache_token']}_{analysis['model']}",
    )
    selected = result[result["transaction_id"].astype(str).eq(selected_id)].iloc[0]
    with st.expander(f"VIEW DETAILS · {selected_id}", expanded=not anomalies.empty):
        detail_columns = st.columns(4)
        detail_columns[0].metric("Amount", f"₹{float(selected['amount']):,.2f}")
        detail_columns[1].metric("Detection", str(selected["detection"]))
        detail_columns[2].metric("Trust Score", f"{float(selected['trust_score']):.0f} / 100")
        _render_decision_badge(detail_columns[3], str(selected["decision"]))
        st.caption(
            f"Account {ACCOUNT_ID} · {selected['timestamp']} · "
            f"Location: {selected.get('location', 'Unknown')} · "
            f"Device: {selected.get('device_id', 'Unknown')} · "
            f"Signature simulation: {'VALID' if selected['signature_valid'] else 'INVALID'}"
        )
        st.markdown(f"**Risk level:** {selected['risk_level']}")
        st.markdown("**Why this trust score? · Calculated risk factors**")
        factors = selected["risk_factors"]
        if not factors:
            st.success("No configured risk factors were detected for this transaction.")
        else:
            for factor in factors:
                st.markdown(
                    f"- **{factor['label']} · {factor['severity']}** "
                    f"(−{factor['points']} trust points)  \n  {factor['evidence']}"
                )
                st.progress(min(100, int(factor["points"] * 100 / 30)))
        st.markdown(f"**Decision explanation:** {selected['reason']}")
        if selected["decision"] in {"DENY", "QUARANTINE"}:
            st.error(f"SECURITY ALERT · {selected['decision']}")


@st.cache_data(show_spinner=False)
def _compare_evaluation_cached(cache_token: str) -> dict[str, Any]:
    return compare_detectors(str(DEFAULT_DATA_PATH), str(GROUND_TRUTH_PATH))


def render_model_comparison() -> None:
    st.subheader("Model Comparison")
    st.caption(
        "Both implementations fit on the same training transactions, then evaluate the same holdout "
        "rows, feature representation, and labels."
    )
    if not DEFAULT_DATA_PATH.is_file() or not GROUND_TRUTH_PATH.is_file():
        st.info("Model performance comparison requires a labelled evaluation dataset.")
        return
    token = hashlib.sha256(
        DEFAULT_DATA_PATH.read_bytes() + GROUND_TRUTH_PATH.read_bytes()
    ).hexdigest()
    try:
        comparison = _compare_evaluation_cached(token)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        st.error(f"The labelled evaluation could not be completed: {exc}")
        return
    if comparison["ground_truth_matched_count"] == 0:
        st.info("Model performance comparison requires labelled evaluation data.")
        return

    metric_rows = []
    for label, summary_key, metrics_key in (
        ("Isolation Forest", "conventional_summary", "conventional_metrics"),
        ("Quantum-Inspired QSVM (prototype)", "quantum_summary", "quantum_metrics"),
    ):
        summary = comparison[summary_key]
        metrics = comparison[metrics_key]
        metric_rows.append({
            "Method": label,
            "Anomalies detected": summary["anomaly_count"],
            "False negatives": metrics["false_negatives"],
            "False positives": metrics["false_positives"],
            "Precision": f"{metrics['precision']:.3f}",
            "Recall": f"{metrics['recall']:.3f}",
            "F1 score": f"{metrics['f1_score']:.3f}",
        })
    st.caption(
        f"Training rows: {comparison['training_transaction_count']:,} · "
        f"Holdout rows: {comparison['evaluation_transaction_count']:,} · "
        f"Labelled holdout rows: {comparison['ground_truth_matched_count']:,}"
    )
    if not comparison["holdout_available"]:
        st.warning(
            "The evaluation dataset is too small for an independent holdout; these metrics are in-sample."
        )
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    conventional = comparison["conventional_metrics"]
    quantum = comparison["quantum_metrics"]
    if quantum["false_negatives"] < conventional["false_negatives"]:
        st.success(
            "On this evaluation dataset, the quantum-inspired prototype identified more labelled "
            "anomalies than Isolation Forest."
        )
    elif quantum["false_negatives"] > conventional["false_negatives"]:
        st.info(
            "On this evaluation dataset, Isolation Forest missed fewer labelled anomalies than the "
            "quantum-inspired prototype."
        )
    else:
        st.info("Both implementations had the same number of missed labelled anomalies on this dataset.")

    evaluation = comparison["evaluation_df"]
    missed = evaluation[
        evaluation["label"].eq("ANOMALY")
        & evaluation["conventional_label"].eq("NORMAL")
    ].copy()
    st.markdown("#### Isolation Forest Missed Cases")
    if missed.empty:
        st.success("No labelled anomalies were missed by Isolation Forest in this evaluation set.")
    else:
        missed["Explanation"] = missed.apply(
            lambda row: (
                "Both methods classified this case as normal."
                if row["quantum_label"] == "NORMAL"
                else "The quantum-inspired method classified this case as anomalous."
            ),
            axis=1,
        )
        st.dataframe(
            missed[[
                "transaction_id",
                "label",
                "conventional_label",
                "quantum_label",
                "amount",
                "hour",
                "is_new_device",
                "location",
                "device_id",
                "distance_from_usual",
                "failed_attempts",
                "Explanation",
            ]]
            .rename(columns={
                "transaction_id": "Transaction",
                "label": "Actual",
                "conventional_label": "Isolation Forest",
                "quantum_label": "Quantum-Inspired QSVM",
                "amount": "Amount",
                "hour": "Hour",
                "is_new_device": "New device",
                "location": "Location",
                "device_id": "Device",
                "distance_from_usual": "Distance from usual",
                "failed_attempts": "Failed attempts",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These are actual labelled evaluation rows where Isolation Forest returned NORMAL. "
            "The displayed transaction features are source values for those holdout rows; no feature-level "
            "causal explanation is inferred from model output alone."
        )


def _history_detection(history: pd.DataFrame) -> pd.Series:
    if "anomaly_status" in history:
        detection = history["anomaly_status"].astype("string").str.upper()
        return detection.fillna("NORMAL")
    conventional = history.get("conventional_label", pd.Series("", index=history.index)).astype("string").str.upper()
    quantum = history.get("quantum_label", pd.Series("", index=history.index)).astype("string").str.upper()
    return (conventional.eq("ANOMALY") | quantum.eq("ANOMALY")).map({True: "ANOMALY", False: "NORMAL"})


def render_history_section() -> None:
    st.subheader("Transaction History")
    try:
        history = load_history()
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"The transaction history could not be loaded: {exc}")
        return
    if history.empty:
        st.info("No transaction results have been saved yet.")
        return

    history = history.copy()
    history["detection"] = _history_detection(history)
    if "detection_method" not in history:
        history["detection_method"] = ""
    if "action" not in history:
        history["action"] = ""
    detection_filter, method_filter = st.columns(2)
    selected_detection = detection_filter.selectbox(
        "Detection",
        ["ALL", "NORMAL", "ANOMALY"],
        key="history_detection_filter",
    )
    selected_method = method_filter.selectbox(
        "Method",
        ["ALL", "ISOLATION FOREST", "QUANTUM-INSPIRED QSVM"],
        key="history_method_filter",
    )
    filtered = history
    if selected_detection != "ALL":
        filtered = filtered[filtered["detection"] == selected_detection]
    if selected_method != "ALL":
        filtered = filtered[
            filtered["detection_method"].astype(str).str.upper() == selected_method
        ]

    columns = [
        column for column in (
            "transaction_id",
            "account",
            "timestamp",
            "amount",
            "detection_method",
            "detection",
            "trust_score",
            "action",
        ) if column in filtered.columns
    ]
    st.caption(f"{len(filtered):,} individual transaction records.")
    history_display = filtered[columns].rename(columns={
            "transaction_id": "Transaction",
            "account": "Account",
            "timestamp": "Time",
            "amount": "Amount",
            "detection_method": "Method",
            "detection": "Detection",
            "trust_score": "Trust Score",
            "action": "Decision",
        })
    st.dataframe(
        history_display.style.map(_decision_cell_style, subset=["Decision"])
        if "Decision" in history_display
        else history_display,
        use_container_width=True,
        hide_index=True,
    )


def render_architecture() -> None:
    with st.expander("Qypher Security Architecture · PROTECT → DETECT → RESPOND"):
        protect, detect, respond = st.columns(3)
        protect.markdown(
            "**🛡 PROTECT**  \nPost-quantum-style signature simulation. "
            "A prototype only; not real ML-DSA."
        )
        detect.markdown(
            "**🔍 DETECT**  \nIsolation Forest baseline versus a quantum-inspired "
            "cosine-similarity prototype."
        )
        respond.markdown(
            "**⚡ RESPOND**  \nTransaction-specific trust score and ALLOW, MONITOR, "
            "DENY, or QUARANTINE decision."
        )


def main() -> None:
    st.set_page_config(page_title="QYPHER · Account Monitoring", page_icon="🛡️", layout="wide")
    render_styles()
    render_header(st.session_state.get("analysis_result"))

    st.markdown("## Analyze Transactions")
    st.caption("Upload a transaction CSV to analyze every transaction individually.")
    uploaded_file = st.file_uploader(
        "UPLOAD CSV",
        type=["csv"],
        help="Each transaction receives its own detector result, trust score, and decision.",
    )
    csv_path, transactions = safe_load_transactions(uploaded_file)
    if transactions is None:
        st.stop()

    token = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
    file_name = uploaded_file.name if uploaded_file is not None else csv_path.name
    st.markdown(
        f"**File:** `{file_name}`　·　**Transactions detected:** {len(transactions):,}"
    )
    st.caption("Required transaction fields: " + ", ".join(REQUIRED_COLUMNS))
    run_analysis = st.button("RUN ANALYSIS", type="primary", use_container_width=True)
    if run_analysis:
        st.session_state["analysis_ready_token"] = token
        st.session_state.pop("analysis_result", None)
        st.rerun()

    analysis = st.session_state.get("analysis_result")
    if analysis is not None and analysis.get("cache_token") != token:
        analysis = None
    if st.session_state.get("analysis_ready_token") == token:
        selected_model = render_method_choices()
        if selected_model is not None:
            progress = st.progress(0, text=f"Analyzing transactions... 0 / {len(transactions):,}")

            def update_progress(completed: int, total: int) -> None:
                progress.progress(
                    completed / total,
                    text=f"Analyzing transaction {completed:,} / {total:,}",
                )

            try:
                with st.spinner(f"Running {MODEL_OPTIONS[selected_model]['label']} on each transaction..."):
                    analysis = analyze_transactions(
                        str(csv_path),
                        selected_model,
                        token,
                        progress_callback=update_progress,
                    )
                if analysis["input_row_count"] != analysis["result_row_count"]:
                    raise ValueError("The analysis did not return exactly one result for every CSV row.")
                _persist_analysis(analysis, token)
                st.session_state["analysis_result"] = analysis
                st.session_state["analysis_ready_token"] = token

                alert_key = f"{token}:{selected_model}"
                anomalous = analysis["monitor_df"][analysis["monitor_df"]["detection"].eq("ANOMALY")]
                if not anomalous.empty and st.session_state.get("alerted_key") != alert_key:
                    trigger_sound_alert()
                    st.session_state["alerted_key"] = alert_key
                st.rerun()
            except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
                st.error(f"Transaction analysis could not be completed: {exc}")

    if analysis is not None:
        render_analysis_results(analysis)
    elif st.session_state.get("analysis_ready_token") != token:
        st.info("Select RUN ANALYSIS to choose a detection method for this transaction file.")

    st.divider()
    render_model_comparison()
    st.divider()
    render_history_section()
    render_architecture()
    st.caption(
        "Student demonstration prototype. The quantum-inspired model is classical cosine-similarity "
        "scoring, not a quantum computer or a real QSVM. No production security or fraud-detection "
        "guarantees are claimed."
    )


if __name__ == "__main__":
    main()
