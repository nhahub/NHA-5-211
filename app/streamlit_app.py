"""Smart IDS v3 — Streamlit Operational SOC Dashboard.

Interactive Security Operations Center (SOC) dashboard providing:
- Multi-model real-time inference (MLP, 1D-CNN, XGBoost, Autoencoder).
- Live flow inspection and detection filtering.
- Visual threat analytics and attack-to-port heatmaps.
- On-demand SHAP feature attribution waterfall plots.
- Active response mitigation simulation with injection-safe, idempotent iptables rules.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src and app are on Python path
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from smart_ids.config import (
    CLASS_NAMES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RATE_LIMIT_CEILING,
    SEED,
    THREAT_ACTIONS,
    THREAT_COLORS,
    THREAT_ICONS,
    THREAT_THRESHOLDS,
)
from smart_ids.data_loader import standardize_labels
from smart_ids.artifacts import (
    load_all_artifacts,
    load_feature_names,
    load_scaler,
)
from smart_ids.evaluation import compute_reconstruction_error
from smart_ids.explain import generate_waterfall_plot
from smart_ids.response import (
    BLOCKED_IPS,
    apply_active_response,
    get_threat_level,
    validate_ip,
)
from components import (
    PLOTLY_DARK_TEMPLATE,
    get_threat_badge_html,
    inject_custom_css,
    render_kpi_card,
)

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Smart IDS v3 Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()


# --- Cached Resource and Model Loading ---
@st.cache_resource(show_spinner="Loading trained models and encoders from disk...")
def get_loaded_artifacts() -> Dict[str, Any]:
    """Load trained models, scaler, label encoder, and threshold metadata."""
    return load_all_artifacts(MODELS_DIR)


@st.cache_data(show_spinner="Generating demo network flow telemetry...")
def generate_demo_logs(n_records: int = 500) -> pd.DataFrame:
    """Generate representative synthetic SOC detection logs for built-in demo mode."""
    rng = np.random.default_rng(SEED)

    # Class distribution mirroring empirical weights
    classes = CLASS_NAMES
    p_weights = [0.75, 0.08, 0.05, 0.04, 0.02, 0.02, 0.02, 0.02]

    assigned_classes = rng.choice(classes, size=n_records, p=p_weights)

    # Generate source IPs with a few recurring attackers
    known_attackers = ["198.51.100.24", "203.0.113.88", "192.0.2.145", "198.51.100.99"]
    src_ips = []
    dst_ports = []
    confidences = []

    for cls in assigned_classes:
        if cls == "BENIGN":
            src_ips.append(f"192.168.1.{rng.integers(10, 240)}")
            dst_ports.append(int(rng.choice([80, 443, 53, 8080])))
            confidences.append(float(rng.uniform(0.10, 0.38)))
        else:
            if rng.random() < 0.6:
                src_ips.append(str(rng.choice(known_attackers)))
            else:
                src_ips.append(f"203.0.113.{rng.integers(10, 250)}")

            if cls == "FTP-Patator":
                dst_ports.append(21)
            elif cls == "SSH-Patator":
                dst_ports.append(22)
            elif cls in ("DoS Hulk", "DoS GoldenEye", "Web Attack"):
                dst_ports.append(80)
            elif cls == "PortScan":
                dst_ports.append(int(rng.choice([21, 22, 23, 25, 80, 443, 3389, 8080])))
            else:
                dst_ports.append(int(rng.choice([80, 443, 8080])))

            confidences.append(float(rng.uniform(0.65, 0.99)))

    timestamps = pd.date_range(end=datetime.now(), periods=n_records, freq="15s").strftime("%Y-%m-%d %H:%M:%S")

    df_logs = pd.DataFrame({
        "timestamp": timestamps,
        "src_ip": src_ips,
        "dst_port": dst_ports,
        "attack_class": assigned_classes,
        "confidence": confidences,
    })

    # Resolve threat level and active response action
    df_logs["threat_level"] = df_logs["confidence"].apply(get_threat_level)
    df_logs["action"] = df_logs["threat_level"].map(THREAT_ACTIONS)
    df_logs["is_anomaly"] = df_logs["attack_class"] != "BENIGN"

    return df_logs


# --- Sidebar Control Surface ---
artifacts = get_loaded_artifacts()

with st.sidebar:
    st.markdown("## 🛡️ Smart IDS v3")
    st.caption("AI-Powered Network Intrusion Defense")
    st.divider()

    st.markdown("### ⚙️ Engine Settings")
    active_model_name = st.selectbox(
        "Active Multi-Class Model",
        options=["MLP", "CNN", "XGBoost"],
        index=0,
        help="Select which supervised neural or tree architecture provides multi-class predictions.",
    )

    conf_slider = st.slider(
        "Minimum Confidence Filter",
        min_value=0.0,
        max_value=1.0,
        value=0.40,
        step=0.05,
        help="Filters the displayed telemetry in the Detection Log to flows exceeding this confidence.",
    )

    dry_run_mode = st.toggle(
        "Dry-Run Mode (Safe Simulation)",
        value=True,
        help="When enabled, firewall blocking rules are synthesized as text only without touching the host OS.",
    )

    # Anomaly Threshold Display
    thresh_data = artifacts.get("threshold_data", {})
    ae_threshold = thresh_data.get("anomaly_threshold", 0.0452)
    st.metric(
        "Autoencoder Anomaly Cutoff",
        f"{ae_threshold:.4f}",
        help="Calibrated 95th percentile reconstruction error threshold on benign validation flows.",
    )

    st.divider()
    st.markdown("### 📡 Telemetry Source")
    data_source_mode = st.radio(
        "Data Mode",
        options=["Built-in Demo Mode", "Upload Custom CSV"],
        index=0,
        help="Choose between simulated real-time logs or your own exported flow CSV.",
    )

    st.divider()
    st.markdown("### 🔍 Model Status")
    for m in ["MLP", "CNN", "XGBoost", "Autoencoder"]:
        status = "🟢 Loaded" if m in artifacts else "🔴 Not Found"
        st.write(f"- **{m}**: {status}")

    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


def normalize_uploaded_logs(
    df_raw: pd.DataFrame,
    artifacts: Dict[str, Any],
    active_model_name: str,
) -> pd.DataFrame:
    """Normalize and enrich an uploaded network flow CSV with model inference and SOC columns."""
    df = df_raw.copy()
    # 1. Clean column headers
    df.columns = df.columns.str.strip()

    # 2. Extract or synthesize metadata identifiers
    # Source IP
    for c in ["Source IP", "source_ip", "src_ip", "Src IP", "SourceIP"]:
        if c in df.columns:
            df["src_ip"] = df[c]
            break
    if "src_ip" not in df.columns:
        df["src_ip"] = [f"192.168.10.{i % 200 + 1}" for i in range(len(df))]

    # Destination Port
    for c in ["Destination Port", "destination_port", "dst_port", "Dst Port", "DestinationPort"]:
        if c in df.columns:
            df["dst_port"] = df[c]
            break
    if "dst_port" not in df.columns:
        df["dst_port"] = 80

    # Timestamp
    for c in ["Timestamp", "timestamp", "Time", "time", "Date"]:
        if c in df.columns:
            df["timestamp"] = df[c]
            break
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.date_range(end=datetime.now(), periods=len(df), freq="1s").strftime("%Y-%m-%d %H:%M:%S")

    # 3. Resolve Attack Taxonomy Category
    label_col = None
    for c in ["Label", "label", "attack_class", "Attack", "attack", "Class", "class"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is not None:
        df["attack_class"] = df[label_col].astype(str).str.strip().map(
            lambda x: standardize_labels(x) or ("BENIGN" if "benign" in str(x).lower() else (x if x in CLASS_NAMES else "BENIGN"))
        )
    else:
        df["attack_class"] = "BENIGN"

    # 4. Attempt real-time ML inference if full feature set is present
    feat_names = artifacts.get("feature_names")
    scaler = artifacts.get("scaler")
    model = artifacts.get(active_model_name)
    ae_model = artifacts.get("Autoencoder")
    thresh_data = artifacts.get("threshold_data", {})
    ae_threshold = thresh_data.get("anomaly_threshold", 0.0452)

    can_predict = (
        feat_names is not None
        and scaler is not None
        and model is not None
        and all(f in df.columns for f in feat_names)
    )

    if can_predict:
        # Evaluate up to 5000 rows for fast, interactive response
        eval_size = min(5000, len(df))
        df_eval = df.iloc[:eval_size].copy()
        X_sub = df_eval[feat_names].replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X_sub)

        try:
            if active_model_name == "CNN":
                cnn_in = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
                probs = model.predict(cnn_in, verbose=0, batch_size=512)
            elif active_model_name == "XGBoost":
                probs = model.predict_proba(X_scaled)
            else:
                probs = model.predict(X_scaled, verbose=0, batch_size=512)

            pred_indices = np.argmax(probs, axis=1)
            pred_classes = [CLASS_NAMES[idx] for idx in pred_indices]
            confidences = np.max(probs, axis=1)

            # Autoencoder zero-day reconstruction check
            if ae_model is not None:
                recon = ae_model.predict(X_scaled, verbose=0, batch_size=512)
                mse = np.mean(np.square(X_scaled - recon), axis=1)
                is_ae_anomaly = mse > ae_threshold
            else:
                is_ae_anomaly = [c != "BENIGN" for c in pred_classes]

            df_eval["attack_class"] = pred_classes
            df_eval["confidence"] = confidences.astype(float)
            df_eval["is_anomaly"] = is_ae_anomaly
            df = df_eval
        except Exception:
            pass

    # 5. Populate fallback columns if still missing
    if "confidence" not in df.columns:
        df["confidence"] = df["attack_class"].map(lambda x: 0.95 if x == "BENIGN" else 0.88).astype(float)
    if "threat_level" not in df.columns:
        df["threat_level"] = df["confidence"].apply(get_threat_level)
    if "action" not in df.columns:
        df["action"] = df["threat_level"].map(THREAT_ACTIONS)
    if "is_anomaly" not in df.columns:
        df["is_anomaly"] = df["attack_class"] != "BENIGN"

    return df


# --- Ingest or Generate Logs ---
if data_source_mode == "Upload Custom CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Network Flow CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_csv_df = pd.read_csv(uploaded_file)
            df_active_logs = normalize_uploaded_logs(raw_csv_df, artifacts, active_model_name)
            display_count = len(df_active_logs)
            total_raw = len(raw_csv_df)
            if display_count < total_raw:
                st.sidebar.success(f"Loaded {total_raw:,} rows (showing top {display_count:,} for interactive speed).")
            else:
                st.sidebar.success(f"Loaded {display_count:,} rows from CSV.")
        except Exception as e:
            st.sidebar.error(f"Error parsing CSV: {e}")
            df_active_logs = generate_demo_logs()
    else:
        st.info("Awaiting CSV upload. Displaying built-in demo telemetry in the meantime.")
        df_active_logs = generate_demo_logs()
else:
    df_active_logs = generate_demo_logs()



# --- Main Dashboard Header Banner ---
st.markdown("""
<div class="soc-header">
    <h1>🛡️ Smart IDS v3 — Security Operations Center</h1>
    <p>Real-Time Hybrid Multi-Modal Intrusion Detection & Zero-Day Anomaly Tracking</p>
</div>
""", unsafe_allow_html=True)


# --- Headline KPI Metrics Strip ---
total_flows = len(df_active_logs)
attack_flows = int((df_active_logs["attack_class"] != "BENIGN").sum()) if "attack_class" in df_active_logs.columns else 0
critical_threats = int((df_active_logs["threat_level"] == "Critical").sum())
anomaly_flows = int(df_active_logs.get("is_anomaly", pd.Series([False] * total_flows)).sum())
avg_confidence = float(df_active_logs["confidence"].mean())

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    render_kpi_card("Total Flows Analyzed", f"{total_flows:,}", "Incoming network conversations")
with c2:
    render_kpi_card("Attacks Detected", f"{attack_flows:,}", f"{attack_flows / max(1, total_flows) * 100:.1f}% detection rate")
with c3:
    render_kpi_card("Zero-Day Anomalies", f"{anomaly_flows:,}", "Reconstruction outliers")
with c4:
    render_kpi_card("Critical Severity", f"{critical_threats:,}", "Immediate blocking recommended", badge_html=get_threat_badge_html("Critical"))
with c5:
    render_kpi_card("Mean Confidence", f"{avg_confidence * 100:.1f}%", f"Active: {active_model_name}")

st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)


# --- 5 Interactive Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Detection Log",
    "📊 Analytics",
    "🗺️ Threat Map",
    "🔎 Predict Flow",
    "⚡ Active Response",
])


# ==============================================================================
# TAB 1: DETECTION LOG
# ==============================================================================
with tab1:
    st.markdown("##### 📋 Filterable Real-Time Flow Telemetry")
    st.caption("Plain English: This table lists all network conversations evaluated by the IDS. Filter by threat tier or attack type.")

    # Filter Controls
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    with fc1:
        selected_tiers = st.multiselect(
            "Filter Threat Severity",
            options=["Low", "Medium", "High", "Critical"],
            default=["Medium", "High", "Critical"],
            help="Low-confidence benign traffic is excluded by default to focus on actionable threats.",
        )
    with fc2:
        selected_classes = st.multiselect(
            "Filter Attack Taxonomy",
            options=CLASS_NAMES,
            default=CLASS_NAMES,
            help="Filter by specific attack categories or Benign traffic.",
        )
    with fc3:
        filter_conf = st.slider(
            "Filter Confidence Above",
            0.0, 1.0, float(conf_slider), 0.05,
            key="tab1_conf_slider",
        )

    # Apply Filters
    if not selected_tiers or not selected_classes:
        st.warning("⚠️ Please select at least one threat severity tier and attack taxonomy category above.")
        df_filtered = pd.DataFrame(columns=df_active_logs.columns)
    else:
        mask = (
            df_active_logs["threat_level"].isin(selected_tiers)
            & df_active_logs["attack_class"].isin(selected_classes)
            & (df_active_logs["confidence"] >= filter_conf)
        )
        df_filtered = df_active_logs[mask].copy()

    # Add visual badge column
    df_display = df_filtered.copy()
    badge_map = {"Low": "🟢 LOW", "Medium": "🟡 MEDIUM", "High": "🟠 HIGH", "Critical": "🔴 CRITICAL"}
    df_display["threat_badge"] = df_display["threat_level"].map(badge_map)

    # Reorder display columns
    display_cols = ["timestamp", "src_ip", "dst_port", "attack_class", "confidence", "threat_badge", "action"]
    avail_cols = [c for c in display_cols if c in df_display.columns]

    st.dataframe(
        df_display[avail_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "timestamp": st.column_config.TextColumn("Timestamp"),
            "src_ip": st.column_config.TextColumn("Source IP"),
            "dst_port": st.column_config.NumberColumn("Target Port", format="%d"),
            "attack_class": st.column_config.TextColumn("Predicted Class"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "threat_badge": st.column_config.TextColumn("Threat Tier"),
            "action": st.column_config.TextColumn("Engine Action"),
        },
    )

    # Export Download Button
    csv_bytes = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Filtered Telemetry (CSV)",
        data=csv_bytes,
        file_name=f"smart_ids_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


# ==============================================================================
# TAB 2: ANALYTICS
# ==============================================================================
with tab2:
    st.markdown("##### 📊 Network Behavioral Analytics & Benchmark Comparison")
    st.caption("Plain English: High-level visual breakdown of detected attack types, threat tiers, and model evaluation benchmarks.")

    a_col1, a_col2 = st.columns(2)

    with a_col1:
        # Donut Chart: Attack Class Breakdown
        class_counts = df_active_logs["attack_class"].value_counts().reset_index()
        class_counts.columns = ["Attack Class", "Count"]

        fig_pie_class = px.pie(
            class_counts,
            values="Count",
            names="Attack Class",
            hole=0.45,
            title="Proportional Breakdown by Attack Category",
            color_discrete_sequence=px.colors.qualitative.Dark24,
        )
        fig_pie_class.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"])
        st.plotly_chart(fig_pie_class, use_container_width=True)

    with a_col2:
        # Donut Chart: Threat Level Breakdown
        tier_counts = df_active_logs["threat_level"].value_counts().reset_index()
        tier_counts.columns = ["Threat Tier", "Count"]

        fig_pie_tier = px.pie(
            tier_counts,
            values="Count",
            names="Threat Tier",
            hole=0.55,
            title="Proportional Breakdown by Threat Severity Tier",
            color="Threat Tier",
            color_discrete_map=THREAT_COLORS,
        )
        fig_pie_tier.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"])
        st.plotly_chart(fig_pie_tier, use_container_width=True)

    # Confidence Distribution Histogram
    fig_hist = px.histogram(
        df_active_logs,
        x="confidence",
        color="threat_level",
        nbins=40,
        title="Confidence Score Distribution Partitioned by Threat Severity",
        labels={"confidence": "Prediction Confidence", "threat_level": "Threat Tier"},
        color_discrete_map=THREAT_COLORS,
    )
    fig_hist.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"])
    st.plotly_chart(fig_hist, use_container_width=True)

    # Benchmark Comparison Table
    st.markdown("#### 🏆 Benchmark Evaluation Matrix (Held-Out Test Set)")
    comp_df = artifacts.get("comparison_df")
    if comp_df is not None:
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("Note: Supervised models are evaluated on 8-class weighted metrics; Autoencoder is evaluated on binary anomaly detection.")
    else:
        st.info("Run Notebook 05 to generate and populate `models/model_comparison.csv`.")


# ==============================================================================
# TAB 3: THREAT MAP
# ==============================================================================
with tab3:
    st.markdown("##### 🗺️ Infrastructure & Attacker Surface Analysis")
    st.caption("Plain English: Shows which external computers (Source IPs) are attacking most frequently, and which network ports are being targeted.")

    tm_col1, tm_col2 = st.columns(2)

    with tm_col1:
        # Top 15 Attacking Source IPs
        attack_only_df = df_active_logs[df_active_logs["attack_class"] != "BENIGN"]
        if not attack_only_df.empty:
            ip_counts = (
                attack_only_df.groupby("src_ip")
                .agg(attack_count=("attack_class", "count"), max_conf=("confidence", "max"))
                .reset_index()
                .sort_values(by="attack_count", ascending=True)
                .tail(15)
            )

            fig_ip = px.bar(
                ip_counts,
                x="attack_count",
                y="src_ip",
                orientation="h",
                color="max_conf",
                color_continuous_scale="Reds",
                title="Top 15 Malicious Source IPs by Attack Volume",
                labels={"attack_count": "Number of Malicious Flows", "src_ip": "Attacker IP", "max_conf": "Max Conf"},
            )
            fig_ip.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"])
            st.plotly_chart(fig_ip, use_container_width=True)
        else:
            st.info("No attack flows detected in active log.")

    with tm_col2:
        # Top Targeted Destination Ports
        port_counts = (
            df_active_logs["dst_port"].value_counts().reset_index()
        )
        port_counts.columns = ["Target Port", "Flow Count"]
        port_counts["Target Port"] = port_counts["Target Port"].astype(str)

        fig_ports = px.bar(
            port_counts.head(12),
            x="Target Port",
            y="Flow Count",
            color="Flow Count",
            color_continuous_scale="Blues",
            title="Top Targeted Infrastructure Services (Destination Ports)",
            labels={"Target Port": "Port Number", "Flow Count": "Flow Volume"},
        )
        fig_ports.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"])
        st.plotly_chart(fig_ports, use_container_width=True)

    # Attack Class vs Target Port Cross-Tabulation
    st.markdown("#### 🔬 Attack Class vs Destination Port Cross-Tabulation")
    if "dst_port" in attack_only_df.columns and not attack_only_df.empty:
        top_ports = attack_only_df["dst_port"].value_counts().head(8).index
        sub_attacks = attack_only_df[attack_only_df["dst_port"].isin(top_ports)]
        pivot = pd.crosstab(sub_attacks["attack_class"], sub_attacks["dst_port"])

        fig_heat = px.imshow(
            pivot,
            text_auto=True,
            color_continuous_scale="Viridis",
            title="Attack Category Concentration Across Targeted Ports",
            labels=dict(x="Destination Port", y="Attack Family", color="Flows"),
        )
        fig_heat.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"])
        st.plotly_chart(fig_heat, use_container_width=True)


# ==============================================================================
# TAB 4: PREDICT FLOW
# ==============================================================================
with tab4:
    st.markdown("##### 🔎 Live Multi-Model Inference & SHAP Forensic Attribution")
    st.caption("Plain English: Pick a network flow to test all models live. Get the predicted attack category, zero-day anomaly score, and a SHAP explanation.")

    # Load test data and scaler
    scaler = artifacts.get("scaler")
    feat_names = artifacts.get("feature_names")
    test_path = PROCESSED_DATA_DIR / "X_test_scaled.npy"
    y_test_path = PROCESSED_DATA_DIR / "y_test.npy"

    if test_path.exists() and feat_names is not None:
        X_test_arr = np.load(test_path)
        y_test_arr = np.load(y_test_path)

        p_c1, p_c2 = st.columns([1, 2])
        with p_c1:
            sample_index = st.number_input(
                "Select Test Flow Index",
                min_value=0,
                max_value=len(X_test_arr) - 1,
                value=42,
                step=1,
                help="Choose a row from the held-out evaluation test set.",
            )
            ground_truth_label = CLASS_NAMES[y_test_arr[sample_index]]
            st.info(f"**Ground Truth Label:** `{ground_truth_label}`")

        single_vector = X_test_arr[sample_index : sample_index + 1]

        # Execute Live Inference
        active_model = artifacts.get(active_model_name)
        ae_model = artifacts.get("Autoencoder")

        if active_model is not None and ae_model is not None:
            # 1. Supervised Prediction
            if active_model_name == "CNN":
                cnn_input = single_vector.reshape(1, single_vector.shape[1], 1)
                probs = active_model.predict(cnn_input, verbose=0)[0]
            elif active_model_name == "XGBoost":
                probs = active_model.predict_proba(single_vector)[0]
            else:  # MLP
                probs = active_model.predict(single_vector, verbose=0)[0]

            pred_class_idx = int(np.argmax(probs))
            pred_class_name = CLASS_NAMES[pred_class_idx]
            conf = float(probs[pred_class_idx])
            threat_tier = get_threat_level(conf)

            # 2. Autoencoder Anomaly Score
            recon_err = float(compute_reconstruction_error(ae_model, single_vector)[0])
            is_anomaly = recon_err > ae_threshold

            # Display Verdict
            v1, v2, v3, v4 = st.columns(4)
            with v1:
                st.metric("Predicted Class", pred_class_name)
            with v2:
                st.metric("Model Confidence", f"{conf * 100:.2f}%")
            with v3:
                st.markdown(f"**Threat Severity**<br>{get_threat_badge_html(threat_tier)}", unsafe_allow_html=True)
            with v4:
                ae_badge = "🔴 ANOMALOUS (Zero-Day)" if is_anomaly else "🟢 NORMAL"
                st.metric("Autoencoder Verdict", ae_badge, f"MSE: {recon_err:.4f} (Cutoff: {ae_threshold:.4f})")

            st.divider()

            # Class Probability Bar Chart
            prob_df = pd.DataFrame({"Class": CLASS_NAMES, "Probability": probs})
            fig_prob = px.bar(
                prob_df,
                x="Class",
                y="Probability",
                title=f"Prediction Probability Distribution ({active_model_name})",
                color="Probability",
                color_continuous_scale="Teal",
            )
            fig_prob.update_layout(template="plotly_dark", **PLOTLY_DARK_TEMPLATE["layout"], yaxis_range=[0, 1])
            st.plotly_chart(fig_prob, use_container_width=True)

            # SHAP Forensic Attribution
            st.markdown("#### 🔬 Explainable AI: SHAP Feature Attribution")
            st.caption("Shows which specific network statistics drove the model's confidence toward this prediction.")

            xgb_m = artifacts.get("XGBoost")
            if xgb_m is not None:
                with st.spinner("Generating SHAP waterfall decomposition..."):
                    try:
                        import shap
                        explainer_xgb = shap.TreeExplainer(xgb_m)
                        fig_wf = generate_waterfall_plot(
                            explainer_xgb,
                            single_vector[0],
                            feat_names,
                            target_class_idx=pred_class_idx if pred_class_idx != 0 else 1,
                            max_display=10,
                        )
                        st.pyplot(fig_wf)
                    except Exception as ex:
                        st.warning(f"Could not render dynamic SHAP plot: {ex}")
        else:
            st.warning("Please train the models via Notebooks 03 and 04 to enable live inference.")
    else:
        st.info("Test dataset arrays not found. Please run Notebook 02 to generate `data/processed/` splits.")


# ==============================================================================
# TAB 5: ACTIVE RESPONSE
# ==============================================================================
with tab5:
    st.markdown("##### ⚡ Automated Mitigation Engine & Dry-Run Firewall Blocking")
    st.caption("Plain English: Connects AI detections to automated defense. Translates confidence into graduated actions: LOG, ALERT, RATE_LIMIT, or BLOCK.")

    # Mitigation Reference Panel
    st.markdown("""
    | Threat Tier | Confidence Band | Engine Action | Operational Mitigation Behavior |
    | :--- | :--- | :--- | :--- |
    | **🟢 Low** | 0.00 – 0.40 | `LOG` | Logged for statistical analysis; zero network disruption. |
    | **🟡 Medium** | 0.40 – 0.70 | `ALERT` | Security alert dispatched to SIEM and SOC dashboard. |
    | **🟠 High** | 0.70 – 0.90 | `RATE_LIMIT` | Throttled to 100 pps ceiling to prevent bandwidth saturation. |
    | **🔴 Critical** | 0.90 – 1.01 | `BLOCK` | Source IP immediately blocked via `iptables -I INPUT 1 -s <IP> -j DROP`. |
    """)

    st.divider()
    st.markdown("#### 🧪 Manual Incident Mitigation Trigger")

    m_c1, m_c2, m_c3 = st.columns([1, 1, 1])
    with m_c1:
        input_ip = st.text_input("Candidate Source IP", value="198.51.100.42", help="IP address to evaluate and potentially block.")
    with m_c2:
        input_class = st.selectbox("Detected Attack Class", options=CLASS_NAMES, index=1)
    with m_c3:
        input_conf = st.slider("Confidence Level", 0.0, 1.0, 0.95, 0.01)

    trigger_btn = st.button("🚀 Trigger Active Mitigation Decision", type="primary")

    if trigger_btn:
        res = apply_active_response(
            src_ip=input_ip,
            attack_class=input_class,
            confidence=input_conf,
            model_name=active_model_name,
            dry_run=dry_run_mode,
        )

        st.markdown(f"### Mitigation Result: `{res['action']}`")
        r_c1, r_c2, r_c3 = st.columns(3)
        with r_c1:
            st.metric("Threat Tier", res["threat_level"])
        with r_c2:
            st.metric("IP Validation", "✅ PASSED" if validate_ip(input_ip) else "❌ REJECTED (INJECTION)")
        with r_c3:
            st.metric("Firewall Blocked Status", "True (Simulated)" if res["blocked"] else "False")

        st.code(res["action_detail"], language="bash")

    # Idempotent Blocked IPs List
    st.divider()
    st.markdown("#### 📋 Active Blocked IP Registry (In-Memory Idempotency Cache)")
    st.caption("Guarantees duplicate block requests for the same attacker do not bloat firewall chains.")

    if BLOCKED_IPS:
        blocked_df = pd.DataFrame(list(BLOCKED_IPS), columns=["Blocked IP Address"])
        st.table(blocked_df)
    else:
        st.write("No IPs currently blocked in this session.")
