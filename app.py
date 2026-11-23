import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve
)
st.set_page_config(
    page_title="Sri Lanka Flood Risk Analytics", layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent
@st.cache_data
def load_data():
    data = pd.read_csv(BASE_DIR / "cleaned_flood_data.csv",
                       parse_dates=["date"])
    results = pd.read_csv(BASE_DIR / "model_results.csv")
    predictions = pd.read_csv(BASE_DIR / "model_predictions.csv")
    importance = pd.read_csv(BASE_DIR / "feature_importance.csv")
    return data, results, predictions, importance
@st.cache_resource
def load_models():

    from xgboost import XGBClassifier

    models = joblib.load(
        BASE_DIR / "trained_models_portable.pkl"
    )

    # Restore standalone XGBoost
    xgb = XGBClassifier()

    xgb.load_model(
        str(BASE_DIR / "xgboost_model.json")
    )

    models["XGBoost"].steps[-1] = ("model", xgb)

    # Restore XGBoost inside stacking
    stacking_xgb = XGBClassifier()

    stacking_xgb.load_model(
        str(BASE_DIR / "stacking_xgboost.json")
    )

    models["Stacking"].named_estimators_["xgb"].steps[-1] = (
        "model", stacking_xgb
    )

    return models
df, results, predictions, importance = load_data()
models = load_models()
best = results.sort_values("F1 Score", ascending=False).iloc[0]
st.sidebar.title("Flood Risk Analytics")
st.sidebar.caption("Sri Lanka | 2015-2024")
page = st.sidebar.radio("Navigation", [
      "Overview", "Climate Analysis", "District Analysis",
    "Ensemble Comparison", "Hybrid Analysis", "Model Evaluation",
    "Risk Classification"
])

# PAGE 1: OVERVIEW
if page == "Overview":
    st.title("Sri Lanka Flood Risk Analytics Dashboard")
    st.subheader("Ensemble and Hybrid Machine Learning")
    st.write("Daily climatic observations across 25 districts.")
    a, b, c, d = st.columns(4)
    a.metric("Total Records", f"{len(df):,}")
    b.metric("Districts", df["district"].nunique())
    c.metric("Elevated Risk", f"{df['flood_risk'].mean():.2%}")
    d.metric("Best Model", best["Model"])
    left, right = st.columns(2)
    with left:
        distribution = df["risk_label"].value_counts().rename_axis(
            "Risk").reset_index(name="Count")
        fig = px.pie(distribution, names="Risk", values="Count",
                     title="Risk distribution", hole=0.5)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(results.sort_values("F1 Score"),
                     x="F1 Score", y="Model", orientation="h",
                     title="Model performance", color="F1 Score")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head(100), use_container_width=True)
# PAGE 2: CLIMATE
elif page == "Climate Analysis":
    st.title("Climate Analysis")
    districts = st.multiselect(
        "Select Districts", sorted(df["district"].unique()),
        default=["Colombo", "Gampaha", "Kandy"]
    )
    years = st.slider("Year range", 2015, 2024, (2015, 2024))
    filtered = df[df["district"].isin(districts)
                  & df["year"].between(*years)]
    if filtered.empty:
        st.warning("Select at least one district.")
        st.stop()
    monthly = filtered.groupby(
        ["month", "district"], as_index=False
    )["rain_24h"].mean()
    fig = px.line(monthly, x="month", y="rain_24h",
                  color="district", markers=True,
                  title="Average monthly daily rainfall")
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    with a:
        st.plotly_chart(px.box(filtered, x="risk_label",
            y="rain_72h", color="risk_label",
            title="Rainfall vs risk"), use_container_width=True)
    with b:
        st.plotly_chart(px.box(filtered, x="risk_label",
            y="soil_moisture_0_to_7cm_mean", color="risk_label",
            title="Soil moisture vs risk"), use_container_width=True)
# PAGE 3: DISTRICTS
elif page == "District Analysis":
    st.title("District-Level Flood Risk Analysis")
    zone = st.selectbox("Climatic Zone",
        ["All"] + sorted(df["climatic_zone"].unique()))
    filtered = df if zone == "All" else df[df["climatic_zone"] == zone]
    summary = filtered.groupby("district", as_index=False).agg(
        risk_rate=("flood_risk", "mean"),
        average_rainfall=("rain_24h", "mean"),
        latitude=("latitude", "first"),
          longitude=("longitude", "first"),
        records=("flood_risk", "size")
    )
    summary["Risk Percentage"] = summary["risk_rate"] * 100
    fig = px.bar(summary.sort_values("Risk Percentage",
                                    ascending=False),
                 x="district", y="Risk Percentage",
                 color="Risk Percentage",
                 title="Elevated modeled risk by district")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter_geo(summary, lat="latitude", lon="longitude",
        hover_name="district", color="Risk Percentage",
        size="Risk Percentage", color_continuous_scale="YlOrRd")
    fig.update_geos(projection_type="mercator",
        lataxis_range=[5.5, 10], lonaxis_range=[79, 82.5],
        showland=True, showcountries=True)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(summary, use_container_width=True)
# PAGE 4: ENSEMBLE COMPARISON
elif page == "Ensemble Comparison":
    st.title("Ensemble Model Comparison")
    metric = st.selectbox("Metric", [
        "Accuracy", "Precision", "Recall", "F1 Score",
        "ROC AUC", "Average Precision"
    ], index=3)
    selected = st.multiselect("Models", results["Model"].tolist(),
                              default=results["Model"].tolist())
    filtered = results[results["Model"].isin(selected)]
    if filtered.empty:
        st.warning("Select at least one model.")
        st.stop()
    st.plotly_chart(px.bar(filtered.sort_values(metric),
        x=metric, y="Model", orientation="h", color="Model"),
        use_container_width=True)
    st.subheader("Random Forest vs gradient boosting")
    comparison = results[results["Model"].isin([
        "Random Forest", "XGBoost", "LightGBM", "CatBoost"
    ])]
    fig = px.bar(comparison, x="Model",
        y=["Precision", "Recall", "F1 Score", "ROC AUC"],
        barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    winner = comparison.loc[comparison["F1 Score"].idxmax()]
    st.success(f"RF/boosting winner: {winner['Model']} "
               f"(F1: {winner['F1 Score']:.4f})")
    st.dataframe(results.round(4), use_container_width=True)

# PAGE 5: HYBRID ANALYSIS
elif page == "Hybrid Analysis":
    st.title("Random Forest + ANN Hybrid Analysis")
    st.write("Soft voting averages the two predicted probabilities.")
    st.latex(r"P_{Hybrid} = \frac{P_{RF}+P_{ANN}}{2}")
    hybrid = results[results["Model"].isin([
        "Random Forest", "ANN", "RF + ANN Hybrid"
    ])]
    fig = px.bar(hybrid, x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score"],
        barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(hybrid.round(4), use_container_width=True)
    score = hybrid.loc[hybrid["Model"] == "RF + ANN Hybrid",
                       "F1 Score"].iloc[0]
    standalone = hybrid.loc[hybrid["Model"] != "RF + ANN Hybrid",
                            "F1 Score"].max()
    st.metric("Hybrid F1-score", f"{score:.4f}",
              delta=f"{score - standalone:+.4f} vs best standalone")

# PAGE 6: MODEL EVALUATION
elif page == "Model Evaluation":
    st.title("Detailed Model Evaluation")
    name = st.selectbox("Model", results["Model"].tolist())
    y_true = predictions["actual"]
    y_pred = predictions[f"{name}_prediction"]
    y_prob = predictions[f"{name}_probability"]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(cm, text_auto=True,
        x=["Normal", "Elevated"], y=["Normal", "Elevated"],
        color_continuous_scale="Blues", title="Confusion matrix")
    fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, use_container_width=True)
    a, b = st.columns(2)
    with a:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", name=name)
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines",
                        line=dict(dash="dash"), name="Reference")
        fig.update_layout(title="ROC curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        fig = px.line(x=recall, y=precision,
            title="Precision-recall curve",
            labels={"x": "Recall", "y": "Precision"})
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Random Forest feature importance")
    fig = px.bar(importance.head(15).sort_values("Importance"),
        x="Importance", y="Feature", orientation="h")
    st.plotly_chart(fig, use_container_width=True)
# PAGE 7: INTERACTIVE CLASSIFICATION
elif page == "Risk Classification":
    st.title("Interactive Modeled Risk Classification")
    options = [n for n in models if n != "Baseline"]
    selected_model = st.selectbox("Model", options,
        index=options.index(best["Model"])
        if best["Model"] in options else 0)
    district = st.selectbox("District", sorted(df["district"].unique()))
    district_data = df[df["district"] == district]
    climatic_zone = district_data["climatic_zone"].iloc[0]
    st.write(f"Climatic zone: {climatic_zone}")
    defaults = district_data.median(numeric_only=True)
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            month = st.selectbox("Month", range(1, 13))
            rain24 = st.number_input("24-hour rainfall (mm)", 0.0,
                value=float(defaults["rain_24h"]))
            rain48 = st.number_input("48-hour rainfall (mm)", 0.0,
                value=float(defaults["rain_48h"]))
            rain72 = st.number_input("72-hour rainfall (mm)", 0.0,
                value=float(defaults["rain_72h"]))
        with b:
            soil1 = st.number_input("Surface soil moisture", 0.0, 1.0,
                value=float(defaults["soil_moisture_0_to_7cm_mean"]),
                format="%.3f")
            soil2 = st.number_input("Deeper soil moisture", 0.0, 1.0,
                value=float(defaults["soil_moisture_7_to_28cm_mean"]),
                format="%.3f")
            temperature = st.number_input("Maximum temperature",
                value=float(defaults["temperature_2m_max"]))
            wind = st.number_input("Maximum wind speed", 0.0,
                value=float(defaults["wind_speed_10m_max"]))
        submitted = st.form_submit_button("Classify modeled risk")
    if submitted:
          if not rain24 <= rain48 <= rain72:
            st.error("Rainfall must satisfy: 24h <= 48h <= 72h.")
            
          else:
            row = pd.DataFrame([{
                "district": district,
                "climatic_zone": climatic_zone,
                "rain_24h": rain24,
                "rain_48h": rain48,
                "rain_72h": rain72,
                "soil_moisture_0_to_7cm_mean": soil1,
                "soil_moisture_7_to_28cm_mean": soil2,
                "temperature_2m_max": temperature,
                "wind_speed_10m_max": wind,
                "month": month
            }])
            model = models[selected_model]
            prediction = model.predict(row)[0]
            probability = float(model.predict_proba(row)[0][1])
            st.subheader("Classification Result")
            if prediction == 1:
                st.error("Modeled Elevated Risk")
            else:
                st.success("Modeled Normal Risk")
            st.metric("Model elevated-risk score", f"{probability:.2%}")
            st.progress(probability)
            st.caption("Not a calibrated chance of actual flooding.")
    