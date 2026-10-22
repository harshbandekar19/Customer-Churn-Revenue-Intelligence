
import os
import joblib
import pandas as pd
import streamlit as st

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Customer Churn & Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

THRESHOLD = 0.45

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "random_forest_churn_model.pkl"
)

FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

CATEGORICAL_VALUES = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}

INTERNET_OPTIONS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================
@st.cache_resource
def load_model(path):
    return joblib.load(path)


if not os.path.exists(MODEL_PATH):
    st.error(
        "Model file not found. Place "
        "'random_forest_churn_model.pkl' in the same folder as app.py."
    )
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.exception(error)
    st.stop()

if getattr(model, "n_features_in_", None) != len(FEATURES):
    st.error(
        "Model feature-count mismatch. This application expects "
        f"{len(FEATURES)} features."
    )
    st.stop()

# Confirm exact feature names when the pipeline exposes them.
if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != FEATURES:
        st.error(
            "The model feature names or their order do not match "
            "the application. Please check the training columns."
        )
        st.stop()


# ============================================================
# 3. VALIDATE CUSTOMER DATA
# ============================================================
def validate_customer_data(data):
    """
    Validate the 19 required model features.
    Returns (is_valid, message, cleaned_data).
    """
    data = data.copy()

    missing_columns = [
        column for column in FEATURES
        if column not in data.columns
    ]

    if missing_columns:
        return (
            False,
            "Missing required columns: " + ", ".join(missing_columns),
            data
        )

    if data.empty:
        return False, "The CSV contains no customer records.", data

    # Convert numeric columns carefully.
    for column in NUMERIC_COLUMNS:
        try:
            data[column] = pd.to_numeric(
                data[column], errors="raise"
            )
        except (ValueError, TypeError):
            return (
                False,
                f"Column '{column}' must contain numeric values.",
                data
            )

    # Reject missing or infinite numeric values.
    if data[FEATURES].isnull().any().any():
        missing = data[FEATURES].columns[
            data[FEATURES].isnull().any()
        ].tolist()

        return (
            False,
            "Missing values found in: " + ", ".join(missing),
            data
        )

    if not data[NUMERIC_COLUMNS].apply(
        lambda column: column.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )
    ).all().all():
        return False, "Numeric features contain infinite or invalid values.", data

    # Validate category values.
    for column, allowed_values in CATEGORICAL_VALUES.items():
        if not data[column].isin(allowed_values).all():
            invalid_values = data.loc[
                ~data[column].isin(allowed_values), column
            ].drop_duplicates().tolist()

            return (
                False,
                f"Unsupported value(s) in '{column}': {invalid_values}. "
                f"Allowed values: {allowed_values}",
                data
            )

    # Validate numeric ranges.
    if not data["SeniorCitizen"].isin([0, 1]).all():
        return False, "SeniorCitizen must contain only 0 or 1.", data

    if not data["tenure"].between(0, 72).all():
        return False, "Tenure must be between 0 and 72 months.", data

    if (
        data[["MonthlyCharges", "TotalCharges"]] < 0
    ).any().any():
        return False, "Charges cannot be negative.", data

    # Validate service combinations.
    no_phone = data["PhoneService"].eq("No")

    if not (
        data.loc[no_phone, "MultipleLines"] == "No phone service"
    ).all():
        return (
            False,
            "When PhoneService is No, MultipleLines must be "
            "'No phone service'.",
            data
        )

    has_phone = data["PhoneService"].eq("Yes")

    if (
        data.loc[has_phone, "MultipleLines"] == "No phone service"
    ).any():
        return (
            False,
            "'No phone service' is only valid when PhoneService is No.",
            data
        )

    no_internet = data["InternetService"].eq("No")

    for column in INTERNET_OPTIONS:
        if not (
            data.loc[no_internet, column] == "No internet service"
        ).all():
            return (
                False,
                f"When InternetService is No, '{column}' must be "
                "'No internet service'.",
                data
            )

        has_internet = ~no_internet

        if (
            data.loc[has_internet, column] == "No internet service"
        ).any():
            return (
                False,
                f"'No internet service' is invalid in '{column}' "
                "when InternetService is DSL or Fiber optic.",
                data
            )

    return True, "Validation successful.", data


# ============================================================
# 4. PREDICTIONS AND RETENTION RECOMMENDATIONS
# ============================================================
def add_predictions(data):
    """Predict churn with the saved preprocessing/model pipeline."""
    data = data.copy()

    # Preserve extra columns, such as customerID, in the output.
    probabilities = model.predict_proba(data[FEATURES])[:, 1]

    result = data.copy()

    result["Churn_Probability"] = probabilities
    result["Predicted_Churn"] = (
        probabilities >= THRESHOLD
    ).astype(int)

    result["Predicted_Outcome"] = result["Predicted_Churn"].map({
        0: "Not flagged to churn",
        1: "Likely to churn"
    })

    # Risk bands are separate from the 0.45 classification threshold.
    result["Risk_Level"] = pd.cut(
        probabilities,
        bins=[0, 0.40, 0.70, 1.000001],
        labels=["Low", "Medium", "High"],
        right=False,
        include_lowest=True
    ).astype(str)

    def recommend(row):
        if row["Predicted_Churn"] == 0:
            return (
                "Maintain service quality and monitor satisfaction"
            )

        if row["Contract"] == "Month-to-month":
            return (
                "Prioritize retention contact and discuss suitable plans"
            )

        if (
            row["InternetService"] != "No"
            and row["OnlineSecurity"] == "No"
        ):
            return "Review online security options with customer"

        if (
            row["InternetService"] != "No"
            and row["TechSupport"] == "No"
        ):
            return "Offer a technical support check-in"

        return "Contact customer to understand concerns"

    result["Recommended_Action"] = result.apply(recommend, axis=1)

    return result


# ============================================================
# 5. PAGE HEADER AND SIDEBAR
# ============================================================
st.title("Customer Churn & Revenue Intelligence")
st.caption(
    "Predict churn risk, analyze customer segments, "
    "and support data-driven retention decisions."
)

st.sidebar.title("Model Settings")
st.sidebar.metric("Classification threshold", "45%")
st.sidebar.write(
    "Customers with a predicted churn probability of at least "
    "45% are flagged as likely to churn."
)

st.sidebar.divider()
st.sidebar.subheader("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Executive Dashboard",
        "Single Customer Prediction",
        "Bulk Customer Prediction"
    ]
)

st.sidebar.caption(
    "Predictions are estimates, not guarantees of future behavior."
)


# ============================================================
# 6. EXECUTIVE DASHBOARD
# ============================================================
if page == "Executive Dashboard":
    st.header("Executive Dashboard")

    st.write(
        "Upload and analyze a customer CSV in the Bulk Customer "
        "Prediction page. Your latest results will appear here."
    )

    if "bulk_results" not in st.session_state:
        st.info(
            "No bulk predictions are available yet. Open Bulk Customer "
            "Prediction, upload your CSV, and run predictions first."
        )
    else:
        results = st.session_state["bulk_results"]

        total_customers = len(results)
        flagged_customers = int(results["Predicted_Churn"].sum())
        high_risk_customers = int(
            (results["Risk_Level"] == "High").sum()
        )

        high_risk_charges = results.loc[
            results["Risk_Level"] == "High",
            "MonthlyCharges"
        ].sum()

        churn_flag_rate = (
            flagged_customers / total_customers
            if total_customers else 0
        )

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Customers analyzed",
            f"{total_customers:,}"
        )
        k2.metric(
            "Flagged for churn",
            f"{flagged_customers:,}"
        )
        k3.metric(
            "High-risk customers",
            f"{high_risk_customers:,}"
        )
        k4.metric(
            "High-risk monthly charges",
            f"{high_risk_charges:,.2f}"
        )

        st.caption(
            "High-risk monthly charges are associated charges, "
            "not guaranteed lost revenue. Currency follows the dataset."
        )

        st.divider()

        # Risk distribution
        left, right = st.columns(2)

        with left:
            st.subheader("Customer Risk Distribution")

            risk_counts = (
                results["Risk_Level"]
                .value_counts()
                .reindex(["Low", "Medium", "High"], fill_value=0)
            )

            st.bar_chart(risk_counts)

        # Probability distribution
        with right:
            st.subheader("Churn Probability Distribution")

            probability_bins = pd.cut(
                results["Churn_Probability"],
                bins=[0, 0.20, 0.40, 0.60, 0.80, 1.000001],
                labels=[
                    "0–20%",
                    "20–40%",
                    "40–60%",
                    "60–80%",
                    "80–100%"
                ],
                right=False,
                include_lowest=True
            )

            probability_counts = (
                probability_bins.value_counts(sort=False)
                .reindex(
                    ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"],
                    fill_value=0
                )
            )

            st.bar_chart(probability_counts)

        st.divider()

        # Retention actions
        st.subheader("Recommended Retention Actions")

        action_counts = results["Recommended_Action"].value_counts()
        st.bar_chart(action_counts)

        st.divider()

        # Prediction table
        st.subheader("Customer Prediction Details")

        risk_filter = st.multiselect(
            "Filter by risk category",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"]
        )

        filtered_results = results[
            results["Risk_Level"].isin(risk_filter)
        ]

        st.dataframe(
            filtered_results,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download prediction results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_bulk_predictions.csv",
            mime="text/csv"
        )


# ============================================================
# 7. SINGLE CUSTOMER PREDICTION
# ============================================================
elif page == "Single Customer Prediction":
    st.header("Single Customer Prediction")
    st.write("Enter customer details to estimate churn probability.")

    with st.form("customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])

            senior = st.selectbox(
                "Senior citizen",
                [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No"
            )

            partner = st.selectbox("Has partner?", ["Yes", "No"])
            dependents = st.selectbox("Has dependents?", ["Yes", "No"])

            tenure = st.number_input(
                "Tenure (months)",
                min_value=0,
                max_value=72,
                value=12
            )

            phone = st.selectbox("Phone service", ["Yes", "No"])

            lines = st.selectbox(
                "Multiple lines",
                ["No", "Yes", "No phone service"]
            )

        with col2:
            internet = st.selectbox(
                "Internet service",
                ["DSL", "Fiber optic", "No"]
            )

            security = st.selectbox(
                "Online security",
                ["Yes", "No", "No internet service"]
            )

            backup = st.selectbox(
                "Online backup",
                ["Yes", "No", "No internet service"]
            )

            device = st.selectbox(
                "Device protection",
                ["Yes", "No", "No internet service"]
            )

            support = st.selectbox(
                "Tech support",
                ["Yes", "No", "No internet service"]
            )

            tv = st.selectbox(
                "Streaming TV",
                ["Yes", "No", "No internet service"]
            )

            movies = st.selectbox(
                "Streaming movies",
                ["Yes", "No", "No internet service"]
            )

        with col3:
            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            paperless = st.selectbox(
                "Paperless billing",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly = st.number_input(
                "Monthly charges",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

            total = st.number_input(
                "Total charges",
                min_value=0.0,
                value=840.0,
                step=10.0
            )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            type="primary",
            use_container_width=True
        )

    if submitted:
        customer = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": lines,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device,
            "TechSupport": support,
            "StreamingTV": tv,
            "StreamingMovies": movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }])

        valid, message, customer = validate_customer_data(customer)

        if not valid:
            st.error(message)
        else:
            try:
                result = add_predictions(customer).iloc[0]

                st.divider()
                st.subheader("Prediction Results")

                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Churn probability",
                    f"{result['Churn_Probability']:.1%}"
                )
                m2.metric(
                    "Risk category",
                    result["Risk_Level"]
                )
                m3.metric(
                    "Predicted outcome",
                    result["Predicted_Outcome"]
                )

                st.progress(
                    float(result["Churn_Probability"])
                )

                if result["Predicted_Churn"] == 1:
                    st.warning(
                        "The model flags this customer for potential churn."
                    )
                else:
                    st.success(
                        "The customer is below the selected churn threshold."
                    )

                st.subheader("Recommended Retention Action")
                st.info(result["Recommended_Action"])

                with st.expander("View customer input"):
                    st.dataframe(
                        customer,
                        use_container_width=True,
                        hide_index=True
                    )

            except Exception as error:
                st.error(f"Prediction failed: {error}")
                st.exception(error)


# ============================================================
# 8. BULK CUSTOMER PREDICTION
# ============================================================
elif page == "Bulk Customer Prediction":
    st.header("Bulk Customer Prediction")

    st.write(
        "Upload a CSV containing customer records with all 19 required "
        "features. You may include a customerID column; it will be "
        "preserved in the output."
    )

    template_columns = ["customerID"] + FEATURES

    template_csv = pd.DataFrame(
        columns=template_columns
    ).to_csv(index=False)

    st.download_button(
        "Download CSV Template",
        data=template_csv,
        file_name="customer_churn_upload_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help="Upload a CSV with the exact feature column names."
    )

    if uploaded_file is not None:
        try:
            uploaded_data = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Unable to read the CSV file: {error}")
            st.stop()

        st.subheader("Uploaded Data Preview")
        st.dataframe(
            uploaded_data.head(10),
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"Rows: {len(uploaded_data):,} | "
            f"Columns: {len(uploaded_data.columns):,}"
        )

        if st.button(
            "Validate and Predict All Customers",
            type="primary"
        ):
            valid, message, cleaned_data = validate_customer_data(
                uploaded_data
            )

            if not valid:
                st.error(message)
            else:
                try:
                    results = add_predictions(cleaned_data)

                    st.session_state["bulk_results"] = results
                    st.session_state["bulk_filename"] = uploaded_file.name

                    st.success(
                        f"Predictions completed for {len(results):,} customers."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")
                    st.exception(error)

    # Display results only after a successful prediction.
    if "bulk_results" in st.session_state:
        results = st.session_state["bulk_results"]

        st.divider()
        st.subheader("Bulk Prediction Summary")

        total_customers = len(results)
        flagged_customers = int(results["Predicted_Churn"].sum())

        high_risk_customers = int(
            (results["Risk_Level"] == "High").sum()
        )

        high_risk_charges = results.loc[
            results["Risk_Level"] == "High",
            "MonthlyCharges"
        ].sum()

        k1, k2, k3, k4 = st.columns(4)

        k1.metric(
            "Customers analyzed",
            f"{total_customers:,}"
        )
        k2.metric(
            "Flagged for churn",
            f"{flagged_customers:,}"
        )
        k3.metric(
            "High-risk customers",
            f"{high_risk_customers:,}"
        )
        k4.metric(
            "High-risk monthly charges",
            f"{high_risk_charges:,.2f}"
        )

        st.caption(
            "High-risk monthly charges are the total monthly charges "
            "associated with High-risk customers, not guaranteed lost revenue."
        )

        st.divider()
        st.subheader("Customer Prediction Details")

        risk_filter = st.multiselect(
            "Filter customer table by risk",
            options=["Low", "Medium", "High"],
            default=["Low", "Medium", "High"],
            key="bulk_risk_filter"
        )

        filtered_results = results[
            results["Risk_Level"].isin(risk_filter)
        ]

        st.dataframe(
            filtered_results,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Prediction Results CSV",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="customer_churn_bulk_predictions.csv",
            mime="text/csv"
        )

        st.caption(
            "Open Executive Dashboard in the sidebar to view the charts "
            "for these results."
        )
