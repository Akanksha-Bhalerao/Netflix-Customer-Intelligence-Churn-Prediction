
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Netflix Customer Intelligence",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #111111;
    color: white;
}
[data-testid="stSidebar"] {
    background-color: #191919;
}
h1, h2, h3 {
    color: #E50914;
}
[data-testid="stMetric"] {
    background-color: #222222;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    features = pd.read_csv("/content/netflix_feature_engineered.csv")
    segments = pd.read_csv("/content/netflix_customer_segments.csv")
    clv = pd.read_csv("/content/netflix_customer_lifetime_value.csv")

    # Add customer segment labels
    data = features.merge(
        segments[["customer_id", "cluster"]],
        on="customer_id",
        how="left"
    )

    # Add customer lifetime value information
    clv_columns = [
        "customer_id",
        "estimated_subscription_value",
        "historical_payment_revenue",
        "customer_value_category"
    ]

    available_clv_columns = [
        col for col in clv_columns if col in clv.columns
    ]

    data = data.merge(
        clv[available_clv_columns],
        on="customer_id",
        how="left"
    )

    return data

@st.cache_resource
def load_model():
    model = joblib.load("/content/netflix_churn_model.pkl")

    with open("/content/netflix_churn_threshold.json", "r") as file:
        threshold_data = json.load(file)

    return model, threshold_data["churn_threshold"]

try:
    df = load_data()
    model, threshold = load_model()
except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.title("🎬 Netflix Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate Dashboard",
    [
        "Overview",
        "Churn Analysis",
        "Customer Segmentation",
        "Customer Lifetime Value",
        "Churn Prediction",
        "AI Analytics Copilot"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Netflix Customer Intelligence Project")

# ---------------- DASHBOARD TITLE ----------------

st.title("🎬 Netflix Customer Intelligence")
st.markdown(
    "Customer Analytics | Churn Prediction | Segmentation | Lifetime Value"
)

st.markdown("---")

# ---------------- OVERVIEW ----------------

if page == "Overview":

    st.header("Business Overview")

    total_customers = df["customer_id"].nunique()
    churned = int(df["churned"].sum())
    churn_rate = df["churned"].mean() * 100
    avg_monthly_price = df["monthly_price"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned:,}")
    col3.metric("Churn Rate", f"{churn_rate:.2f}%")
    col4.metric("Avg Monthly Price", f"{avg_monthly_price:.2f}")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Churn Distribution")

        churn_counts = df["churned"].value_counts().rename(
            index={0: "Not Churned", 1: "Churned"}
        )

        fig = px.pie(
            values=churn_counts.values,
            names=churn_counts.index,
            hole=0.55,
            title="Customer Churn Status"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Customers by Subscription Plan")

        plan_counts = df["plan_id"].value_counts().reset_index()
        plan_counts.columns = ["Plan", "Customers"]

        fig = px.bar(
            plan_counts,
            x="Plan",
            y="Customers",
            title="Subscription Plan Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------------- CHURN ANALYSIS ----------------

elif page == "Churn Analysis":

    st.header("Customer Churn Analysis")

    churn_by_plan = (
        df.groupby("plan_id")["churned"]
        .mean()
        .mul(100)
        .reset_index()
    )

    churn_by_plan.columns = ["Plan", "Churn Rate"]

    fig = px.bar(
        churn_by_plan,
        x="Plan",
        y="Churn Rate",
        title="Churn Rate by Subscription Plan",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:
        st.subheader("Churn by Auto-Renewal")

        auto = (
            df.groupby("auto_renew")["churned"]
            .mean()
            .mul(100)
            .reset_index()
        )

        auto["auto_renew"] = auto["auto_renew"].astype(str)

        fig = px.bar(
            auto,
            x="auto_renew",
            y="churned",
            title="Churn Rate by Auto-Renewal Setting",
            labels={
                "auto_renew": "Auto-Renew",
                "churned": "Churn Rate (%)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Churn by Customer Value")

        if "customer_value_category" in df.columns:
            value_churn = (
                df.groupby("customer_value_category")["churned"]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                value_churn,
                x="customer_value_category",
                y="churned",
                title="Churn Rate by Customer Value",
                labels={
                    "customer_value_category": "Customer Value",
                    "churned": "Churn Rate (%)"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Data Summary")

    st.dataframe(
        df.groupby("plan_id")["churned"]
        .agg(["count", "sum", "mean"])
        .rename(columns={
            "count": "Total Customers",
            "sum": "Churned Customers",
            "mean": "Churn Rate"
        }),
        use_container_width=True
    )

# ---------------- CUSTOMER SEGMENTATION ----------------

elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    if "cluster" not in df.columns:
        st.warning("Customer segment information is unavailable.")
    else:
        cluster_summary = (
            df.groupby("cluster")
            .agg(
                Customers=("customer_id", "count"),
                Average_Age=("age", "mean"),
                Average_Watch_Minutes=("total_watch_minutes", "mean"),
                Average_Sessions=("viewing_sessions", "mean"),
                Churn_Rate=("churned", "mean")
            )
            .reset_index()
        )

        cluster_summary["Churn_Rate"] *= 100

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                cluster_summary,
                x="cluster",
                y="Customers",
                title="Customer Count by Segment",
                text_auto=True
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(
                cluster_summary,
                x="cluster",
                y="Churn_Rate",
                title="Churn Rate by Segment",
                text_auto=".2f"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Segment Profile")
        st.dataframe(
            cluster_summary.round(2),
            use_container_width=True
        )

# ---------------- CUSTOMER LIFETIME VALUE ----------------

elif page == "Customer Lifetime Value":

    st.header("Customer Lifetime Value Analysis")

    if "estimated_subscription_value" in df.columns:

        avg_value = df["estimated_subscription_value"].mean()
        total_value = df["estimated_subscription_value"].sum()
        high_value = (
            df["customer_value_category"] == "High"
        ).sum()

        col1, col2, col3 = st.columns(3)

        col1.metric("Average Estimated Value", f"{avg_value:,.2f}")
        col2.metric("Total Estimated Value", f"{total_value:,.2f}")
        col3.metric("High-Value Customers", f"{high_value:,}")

        left, right = st.columns(2)

        with left:
            fig = px.histogram(
                df,
                x="estimated_subscription_value",
                nbins=30,
                title="Estimated Customer Value Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        with right:
            value_summary = (
                df.groupby("customer_value_category")
                ["estimated_subscription_value"]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                value_summary,
                x="customer_value_category",
                y="estimated_subscription_value",
                title="Average Value by Customer Category"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.info(
            "Estimated subscription value is a historical proxy "
            "calculated from monthly price and subscription tenure. "
            "It is not a forecast of future revenue or profit."
        )

    else:
        st.warning("Customer value information is unavailable.")

# ---------------- CHURN PREDICTION ----------------

elif page == "Churn Prediction":

    st.header("Individual Customer Churn Prediction")

    st.write(
        "Select a customer from the dataset to view the model's "
        "churn prediction."
    )

    customer_ids = df["customer_id"].tolist()

    selected_id = st.selectbox(
        "Select Customer ID",
        customer_ids
    )

    customer_row = df[
        df["customer_id"] == selected_id
    ].iloc[[0]]

    feature_columns = [
        col for col in df.columns
        if col not in [
            "customer_id",
            "subscription_start_date",
            "churned",
            "cluster",
            "estimated_subscription_value",
            "historical_payment_revenue",
            "customer_value_category"
        ]
    ]

    input_data = customer_row[feature_columns]

    if st.button("Predict Churn Risk"):

        probability = model.predict_proba(input_data)[0][1]

        prediction = int(probability >= threshold)

        col1, col2 = st.columns(2)

        col1.metric(
            "Churn Probability",
            f"{probability * 100:.2f}%"
        )

        col2.metric(
            "Selected Threshold",
            f"{threshold:.2f}"
        )

        if prediction == 1:
            st.error("⚠️ Model Prediction: Likely to Churn")
            st.write(
                "Consider reviewing this customer's engagement, "
                "payment history, and support experience."
            )
        else:
            st.success("Model Prediction: Not Likely to Churn")
            st.write(
                "Continue monitoring customer engagement "
                "and satisfaction."
            )

        st.caption(
            "This is a model-based prediction, not a guarantee "
            "of future customer behavior."
        )



# ---------------- AI ANALYTICS COPILOT ----------------

elif page == "AI Analytics Copilot":

    st.header("🤖 AI Analytics Copilot")
    st.write(
        "Ask questions about Netflix customers, subscriptions, "
        "payments, viewing activity and churn."
    )

    import sqlite3
    import re

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Your key is used only for this application session."
    )

    question = st.text_area(
        "Ask a business question",
        placeholder="Example: What is the overall customer churn rate?"
    )

    if st.button("Generate Answer"):

        if not api_key:
            st.warning("Please enter your Groq API key.")

        elif not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                from groq import Groq

                client = Groq(api_key=api_key)

                schema_description = """
                SQLite database tables:

                customers
                subscriptions
                subscription_plans
                viewing_activity
                support_tickets
                payments
                churn_labels
                customer_feedback
                content

                Important business definition:
                Overall churn rate =
                COUNT of churn_labels where churned=1
                divided by total churn_labels rows, multiplied by 100.

                For final plan-wise churn, use subscriptions records
                where subscription_status is Active or Cancelled.
                Exclude Plan Changed historical records.
                """

                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a careful Netflix data analyst. "
                                "Generate only one SQLite SELECT query. "
                                "Use only the tables and columns described "
                                "in the database schema. Never generate "
                                "INSERT, UPDATE, DELETE, DROP, ALTER, "
                                "CREATE or other modifying commands. "
                                "Use LIMIT 100. Return only SQL, "
                                "without markdown or explanation. "
                                + schema_description
                            )
                        },
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    temperature=0,
                    max_tokens=2000
                )

                message = response.choices[0].message
                raw_response = message.content or ""
                if not raw_response.strip():
                    raw_response = getattr(message, "reasoning", "") or ""

                # Extract only the first SQL statement
                match = re.search(
                    r"\b(SELECT|WITH)\b[\s\S]*?;",
                    raw_response,
                    re.I
                )

                if not match:
                    raise ValueError("The AI did not return a valid SQL query.")

                sql = match.group(0).strip()

                st.subheader("Generated SQL")
                st.code(sql, language="sql")

                # Basic read-only SQL safety checks
                if not re.match(r"^(SELECT|WITH)\b", sql, re.I):
                    raise ValueError("Only SELECT queries are allowed.")

                if ";" in sql.rstrip(";"):
                    raise ValueError("Multiple SQL statements are not allowed.")

                blocked = re.search(
                    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|"
                    r"REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
                    sql,
                    re.I
                )

                if blocked:
                    raise ValueError("Unsafe SQL command detected.")

                if "--" in sql or "/*" in sql:
                    raise ValueError("SQL comments are not allowed.")

                # Execute using a read-only database connection
                db_uri = "file:/content/netflix_customer_intelligence.db?mode=ro"
                connection = sqlite3.connect(db_uri, uri=True)

                try:
                    connection.set_authorizer(
                        lambda action, arg1, arg2, db, trigger:
                        sqlite3.SQLITE_OK
                        if action in (
                            sqlite3.SQLITE_SELECT,
                            sqlite3.SQLITE_READ,
                            sqlite3.SQLITE_FUNCTION,
                            sqlite3.SQLITE_RECURSIVE
                        )
                        else sqlite3.SQLITE_DENY
                    )

                    connection.set_progress_handler(
                        lambda: 1, 1000000
                    )

                    result = pd.read_sql_query(sql, connection)

                finally:
                    connection.close()

                result = result.head(100)

                st.subheader("Query Results")
                st.dataframe(result, use_container_width=True)

                st.success("SQL query executed successfully.")

            except Exception as e:
                st.error(f"Copilot could not complete the request: {e}")


# ---------------- FOOTER ----------------

st.markdown("---")
st.caption(
    "Netflix Customer Intelligence & AI Analytics Copilot | "
    "Developed as a Data Science Project"
)
