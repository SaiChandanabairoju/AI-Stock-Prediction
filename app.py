import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from predict import predict_stock

# ======================================================
# Page Config
# ======================================================

st.set_page_config(
    page_title="AI Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)

# ======================================================
# Sidebar
# ======================================================

st.sidebar.title("📈 AI Stock Predictor")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

st.sidebar.markdown("---")

st.sidebar.info(
"""
Examples

AAPL

MSFT

GOOGL

TSLA

AMZN

META

TCS.NS

INFY.NS

RELIANCE.NS
"""
)

# ======================================================
# Title
# ======================================================

st.title("📈 AI Stock Price Prediction using LSTM")

st.write(
"""
This application predicts tomorrow's stock closing price
using an LSTM Deep Learning model.
"""
)

# ======================================================
# Predict Button
# ======================================================

if st.button("Predict"):

    with st.spinner("Loading Model..."):

        result = predict_stock(stock)

    if result is None:

        st.error("Invalid Stock Symbol")

    else:

        data = result["data"]

        company = result["company"]

        today = result["today_price"]

        tomorrow = result["tomorrow_price"]

        future = result["future"]

        rmse = result["rmse"]

        mae = result["mae"]

        accuracy = result["accuracy"]

        actual = result["actual"]

        predictions = result["predictions"]

        # ======================================================
        # Company Information
        # ======================================================

        st.header("🏢 Company Information")

        c1, c2 = st.columns(2)

        with c1:

            st.write("**Company:**", company["Name"])

            st.write("**Sector:**", company["Sector"])

            st.write("**Industry:**", company["Industry"])

            st.write("**Country:**", company["Country"])

        with c2:

            st.write("**Market Cap:**", company["Market Cap"])

            st.write("**52 Week High:**", company["52 Week High"])

            st.write("**52 Week Low:**", company["52 Week Low"])

            st.write("**Website:**", company["Website"])

        st.divider()

        # ======================================================
        # Metrics
        # ======================================================

        st.header("📊 Prediction Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Today's Price",
                f"${today:.2f}"
            )

        with col2:

            st.metric(
                "Tomorrow Prediction",
                f"${tomorrow:.2f}"
            )

        with col3:

            diff = tomorrow - today

            st.metric(
                "Difference",
                f"{diff:.2f}"
            )

        col4, col5, col6 = st.columns(3)

        with col4:

            st.metric(
                "Accuracy",
                f"{accuracy:.2f}%"
            )

        with col5:

            st.metric(
                "RMSE",
                f"{rmse:.2f}"
            )

        with col6:

            st.metric(
                "MAE",
                f"{mae:.2f}"
            )

        st.divider()

        # ======================================================
        # Candlestick Chart
        # ======================================================

        st.header("📈 Candlestick Chart")

        fig = go.Figure()

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Candlestick"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA50"],
                name="MA50"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MA200"],
                name="MA200"
            )
        )

        fig.update_layout(

            title=f"{stock} Stock Chart",

            xaxis_title="Date",

            yaxis_title="Price",

            xaxis_rangeslider_visible=False,

            height=700
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        # ======================================================
        # Actual vs Prediction
        # ======================================================

        st.header("🤖 Actual vs Predicted")

        fig2 = go.Figure()

        fig2.add_trace(

            go.Scatter(

                y=actual.flatten(),

                name="Actual"

            )

        )

        fig2.add_trace(

            go.Scatter(

                y=predictions.flatten(),

                name="Predicted"

            )

        )

        fig2.update_layout(

            title="Model Prediction",

            xaxis_title="Days",

            yaxis_title="Price"

        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.divider()

        # ======================================================
        # Future Prediction
        # ======================================================

        st.header("📅 Next 7 Days Forecast")

        future_df = pd.DataFrame({

            "Day": [

                "Day 1",

                "Day 2",

                "Day 3",

                "Day 4",

                "Day 5",

                "Day 6",

                "Day 7"

            ],

            "Predicted Price": future

        })

        st.dataframe(
            future_df,
            use_container_width=True
        )

        fig3 = go.Figure()

        fig3.add_trace(

            go.Scatter(

                x=future_df["Day"],

                y=future_df["Predicted Price"],

                mode="lines+markers",

                name="Forecast"

            )

        )

        fig3.update_layout(

            title="Next 7 Days Forecast",

            xaxis_title="Days",

            yaxis_title="Predicted Price"

        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

        st.divider()

        # ======================================================
        # Historical Data
        # ======================================================

        st.header("📄 Historical Data")

        st.dataframe(
            data.tail(20),
            use_container_width=True
        )

        # ======================================================
        # Download Button
        # ======================================================

        csv = data.to_csv().encode("utf-8")

        st.download_button(

            "📥 Download Dataset",

            csv,

            file_name=f"{stock}.csv",

            mime="text/csv"

        )

# ======================================================
# Footer
# ======================================================

st.markdown("---")

st.caption(
    "Developed using Python, TensorFlow, LSTM, Streamlit, Plotly and Yahoo Finance API"
)