import streamlit as st
import pandas as pd
import joblib


# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI Wi-Fi Optimizer",
    page_icon="📡",
    layout="wide"
)


# --------------------------------
# LOAD AI MODEL
# --------------------------------

model = joblib.load("wifi_channel_model.pkl")


# --------------------------------
# TITLE
# --------------------------------

st.title("📡 AI-Based Smart Wi-Fi Channel Selection")

st.write(
    "Machine Learning based wireless network "
    "channel optimization system."
)

st.divider()


# --------------------------------
# CHANNEL INPUTS
# --------------------------------

st.subheader("📶 Enter Wi-Fi Network Conditions")

channels = [1, 6, 11]

data = []


for channel in channels:

    st.markdown(
        f"### 📡 Wi-Fi Channel {channel}"
    )

    col1, col2, col3, col4, col5 = st.columns(5)


    with col1:

        rssi = st.slider(
            "RSSI (dBm)",
            min_value=-90,
            max_value=-30,
            value=-60,
            key=f"rssi_{channel}"
        )


    with col2:

        networks = st.slider(
            "Nearby Networks",
            min_value=1,
            max_value=20,
            value=5,
            key=f"networks_{channel}"
        )


    with col3:

        interference = st.slider(
            "Interference (%)",
            min_value=0,
            max_value=100,
            value=30,
            key=f"interference_{channel}"
        )


    with col4:

        latency = st.slider(
            "Latency (ms)",
            min_value=1,
            max_value=150,
            value=30,
            key=f"latency_{channel}"
        )


    with col5:

        packet_loss = st.slider(
            "Packet Loss (%)",
            min_value=0.0,
            max_value=15.0,
            value=2.0,
            key=f"packet_{channel}"
        )


    data.append([
        channel,
        rssi,
        networks,
        interference,
        latency,
        packet_loss
    ])


# --------------------------------
# ANALYZE BUTTON
# --------------------------------

st.divider()

if st.button(
    "🤖 Analyze Wi-Fi Network",
    use_container_width=True
):

    input_data = pd.DataFrame(
        data,
        columns=[
            "channel",
            "rssi",
            "num_networks",
            "interference",
            "latency",
            "packet_loss"
        ]
    )


    # --------------------------------
    # AI PREDICTION
    # --------------------------------

    predictions = model.predict(
        input_data
    )


    input_data[
        "predicted_quality"
    ] = predictions


    # Make values between 0 and 100
    input_data[
        "predicted_quality"
    ] = input_data[
        "predicted_quality"
    ].clip(0, 100)


    # --------------------------------
    # RESULTS
    # --------------------------------

    st.subheader("📊 AI Analysis")


    display_data = input_data.copy()

    display_data[
        "predicted_quality"
    ] = display_data[
        "predicted_quality"
    ].round(2)


    st.dataframe(
        display_data,
        use_container_width=True
    )


    # --------------------------------
    # FIND BEST CHANNEL
    # --------------------------------

    best_index = input_data[
        "predicted_quality"
    ].idxmax()


    best_channel = input_data.loc[
        best_index,
        "channel"
    ]


    best_quality = input_data.loc[
        best_index,
        "predicted_quality"
    ]


    # --------------------------------
    # SHOW RECOMMENDATION
    # --------------------------------

    st.success(
        f"🎯 AI Recommended Wi-Fi Channel: "
        f"{int(best_channel)}"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Predicted Quality",
            f"{best_quality:.2f}%"
        )


    with col2:

        if best_quality >= 80:

            quality = "Excellent 🟢"

        elif best_quality >= 60:

            quality = "Good 🟢"

        elif best_quality >= 40:

            quality = "Average 🟡"

        else:

            quality = "Poor 🔴"


        st.metric(
            "Network Quality",
            quality
        )


    st.info(
        f"Based on the entered wireless conditions, "
        f"the AI model predicts that Channel "
        f"{int(best_channel)} will provide the "
        f"best network quality."
    )


    # --------------------------------
    # GRAPH
    # --------------------------------

    st.subheader(
        "📈 Predicted Quality by Channel"
    )


    chart_data = input_data[
        ["channel", "predicted_quality"]
    ].copy()


    chart_data = chart_data.set_index(
        "channel"
    )


    st.bar_chart(
        chart_data
    )