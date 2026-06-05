import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

st.set_page_config(page_title="Agro-LSTM Predictor", page_icon="🌾", layout="wide")

if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
        margin: 0 auto;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2a 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(20, 20, 40, 0.98) 0%, rgba(15, 15, 35, 0.98) 100%);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(100, 100, 255, 0.2);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #e8e8ff;
    }

    [data-testid="stSidebar"] .st-emotion-cache-1v0mbdj {
        background: rgba(100, 100, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(100, 100, 255, 0.3);
    }

    [data-testid="stSidebar"] .st-emotion-cache-1v0mbdj:hover {
        background: rgba(100, 100, 255, 0.2);
        border-color: rgba(100, 100, 255, 0.5);
        transition: all 0.3s ease;
    }

    h1, h2, h3, h4, h5, h6 {
        background: linear-gradient(135deg, #7c9cff, #5a7cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }

    h2 {
        font-size: 1.8rem;
        margin-top: 1rem;
    }

    h3 {
        font-size: 1.4rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #5a7cff, #7c9cff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.8rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(90, 124, 255, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(90, 124, 255, 0.4);
        transition: all 0.3s ease;
    }

    .reset-button > button {
        background: linear-gradient(90deg, #ff5a7c, #ff7c9c);
    }

    [data-testid="stMetric"] {
        background: rgba(30, 30, 60, 0.6);
        border-radius: 16px;
        padding: 1rem;
        border: 1px solid rgba(100, 100, 255, 0.2);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(90, 124, 255, 0.15);
        border-color: rgba(90, 124, 255, 0.4);
    }

    [data-testid="stMetricValue"] {
        background: linear-gradient(135deg, #7c9cff, #5a7cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2rem;
    }

    [data-testid="stMetricLabel"] {
        color: #b8b8ff;
        font-weight: 500;
        font-size: 0.9rem;
    }

    .stRadio > div {
        gap: 0.5rem;
    }

    .stRadio label {
        background: rgba(30, 30, 60, 0.4);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(100, 100, 255, 0.2);
        transition: all 0.3s ease;
        color: #e8e8ff;
    }

    .stRadio label:hover {
        background: rgba(90, 124, 255, 0.2);
        border-color: rgba(90, 124, 255, 0.5);
    }

    .stSlider [data-testid="stThumbValue"] {
        background: linear-gradient(90deg, #5a7cff, #7c9cff);
    }

    .stSlider [data-testid="stSliderTrack"] {
        background: rgba(90, 124, 255, 0.3);
    }

    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: rgba(20, 20, 45, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(100, 100, 255, 0.2);
    }

    .stExpander {
        background: rgba(20, 20, 45, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(100, 100, 255, 0.2);
    }

    .stExpander > div:first-child {
        background: rgba(90, 124, 255, 0.1);
        border-radius: 12px;
    }

    .stAlert {
        background: rgba(90, 124, 255, 0.1);
        border: 1px solid rgba(90, 124, 255, 0.3);
        border-radius: 12px;
        color: #e8e8ff;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #5a7cff, #7c9cff);
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in {
        animation: fadeInUp 0.5s ease-out;
    }

    hr {
        background: linear-gradient(90deg, transparent, #5a7cff, #7c9cff, transparent);
        height: 1px;
        border: none;
        margin: 1.5rem 0;
    }

    .stSelectbox label, .stSlider label, .stRadio label {
        color: #b8b8ff;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    [data-testid="stSelectbox"] > div {
        background: rgba(30, 30, 60, 0.4);
        border: 1px solid rgba(100, 100, 255, 0.2);
        border-radius: 8px;
    }

    .stColumns {
        gap: 1.5rem;
    }

    code {
        background: rgba(90, 124, 255, 0.15);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        color: #7c9cff;
        font-family: 'Monaco', monospace;
    }

    .stMarkdown {
        color: #e8e8ff;
        line-height: 1.6;
    }

    .stInfo {
        background: rgba(90, 124, 255, 0.1);
        border-left: 3px solid #5a7cff;
    }

    .stSuccess {
        background: rgba(90, 255, 150, 0.1);
        border-left: 3px solid #5aff96;
    }

    .element-container {
        margin-bottom: 1rem;
    }

    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }

    .st-emotion-cache-1r6slb0 {
        background: rgba(20, 20, 45, 0.8);
        border-radius: 12px;
        padding: 1rem;
    }

    .metric-card {
        background: rgba(30, 30, 60, 0.6);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(100, 100, 255, 0.2);
    }

    .metric-label {
        color: #b8b8ff;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .metric-value {
        background: linear-gradient(135deg, #7c9cff, #5a7cff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<div class="fade-in">', unsafe_allow_html=True)


@st.cache_data
@st.cache_data
def load_data():
    date_rng = pd.date_range(start='2021-01-01', end='2025-12-31', freq='D')
    days = len(date_rng)
    day_of_year = date_rng.dayofyear.values

    temp_base = 13 - 16 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    temp = temp_base + np.random.normal(0, 3, days)

    prob_ploaie = 0.2 + 0.1 * np.sin(2 * np.pi * (day_of_year - 60) / 365.25)
    mask_ploaie = np.random.uniform(0, 1, days) < prob_ploaie
    precip = mask_ploaie * np.random.exponential(5, days)
    precip = np.clip(precip, 0, 50)

    soil_moisture = 250 + precip * 3 - temp * 0.8 + np.random.normal(0, 2, days)
    soil_moisture = np.clip(soil_moisture, 100, 400)

    ndvi_base = 0.45 - 0.35 * np.cos(2 * np.pi * (day_of_year - 45) / 365.25)
    ndvi = ndvi_base + np.random.normal(0, 0.04, days)
    ndvi = np.clip(ndvi, 0, 1)

    df = pd.DataFrame({
        'Temperature_C': temp,
        'Precipitation_mm': precip,
        'Soil_Moisture': soil_moisture,
        'NDVI': ndvi
    }, index=date_rng)

    df.index.name = 'Date'
    return df


df_master = load_data()

st.sidebar.title("🌾 Agro-LSTM")
st.sidebar.markdown("Intelligent vegetation monitoring system")

menu = st.sidebar.radio(
    "",
    ["Exploratory Data Analysis",
     "LSTM Model & Forecasts",
     "Satellite Imagery",
     "Reports & Documentation"]
)

st.sidebar.markdown("---")
st.sidebar.info("**System Status:** Operational ✓")

if menu == "LSTM Model & Forecasts" and st.session_state.training_complete:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Training", type="secondary", key="reset_button"):
        st.session_state.training_complete = False
        st.session_state.results = None
        st.rerun()

if menu == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Days", f"{len(df_master)}")
    with col2:
        st.metric("Avg Temperature", f"{df_master['Temperature_C'].mean():.1f}°C")
    with col3:
        st.metric("Max Precipitation", f"{df_master['Precipitation_mm'].max():.1f}mm")
    with col4:
        st.metric("Avg NDVI", f"{df_master['NDVI'].mean():.2f}")

    st.markdown("### Time Series Analysis")
    variable_option = st.selectbox("Select Variable", df_master.columns)

    chart_data = df_master[variable_option].reset_index()
    chart_data.columns = ['Date', variable_option]
    st.line_chart(chart_data.set_index('Date'), color="#5a7cff")

    st.markdown("### Correlation Analysis")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_master.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax,
                annot_kws={'color': 'white', 'size': 10},
                cbar_kws={'label': 'Correlation Coefficient'})
    ax.set_facecolor('#0a0a1a')
    fig.patch.set_facecolor('#0a0a1a')
    plt.setp(ax.get_xticklabels(), color='white', rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), color='white')
    ax.tick_params(colors='white')
    st.pyplot(fig)

elif menu == "LSTM Model & Forecasts":
    st.title("Deep Learning Forecast System")
    st.markdown("Train LSTM neural network and simulate vegetation response under different climate scenarios.")

    with st.container():
        st.markdown("### Training Configuration")
        col_p1, col_p2 = st.columns(2)
        look_back = col_p1.slider("Look-back Window (days)", min_value=3, max_value=30, value=20, step=1,
                                  key="look_back")
        epochs = col_p2.slider("Training Epochs", min_value=10, max_value=100, value=100, step=10, key="epochs")

        st.markdown("### Climate Scenario")
        future_scenario = st.radio(
            "Select weather pattern for the next 60 days:",
            ["Normal Conditions", "Extreme Drought", "Rainy Period"],
            horizontal=True,
            key="scenario"
        )

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            train_clicked = st.button("Execute Training & Forecast", type="primary", use_container_width=True,
                                      key="train_btn")
        with col_btn2:
            if st.session_state.training_complete:
                if st.button("🔄 Reset", type="secondary", use_container_width=True, key="reset_inline"):
                    st.session_state.training_complete = False
                    st.session_state.results = None
                    st.rerun()

        if train_clicked:
            with st.spinner("Processing LSTM neural network training..."):

                scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
                X_data = df_master[['Temperature_C', 'Precipitation_mm', 'Soil_Moisture']].values
                y_data = df_master[['NDVI']].values

                X_scaled = scaler_X.fit_transform(X_data)
                y_scaled = scaler_y.fit_transform(y_data)

                X_seq, y_seq = [], []
                for i in range(len(X_scaled) - look_back):
                    X_seq.append(X_scaled[i:(i + look_back), :])
                    y_seq.append(y_scaled[i + look_back, 0])
                X_seq, y_seq = np.array(X_seq), np.array(y_seq)

                train_size = int(len(X_seq) * 0.7)
                X_train, X_test = X_seq[:train_size], X_seq[train_size:]
                y_train, y_test = y_seq[:train_size], y_seq[train_size:]

                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]),
                         kernel_regularizer=l2(0.001)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False, kernel_regularizer=l2(0.001)),
                    Dropout(0.2),
                    Dense(16, kernel_regularizer=l2(0.001)),
                    LeakyReLU(alpha=0.1),
                    Dense(1, activation='sigmoid')
                ])
                model.compile(optimizer='adam', loss=Huber(delta=1.0))

                lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=0)
                early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

                history = model.fit(X_train, y_train, epochs=epochs, batch_size=16,
                                    validation_data=(X_test, y_test),
                                    callbacks=[lr_scheduler, early_stopping], verbose=0)

                predictions_scaled = model.predict(X_test, verbose=0)
                predictions = scaler_y.inverse_transform(predictions_scaled)
                y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1))

                rmse = np.sqrt(mean_squared_error(y_test_real, predictions))
                mae = mean_absolute_error(y_test_real, predictions)
                if np.any(y_test_real == 0):
                    mape = float('inf')
                    mape_display = "N/A (Div/0)"
                else:
                    mape = np.mean(np.abs((y_test_real - predictions) / y_test_real)) * 100
                    mape_display = f"{mape:.2f}%"
                r2 = r2_score(y_test_real, predictions)

                final_train_loss = history.history['loss'][-1]
                final_val_loss = history.history['val_loss'][-1]
                best_val_loss = min(history.history['val_loss'])
                best_epoch = np.argmin(history.history['val_loss']) + 1

                future_days = 60
                last_timestamp = df_master.index[-1]
                future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(days=1), periods=future_days, freq='D')

                if "Drought" in future_scenario:
                    f_temp = 32 + 2 * np.sin(np.linspace(0, 1, future_days)) + np.random.normal(0, 1, future_days)
                    f_precip = np.zeros(future_days)
                elif "Rainy" in future_scenario:
                    f_temp = 18 + 2 * np.sin(np.linspace(0, 1, future_days)) + np.random.normal(0, 1, future_days)
                    f_precip = np.clip(np.random.normal(15, 5, future_days), 5, 40)
                else:
                    f_temp = 22 + 4 * np.sin(np.linspace(0, 1, future_days)) + np.random.normal(0, 2, future_days)
                    f_precip = np.clip(np.random.normal(2, 5, future_days), 0, 15)

                f_soil = 250 + f_precip * 2 - f_temp * 0.5 + np.random.normal(0, 2, future_days)

                df_future_weather = pd.DataFrame(
                    {'Temperature_C': f_temp, 'Precipitation_mm': f_precip, 'Soil_Moisture': f_soil})

                df_context = df_master[['Temperature_C', 'Precipitation_mm', 'Soil_Moisture']].iloc[-look_back:]
                df_combined = pd.concat([df_context, df_future_weather.set_index(future_dates)])

                X_comb_scaled = scaler_X.transform(df_combined.values)

                X_fut_seq = []
                for i in range(len(X_comb_scaled) - look_back):
                    X_fut_seq.append(X_comb_scaled[i:(i + look_back), :])
                X_fut_seq = np.array(X_fut_seq)

                pred_fut_scaled = model.predict(X_fut_seq, verbose=0)
                future_predictions = scaler_y.inverse_transform(pred_fut_scaled)

                st.session_state.results = {
                    'rmse': rmse,
                    'mae': mae,
                    'mape': mape,
                    'mape_display': mape_display,
                    'r2': r2,
                    'final_train_loss': final_train_loss,
                    'final_val_loss': final_val_loss,
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'history': history.history,
                    'y_test_real': y_test_real,
                    'predictions': predictions,
                    'future_dates': future_dates,
                    'future_predictions': future_predictions,
                    'df_future_weather': df_future_weather,
                    'last_timestamp': last_timestamp,
                    'future_scenario': future_scenario
                }
                st.session_state.training_complete = True
                st.rerun()

    if st.session_state.training_complete and st.session_state.results is not None:
        results = st.session_state.results

        st.success("Training completed successfully")

        st.markdown("### Model Performance Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("RMSE", f"{results['rmse']:.4f}")
        with col_m2:
            st.metric("MAE", f"{results['mae']:.4f}")
        with col_m3:
            st.metric("MAPE", results['mape_display'])
        with col_m4:
            st.metric("R² Score", f"{results['r2']:.4f}")

        st.markdown("### Training Metrics")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.metric("Final Training Loss", f"{results['final_train_loss']:.6f}")
        with col_t2:
            st.metric("Final Validation Loss", f"{results['final_val_loss']:.6f}")
        with col_t3:
            st.metric("Best Validation Loss", f"{results['best_val_loss']:.6f}",
                      delta=f"Epoch {results['best_epoch']}")

        st.markdown("### Loss Curves")
        fig_loss, ax_loss = plt.subplots(figsize=(12, 5))
        ax_loss.plot(results['history']['loss'], label='Training Loss', color='#5a7cff', linewidth=2)
        ax_loss.plot(results['history']['val_loss'], label='Validation Loss', color='#ff7c5a', linewidth=2)
        ax_loss.axvline(x=results['best_epoch'] - 1, color='#5aff96', linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'Best Model (Epoch {results["best_epoch"]})')
        ax_loss.set_title('Training and Validation Loss Over Epochs', fontsize=14)
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss (Huber)')
        ax_loss.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        ax_loss.grid(True, linestyle='--', alpha=0.3)
        ax_loss.set_facecolor('#0a0a1a')
        fig_loss.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_loss.get_xticklabels(), color='white')
        plt.setp(ax_loss.get_yticklabels(), color='white')
        st.pyplot(fig_loss)

        st.markdown("### Actual vs Predicted Values")
        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
        ax_pred.scatter(results['y_test_real'], results['predictions'], alpha=0.5, color='#5a7cff', s=20)
        min_val = min(results['y_test_real'].min(), results['predictions'].min())
        max_val = max(results['y_test_real'].max(), results['predictions'].max())
        ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, color='#ff5a7c',
                     label='Perfect Prediction')
        ax_pred.set_xlabel('Actual NDVI')
        ax_pred.set_ylabel('Predicted NDVI')
        ax_pred.set_title(f'Prediction Scatter Plot (R² = {results["r2"]:.4f})')
        ax_pred.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        ax_pred.grid(True, linestyle='--', alpha=0.3)
        ax_pred.set_facecolor('#0a0a1a')
        fig_pred.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_pred.get_xticklabels(), color='white')
        plt.setp(ax_pred.get_yticklabels(), color='white')
        st.pyplot(fig_pred)

        st.markdown("### Prediction Error Distribution")
        errors = (results['y_test_real'].flatten() - results['predictions'].flatten())
        fig_err, ax_err = plt.subplots(figsize=(12, 5))
        ax_err.hist(errors, bins=50, color='#5a7cff', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_err.axvline(x=0, color='#ff5a7c', linestyle='--', linewidth=2, label='Zero Error')
        ax_err.axvline(x=np.mean(errors), color='#5aff96', linestyle='--', linewidth=2,
                       label=f'Mean Error: {np.mean(errors):.4f}')
        ax_err.set_xlabel('Prediction Error')
        ax_err.set_ylabel('Frequency')
        ax_err.set_title('Distribution of Prediction Errors')
        ax_err.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        ax_err.grid(True, linestyle='--', alpha=0.3)
        ax_err.set_facecolor('#0a0a1a')
        fig_err.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_err.get_xticklabels(), color='white')
        plt.setp(ax_err.get_yticklabels(), color='white')
        st.pyplot(fig_err)

        with st.expander("Training History Details"):
            history_df = pd.DataFrame({
                'Epoch': range(1, len(results['history']['loss']) + 1),
                'Training Loss': results['history']['loss'],
                'Validation Loss': results['history']['val_loss']
            })
            st.dataframe(history_df, use_container_width=True)

        st.markdown("### Future Forecast")
        fig_fut, ax_f = plt.subplots(figsize=(14, 5))
        historical_display_days = 90
        ax_f.plot(df_master.index[-historical_display_days:],
                  df_master['NDVI'].iloc[-historical_display_days:],
                  label='Historical NDVI', color='#5a7cff', linewidth=2)
        ax_f.plot(results['future_dates'], results['future_predictions'],
                  label='Forecasted NDVI', color='#ff7c5a',
                  linestyle='dashed', linewidth=2.5)
        ax_f.axvline(x=results['last_timestamp'], color='#ff5a7c', linestyle=':', label='Present', linewidth=2)
        ax_f.set_title('NDVI Evolution: Historical Data & Future Forecast', fontsize=14)
        ax_f.set_ylabel('NDVI Value')
        ax_f.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        ax_f.grid(True, linestyle='--', alpha=0.3)
        ax_f.set_facecolor('#0a0a1a')
        fig_fut.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_f.get_xticklabels(), color='white')
        plt.setp(ax_f.get_yticklabels(), color='white')
        st.pyplot(fig_fut)

        with st.expander("Weather Data Preview"):
            st.line_chart(
                results['df_future_weather'][['Temperature_C', 'Precipitation_mm']].set_index(results['future_dates']))

elif menu == "Satellite Imagery":
    st.title("Satellite Vegetation Monitoring")
    st.markdown("Spatial visualization of NDVI index with satellite imagery simulation.")

    col_map, col_img = st.columns(2)

    with col_map:
        st.markdown("### Geographic Location")
        m = folium.Map(location=[44.5, 27.3], zoom_start=8, tiles=None)
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri', name='Satellite View', overlay=False, control=True
        ).add_to(m)
        folium.Rectangle(
            bounds=[[44.2, 26.8], [44.8, 27.8]],
            color='#5a7cff',
            fill=True,
            fill_opacity=0.2,
            tooltip="Analysis Zone"
        ).add_to(m)
        st_folium(m, width=500, height=400)

    with col_img:
        st.markdown("### NDVI Simulation")
        selected_day = st.slider("Select Date", min_value=0, max_value=len(df_master) - 1, value=150)
        current_ndvi = df_master['NDVI'].iloc[selected_day]
        current_temp = df_master['Temperature_C'].iloc[selected_day]

        st.info(
            f"**Date:** {df_master.index[selected_day].date()} | **Temperature:** {current_temp:.1f}°C | **NDVI:** {current_ndvi:.3f}")

        np.random.seed(selected_day)
        ndvi_matrix = np.clip(np.random.normal(loc=current_ndvi, scale=0.1, size=(50, 50)), 0, 1)

        fig_sat, ax_sat = plt.subplots(figsize=(6, 5))
        cax = ax_sat.imshow(ndvi_matrix, cmap='RdYlGn', vmin=0, vmax=1)
        ax_sat.axis('off')
        cbar = fig_sat.colorbar(cax, label='NDVI Value')
        cbar.set_label('NDVI Value', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        fig_sat.patch.set_facecolor('#0a0a1a')
        st.pyplot(fig_sat)

elif menu == "Reports & Documentation":
    st.title("Technical Documentation")

    st.markdown("""
    ### Forecasting Architecture

    The system employs a stacked Long Short-Term Memory (LSTM) neural network architecture designed for time series forecasting of vegetation indices. The model processes historical climate data including temperature, precipitation, and soil moisture to predict NDVI (Normalized Difference Vegetation Index) values.

    ### Model Specifications

    | Component | Description |
    |-----------|-------------|
    | Architecture | Stacked LSTM with 64 and 32 units |
    | Regularization | L2 regularization (0.001) with dropout layers (0.2) |
    | Loss Function | Huber loss with delta=1.0 |
    | Optimizer | Adam with adaptive learning rate |

    ### Data Sources

    | Source | Period | Days | Variables |
    |--------|--------|------|-----------|
    | Synthetic | 2021–2025 | 1826 | Temperature, Precipitation, Soil Moisture, NDVI |

    ### Data Processing Pipeline

    1. **Data Normalization**: Min-Max scaling applied to all features
    2. **Sequence Generation**: Sliding window approach with configurable look-back period
    3. **Train-Test Split**: 70% training, 30% validation
    4. **Early Stopping**: Monitors validation loss with 15 epochs patience

    ### Performance Metrics

    - **RMSE**: Root Mean Square Error - measures prediction accuracy
    - **MAE**: Mean Absolute Error - average magnitude of errors
    - **MAPE**: Mean Absolute Percentage Error - percentage-based accuracy measure
    - **R² Score**: Coefficient of determination - explains variance in predictions

    ### Scenario Generation

    Three climate scenarios are available for what-if analysis:
    - **Normal**: Seasonal patterns with moderate variability
    - **Extreme Drought**: High temperatures with zero precipitation
    - **Rainy Period**: Increased precipitation with moderate temperatures
    """)

    export_df = df_master[[c for c in df_master.columns if c != '_source']]
    csv_data = export_df.to_csv().encode('utf-8')
    st.download_button(
        "Export Combined Dataset (CSV)",
        data=csv_data,
        file_name='Data.csv',
        mime='text/csv',
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)
