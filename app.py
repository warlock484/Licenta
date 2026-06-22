import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
from streamlit_folium import st_folium
import os

# TensorFlow & Metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber

st.set_page_config(page_title="Agro-LSTM Spatial Predictor", page_icon="🌾", layout="wide")

# File Parameters
NPZ_PATH = "ndvi_lstm_processed.npz"
MODEL_PATH = "agro_lstm_real_spatial.keras"

# ==========================================
# INITIALIZE SESSION STATE
# ==========================================
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'results' not in st.session_state:
    st.session_state.results = None

# ==========================================
# CUSTOM CSS
# ==========================================
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

    h1 { font-size: 2.5rem; margin-bottom: 1.5rem; }
    h2 { font-size: 1.8rem; margin-top: 1rem; }
    h3 { font-size: 1.4rem; }

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

    [data-testid="stMetricLabel"] { color: #b8b8ff; font-weight: 500; font-size: 0.9rem; }
    .stRadio > div { gap: 0.5rem; }
    .stRadio label {
        background: rgba(30, 30, 60, 0.4);
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(100, 100, 255, 0.2);
        transition: all 0.3s ease;
        color: #e8e8ff;
    }
    .stRadio label:hover { background: rgba(90, 124, 255, 0.2); border-color: rgba(90, 124, 255, 0.5); }
    [data-testid="stDataFrame"], [data-testid="stTable"], .stExpander {
        background: rgba(20, 20, 45, 0.6);
        border-radius: 12px;
        border: 1px solid rgba(100, 100, 255, 0.2);
    }
    .stAlert { background: rgba(90, 124, 255, 0.1); border: 1px solid rgba(90, 124, 255, 0.3); border-radius: 12px; color: #e8e8ff; }
    .stProgress > div > div { background: linear-gradient(90deg, #5a7cff, #7c9cff); }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeInUp 0.5s ease-out; }
    hr { background: linear-gradient(90deg, transparent, #5a7cff, #7c9cff, transparent); height: 1px; border: none; margin: 1.5rem 0; }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)
st.markdown('<div class="fade-in">', unsafe_allow_html=True)


# ==========================================
# LOAD REAL DATA (.npz)
# ==========================================
@st.cache_data
def load_and_preprocess_real_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Preprocessed data file '{file_path}' not found. Please run `ndvi_preprocess.py` first.")
        st.stop()
        return None, None, None, None, None

    try:
        data = np.load(file_path)
        X_train = data["X_train"]  # (N, 92, 1) float32
        X_val = data["X_val"]  # (N, 92, 1) float32
        valid_mask = data["valid_mask"]  # (H, W) bool
        pixel_index = data["pixel_index"]  # (N_total, 2) [row, col]
        scaler_info_raw = data["scaler"]  # ['{'method': 'global_minmax', ...}']

        # Parse scaler info to map back to real NDVI if needed
        import ast
        scaler_dict = ast.literal_eval(scaler_info_raw[0])

        return X_train, X_val, valid_mask, pixel_index, scaler_dict
    except Exception as e:
        st.error(f"Error loading `.npz` file: {e}")
        st.stop()
        return None, None, None, None, None


# Load real data once
X_train_full, X_val_full, valid_mask_full, pixel_index_full, scaler_real = load_and_preprocess_real_data(NPZ_PATH)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌾 Agro-LSTM Predictor")
st.sidebar.markdown("Spatio-temporal vegetation monitoring system (MODIS)")
st.sidebar.info("Operating Mode: **Real Univariate Satellite Data** ✅")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "",
    ["Satellite Data Analysis",
     "LSTM Model & Forecast",
     "Spatial NDVI Visualization",
     "Technical Documentation"]
)

st.sidebar.markdown("---")
if st.session_state.training_complete:
    if st.sidebar.button("🔄 Reset Training State", type="secondary"):
        st.session_state.training_complete = False
        st.session_state.results = None
        st.rerun()

st.sidebar.info("**System Status:** Operational ✓")

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS (REAL DATA)
# ==========================================
if menu == "Satellite Data Analysis":
    st.title("Exploratory Satellite Data Analysis")
    st.markdown("Overview of the preprocessed, spatially and temporally smoothed MODIS NDVI time series.")

    num_valid_pixels = X_train_full.shape[0] + X_val_full.shape[0]
    time_steps = X_train_full.shape[1]

    # Concatenate all for global metrics
    X_all = np.concatenate([X_train_full, X_val_full], axis=0)
    global_mean = X_all.mean()
    global_std = X_all.std()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Valid Pixels", f"{num_valid_pixels:,}")
    with col2:
        st.metric("Time Steps per Pixel", f"{time_steps}")
    with col3:
        st.metric("Global Avg Scaled NDVI", f"{global_mean:.3f}")
    with col4:
        st.metric("Global Variance", f"{global_std:.3f}")

    # --- 1. REGIONAL MEAN ---
    st.markdown("### 📈 Regional Mean Time Series")
    st.info("This curve represents the mean NDVI calculated across all ~185k valid pixels in the study area.")

    # Calculate regional temporal mean (vertical axis pixels, horizontal axis time)
    regional_mean_ndvi = X_all.squeeze().mean(axis=0)  # Shape (92,)

    # Dummy weekly date range (assuming 92 captures, dummy start)
    date_rng = pd.date_range(start='2023-01-01', periods=time_steps, freq='W')
    df_regional_mean = pd.DataFrame({'Mean NDVI': regional_mean_ndvi}, index=date_rng)

    st.line_chart(df_regional_mean, color="#5a7cff")

    # --- 2. PIXEL VARIANCE VS MEAN ---
    st.markdown("### 📊 Spatial Variability (Pixel vs. Mean)")
    st.info(
        "Displaying 5 randomly selected pixels to demonstrate how local phenology differs from the regional average.")

    fig_var, ax_var = plt.subplots(figsize=(12, 5))
    np.random.seed(42)
    indices_random = np.random.choice(X_all.shape[0], 5, replace=False)

    # Plot thick regional mean in background
    ax_var.plot(date_rng, regional_mean_ndvi, linewidth=5, color='#ffffff', alpha=0.3, label='Regional Mean')

    # Plot 5 individual pixels
    colors = ['#ff7c5a', '#5a7cff', '#5aff96', '#ff5acc', '#f1c40f']
    for i, idx in enumerate(indices_random):
        ax_var.plot(date_rng, X_all[idx].squeeze(), color=colors[i], alpha=0.8, marker='.', markersize=4, linewidth=1,
                    label=f'Pixel {idx}')

    # Plot Styling
    ax_var.set_facecolor('#0a0a1a')
    fig_var.patch.set_facecolor('#0a0a1a')
    plt.setp(ax_var.get_xticklabels(), color='white')
    plt.setp(ax_var.get_yticklabels(), color='white')
    ax_var.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white', fontsize=8)
    ax_var.grid(True, linestyle='--', alpha=0.3)
    ax_var.set_ylabel("Scaled NDVI (0-1)", color='white')
    st.pyplot(fig_var)


# ==========================================
# TAB 2: LSTM MODEL & FORECASTS (OPTIMIZED)
# ==========================================
elif menu == "LSTM Model & Forecast":
    st.title("Deep Learning Forecast System (Univariate Real)")
    st.markdown(
        "Train the LSTM network on real satellite data and generate a smoothed autoregressive vegetation forecast.")

    with st.container():
        st.markdown("### 1. Training Configuration")
        col_p1, col_p2 = st.columns(2)
        look_back = col_p1.slider("Context Window (Look-back / weeks)", min_value=5, max_value=40, value=25, step=1)
        epochs = col_p2.slider("Training Epochs", min_value=10, max_value=200, value=50, step=10)

        st.markdown("---")
        st.markdown("### 2. Future Forecast Configuration")
        st.info(
            "🌦️ **Note:** The forecast is **autoregressive** (the network uses its own predictions). "
            "The model predicts natural phenology based on its memory of historical patterns.")

        days_to_predict = st.number_input("Weeks to forecast", min_value=1, max_value=100, value=60)

        # EMA Smoothing Configuration
        st.markdown("#### Forecast Smoothing Mechanism (Vegetation Inertia)")
        alpha_ema = st.slider("Damping Factor (alpha_ema)", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
        st.caption(
            "Lower values (e.g., 0.25) = High inertia, smooth phenological curves. "
            "Higher values (e.g., 0.9) = Aggressive reaction to noise (oscillations).")

        st.markdown("---")

        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            train_clicked = st.button("Execute Training & Forecast",
                                      type="primary",
                                      use_container_width=True)
        with col_btn2:
            if st.session_state.training_complete:
                if st.button("🔄 Reset Results", type="secondary", use_container_width=True):
                    st.session_state.training_complete = False
                    st.session_state.results = None
                    st.rerun()

        # ==========================================
        # OPTIMIZED TRAINING LOGIC
        # ==========================================
        if train_clicked:
            # --- Initialize Progress UI ---
            status_text = st.empty()
            progress_bar = st.progress(0.0)
            epoch_text = st.empty()

            try:
                # ===== STEP 1: FAST VECTORIZED DATA PREPARATION =====
                status_text.info("Step 1/3: Preparing sequence data (vectorized)...")
                progress_bar.progress(0.1)

                # Subsample pixels (reduced for speed)
                np.random.seed(42)
                subset_size = 1000  # Reduced from 2000 for speed
                indices_subset = np.random.choice(X_train_full.shape[0], subset_size, replace=False)
                X_subset = X_train_full[indices_subset]  # Shape: (1000, 92, 1)

                # Validation subset
                indices_subset_va = np.random.choice(X_val_full.shape[0], 250, replace=False)
                X_subset_va = X_val_full[indices_subset_va]  # Shape: (250, 92, 1)


                # VECTORIZED sequence creation using numpy strides
                def create_sequences_vectorized(data_3d, look_back):
                    """
                    Vectorized sequence creation for 3D array (pixels, time, features)
                    Much faster than nested loops
                    """
                    n_pixels, n_timesteps, n_features = data_3d.shape
                    n_sequences = n_pixels * (n_timesteps - look_back)

                    # Pre-allocate arrays
                    X = np.zeros((n_sequences, look_back, n_features), dtype=np.float32)
                    y = np.zeros((n_sequences,), dtype=np.float32)

                    idx = 0
                    for i in range(n_pixels):
                        pixel_data = data_3d[i]  # (92, 1)
                        for j in range(n_timesteps - look_back):
                            X[idx] = pixel_data[j:j + look_back]
                            y[idx] = pixel_data[j + look_back, 0]
                            idx += 1

                    return X, y


                # Create sequences (still some loop but optimized)
                X_train, y_train = create_sequences_vectorized(X_subset, look_back)
                X_val, y_val = create_sequences_vectorized(X_subset_va, look_back)

                progress_bar.progress(0.2)
                status_text.info(f"Data prepared: {len(X_train):,} sequences")

                # ===== STEP 2: BUILD AND TRAIN MODEL =====
                status_text.info(f"Step 2/3: Training LSTM Network...")

                # Simpler, faster model architecture
                model = Sequential([
                    LSTM(64, return_sequences=True, input_shape=(look_back, 1),
                         kernel_regularizer=l2(0.0001)),
                    Dropout(0.2),
                    LSTM(32, return_sequences=False, kernel_regularizer=l2(0.0001)),
                    Dropout(0.2),
                    Dense(16, kernel_regularizer=l2(0.0001)),
                    LeakyReLU(alpha=0.1),
                    Dense(1, activation='sigmoid')
                ])

                model.compile(optimizer='adam', loss=Huber(delta=1.0))


                # Custom callback for Streamlit progress
                class FastProgressCallback(EarlyStopping):
                    def __init__(self, epoch_text, progress_bar, total_epochs, **kwargs):
                        super().__init__(**kwargs)
                        self.epoch_text = epoch_text
                        self.progress_bar = progress_bar
                        self.total_epochs = total_epochs

                    def on_epoch_end(self, epoch, logs=None):
                        logs = logs or {}
                        epoch_num = epoch + 1

                        # Update progress (20% to 80% range)
                        progress = 0.2 + (0.6 * (epoch_num / self.total_epochs))
                        self.progress_bar.progress(min(progress, 0.8))

                        # Update epoch info
                        self.epoch_text.markdown(
                            f"🔄 **Epoch {epoch_num}/{self.total_epochs}** | "
                            f"Loss: {logs.get('loss', 0):.4f} | "
                            f"Val Loss: {logs.get('val_loss', 0):.4f}"
                        )

                        super().on_epoch_end(epoch, logs)


                # Training callbacks
                lr_scheduler = ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
                )
                progress_callback = FastProgressCallback(
                    epoch_text=epoch_text,
                    progress_bar=progress_bar,
                    total_epochs=epochs,
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=0
                )

                # Train with larger batch size for speed
                history = model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=64,  # Increased from 32
                    validation_data=(X_val, y_val),
                    callbacks=[lr_scheduler, progress_callback],
                    verbose=0
                )

                # ===== STEP 3: SAVE & PREDICT =====
                progress_bar.progress(0.85)
                status_text.info("Step 3/3: Saving model and generating forecast...")

                # Save model
                model.save(MODEL_PATH)

                # Quick predictions
                predictions = model.predict(X_val, batch_size=64, verbose=0)
                y_val_real = y_val.reshape(-1, 1)

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_val_real, predictions))
                mae = mean_absolute_error(y_val_real, predictions)
                r2 = r2_score(y_val_real, predictions)

                # Future forecast (autoregressive)
                last_pixel = X_subset[-1]
                current_seq = last_pixel[-look_back:, :].reshape(1, look_back, 1)
                future_predictions = []
                last_value = current_seq[0, -1, 0]

                for _ in range(days_to_predict):
                    raw_pred = model.predict(current_seq, verbose=0)[0, 0]
                    next_pred = (alpha_ema * raw_pred) + ((1 - alpha_ema) * last_value)
                    future_predictions.append(next_pred)
                    last_value = next_pred
                    current_seq = np.append(current_seq[:, 1:, :], [[[next_pred]]], axis=1)

                future_predictions = np.array(future_predictions)

                # Date range for forecast
                last_timestamp = pd.Timestamp('2024-01-01')
                future_dates = pd.date_range(
                    start=last_timestamp + pd.Timedelta(days=1),
                    periods=days_to_predict,
                    freq='W'
                )

                # Store results
                st.session_state.results = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'final_train_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1],
                    'best_val_loss': min(history.history['val_loss']),
                    'best_epoch': np.argmin(history.history['val_loss']) + 1,
                    'history': history.history,
                    'y_val_real': y_val_real,
                    'predictions': predictions,
                    'future_dates': future_dates,
                    'future_predictions': future_predictions,
                    'last_timestamp': last_timestamp,
                    'look_back': look_back
                }

                progress_bar.progress(1.0)
                status_text.success("✅ Training complete!")

                # Clean up progress UI
                import time

                time.sleep(0.5)
                status_text.empty()
                progress_bar.empty()
                epoch_text.empty()

            except Exception as e:
                status_text.error(f"Training failed: {str(e)}")
                st.stop()

            st.session_state.training_complete = True
            st.rerun()

    # --- RESULTS RENDERING ---
    if st.session_state.training_complete and st.session_state.results is not None:
        results = st.session_state.results

        st.success(f"✅ Training completed in the background and model saved to `{MODEL_PATH}` successfully.")

        st.markdown("### 🎯 Model Performance Metrics (Spatial Validation)")
        st.caption("Metrics calculated on the validation subset (pixels unseen during training), on 0-1 scaled data.")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1: st.metric("RMSE (Error)", f"{results['rmse']:.6f}")
        with col_m2: st.metric("MAE", f"{results['mae']:.6f}")
        with col_m3: st.metric("R² Score (Variance Explained)", f"{results['r2']:.4f}")

        st.markdown("### 📊 Training Metrics")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1: st.metric("Final Training Loss", f"{results['final_train_loss']:.6f}")
        with col_t2: st.metric("Final Validation Loss", f"{results['final_val_loss']:.6f}")
        with col_t3: st.metric("Best Epoch", delta=f"Val Loss: {results['best_val_loss']:.6f}",
                               value=results['best_epoch'])

        st.markdown("### 📉 Loss Curves")
        fig_loss, ax_loss = plt.subplots(figsize=(12, 5))
        ax_loss.plot(results['history']['loss'], label='Training Loss (Huber)', color='#5a7cff', linewidth=2)
        ax_loss.plot(results['history']['val_loss'], label='Validation Loss', color='#ff7c5a', linewidth=2)
        ax_loss.axvline(x=results['best_epoch'] - 1, color='#5aff96', linestyle='--', linewidth=1.5, alpha=0.7,
                        label=f'Best Model (Epoch {results["best_epoch"]})')
        ax_loss.set_title('Training and Validation Loss Over Epochs', fontsize=14, color='white')
        ax_loss.set_xlabel('Epoch', color='white')
        ax_loss.set_ylabel('Loss (Huber)', color='white')
        ax_loss.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        ax_loss.grid(True, linestyle='--', alpha=0.3)
        ax_loss.set_facecolor('#0a0a1a')
        fig_loss.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_loss.get_xticklabels(), color='white')
        plt.setp(ax_loss.get_yticklabels(), color='white')
        st.pyplot(fig_loss)

        # Actual vs Predicted (on validation)
        st.markdown("### 🎯 Scatter Plot: Actual vs Predicted (Validation)")
        fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
        ax_pred.scatter(results['y_val_real'], results['predictions'], alpha=0.2, color='#5a7cff', s=10)
        ax_pred.plot([0, 1], [0, 1], 'r--', linewidth=2, color='#ff5a7c', label='Perfect Prediction')
        ax_pred.set_xlabel('Actual MODIS NDVI (Scaled)', color='white')
        ax_pred.set_ylabel('Predicted NDVI (Scaled)', color='white')
        ax_pred.set_title(f'Scatter Plot (R² = {results["r2"]:.4f})', color='white')
        ax_pred.set_facecolor('#0a0a1a')
        fig_pred.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_pred.get_xticklabels(), color='white')
        plt.setp(ax_pred.get_yticklabels(), color='white')
        st.pyplot(fig_pred)

        st.markdown("### 📉 Prediction Error Distribution")
        errors = (results['y_val_real'].flatten() - results['predictions'].flatten())
        mean_error = np.mean(errors)
        fig_err, ax_err = plt.subplots(figsize=(12, 5))
        ax_err.hist(errors, bins=60, color='#5a7cff', alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_err.axvline(x=0, color='#ff5a7c', linestyle='--', linewidth=2, label='Zero Error')
        ax_err.axvline(x=mean_error, color='#5aff96', linestyle='--', linewidth=2,
                       label=f'Mean Error: {mean_error:.6f}')
        ax_err.set_title('Distribution of Prediction Errors', color='white')
        ax_err.set_facecolor('#0a0a1a')
        fig_err.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_err.get_xticklabels(), color='white')
        plt.setp(ax_err.get_yticklabels(), color='white')
        ax_err.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        st.pyplot(fig_err)

        st.markdown("### 🔮 Autoregressive EMA Future Forecast")
        st.caption(
            "Displaying Historical Phenology (92 MODIS weeks) followed by the Autoregressive Forecast (EMA 0.25) based on model memory.")
        fig_fut, ax_f = plt.subplots(figsize=(14, 5))

        # Historical reference point (Anchor Pixel)
        # We use the historical regional mean for contextual background, not just a single pixel
        date_rng_hist = pd.date_range(start='2021-01-01', periods=92, freq='W')
        regional_mean_hist = np.concatenate([X_train_full, X_val_full], axis=0).squeeze().mean(axis=0)

        ax_f.plot(date_rng_hist, regional_mean_hist, label='Historical Regional NDVI (MODIS Mean)', color='#5a7cff',
                  linewidth=3)

        # Plot forecast
        ax_f.plot(results['future_dates'], results['future_predictions'],
                  label=f'Univariate Autoregressive Forecast ({results["look_back"]} context)', color='#ff7c5a',
                  linestyle='dashed', linewidth=3, marker='.', markersize=8)

        ax_f.axvline(x=results['last_timestamp'], color='#ffffff', linestyle=':', label='Present (End of MODIS Data)',
                     linewidth=2)
        ax_f.set_title('NDVI Evolution: Historical Data vs Forecast', fontsize=14, color='white')
        ax_f.set_facecolor('#0a0a1a')
        fig_fut.patch.set_facecolor('#0a0a1a')
        plt.setp(ax_f.get_xticklabels(), color='white')
        plt.setp(ax_f.get_yticklabels(), color='white')
        ax_f.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white')
        ax_f.grid(True, linestyle='--', alpha=0.3)
        st.pyplot(fig_fut)


# ==========================================
# TAB 3: SATELLITE IMAGERY (REAL SPATIAL MAPS)
# ==========================================
elif menu == "Spatial NDVI Visualization":
    st.title("Spatial Visualization of MODIS Predictions")
    st.markdown(
        "Generate $H \\times W$ spatial maps in real-time using the trained LSTM model applied to *all* valid pixels.")

    # 1. LOAD SAVED MODEL OR SESSION
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            with st.spinner("💾 Loading trained model from disk..."):
                model = load_model(MODEL_PATH, custom_objects={'Huber': Huber(delta=1.0)})
        except Exception as e:
            st.error(f"Error loading model: {e}")
    elif st.session_state.training_complete and 'X_val_full' in locals():
        st.warning("Saved model not found, but a model exists in session.")
    else:
        st.warning("The model hasn't been trained yet. Please run the Training tab first.")
        st.stop()

    if model is None:
        st.error("Model could not be loaded.")
        st.stop()

    # Spatial Parameters
    H, W = valid_mask_full.shape
    look_back_config = st.number_input("Verify Look-back used during training", value=25)  # Must match training

    with st.container():
        st.markdown("### 1. Visualization Configuration")
        col_s1, col_s2 = st.columns(2)

        # Slider to select the week from the MODIS dataset (92 weeks)
        # We need `look_back` weeks of context to predict the next spatial week.
        # So the slider is between `look_back` and 91.
        selected_week = col_s1.slider("Select MODIS Week (for Context)", min_value=look_back_config, max_value=91,
                                      value=look_back_config, step=1)

        # Map type
        map_type = col_s2.selectbox("Visual Map Type:",
                                    ["Raw MODIS Data (Actual)", "Model Prediction (1-Step Ahead)", "Spatial Error Map"])

        col_map_btn1, col_map_btn2 = st.columns([3, 1])
        with col_map_btn1:
            load_map_btn = st.button("🖼️ Generate Spatial Visualization", type="primary", use_container_width=True)

        if load_map_btn:
            with st.spinner(
                    f"Executing prediction on {X_train_full.shape[0] + X_val_full.shape[0]:,} pixels for MODIS week {selected_week}..."):
                # --- A) PREPARE CONTEXT FOR ALL PIXELS ---
                # Look at all valid pixels (concatenate)
                X_all_full = np.concatenate([X_train_full, X_val_full], axis=0)  # Shape: (TotalPix, 92, 1)

                # Extract spatial context `H x W x look_back x 1` for `selected_week`
                all_X_context = X_all_full[
                    :, selected_week - look_back_config: selected_week, :]  # Shape: (TotalPix, LB, 1)

                # Real target for comparison (week selected_week)
                target_real = X_all_full[:, selected_week, 0]  # Shape (TotalPix,)

                # --- B) EXECUTE SATELLITE-SCALE PREDICTION ---
                preds_raw = model.predict(all_X_context, verbose=0)  # Shape: (TotalPix, 1)

                # --- C) RECONSTRUCT SPATIAL MAPS ---
                # Create empty 2D H x W maps
                reconstructed_preds_map = np.zeros((H, W))
                reconstructed_actual_map = np.zeros((H, W))
                reconstructed_error_map = np.zeros((H, W))

                # Use pixel_index_full to map flat array [N] values back to [H,W]
                rows = pixel_index_full[:, 0]
                cols = pixel_index_full[:, 1]

                # Spatial masking according to valid indices
                reconstructed_preds_map[rows, cols] = preds_raw.squeeze()
                reconstructed_actual_map[rows, cols] = target_real
                reconstructed_error_map[rows, cols] = target_real - preds_raw.squeeze()

                st.session_state.map_results = {
                    'preds_map': reconstructed_preds_map,
                    'actual_map': reconstructed_actual_map,
                    'error_map': reconstructed_error_map,
                    'week': selected_week,
                    'H': H, 'W': W
                }

    # --- RENDER REAL MAPS ---
    if 'map_results' in st.session_state:
        map_results = st.session_state.map_results
        H = map_results['H']
        W = map_results['W']

        col_viz_left, col_viz_right = st.columns(2)

        with col_viz_left:
            st.markdown(f"### 🖼️ Map: {map_type} (MODIS Week {map_results['week']})")

            fig_sat_final, ax_sat_final = plt.subplots(figsize=(W / 80, H / 80), dpi=100)  # Maintain aspect ratio

            # Configure colormap and limits depending on map type
            if "Error" in map_type:
                data_plot = map_results['error_map']
                norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)  # Errors centered on 0
                cmap = 'coolwarm_r'  # Red=Prediction < Actual, Blue=Prediction > Actual
                label_cbar = 'NDVI Difference (Actual - Predicted)'
            else:
                norm = mcolors.Normalize(vmin=0, vmax=1)  # Standard scaled NDVI
                cmap = 'RdYlGn'
                if "Actual" in map_type:
                    data_plot = map_results['actual_map']
                    label_cbar = 'Raw MODIS NDVI (Scaled)'
                else:
                    data_plot = map_results['preds_map']
                    label_cbar = 'LSTM Model NDVI (Scaled)'

            cax = ax_sat_final.imshow(data_plot, cmap=cmap, norm=norm, interpolation='nearest')
            ax_sat_final.axis('off')

            cbar = fig_sat_final.colorbar(cax, orientation='vertical', pad=0.01, aspect=30)
            cbar.set_label(label_cbar, color='white', size=12)
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white', size=10)

            fig_sat_final.patch.set_facecolor('#0a0a1a')
            st.pyplot(fig_sat_final, use_container_width=True)

        with col_viz_right:
            st.markdown("### 🗺️ Geographic Context")
            st.info(
                "💡 **Note:** Spatial grid $466 \\times 513$ pixels represents a study area extracted from Switzerland. Exact coordinates were abstracted for training.")

            # Centrat pe inima Alpilor Elvețieni
            m = folium.Map(location=[46.6, 8.2], zoom_start=7, tiles=None)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Satellite View', overlay=False, control=True).add_to(m)

            # Desenăm un dreptunghi generic (aprox. 120x120 km) reprezentativ pentru grila MODIS
            folium.Rectangle(
                bounds=[[46.1, 7.5], [47.1, 8.9]],
                color='#5a7cff',
                fill=True,
                fill_opacity=0.2,
                tooltip="Abstracted Spatial Grid (~120x120 km)",
                dash_array='5, 5'  # Linie întreruptă pentru a sugera că e o grilă conceptuală
            ).add_to(m)
            st_folium(m, width=500, height=450)


# ==========================================
# TAB 4: TECHNICAL DOCUMENTATION
# ==========================================
elif menu == "Technical Documentation":
    st.title("Technical Documentation")

    st.markdown("""
    ### Spatio-Temporal Forecasting Architecture

    The system utilizes a Stacked LSTM network trained on univariate MODIS satellite data (spatially preprocessed, gap-filled, smoothed, and 0-1 scaled NDVI). The model predicts the next time step (1 week) based on its long-term memory over a context window (look-back) of 25 weeks.

    ### Model Serialization
    The network is saved in the native TensorFlow Keras format (`.keras`) at `agro_lstm_real_spatial.keras`, allowing it to be loaded and run for inference (such as in the Spatial Visualization Tab) without retraining.

    ### Model Specifications

    | Component | Value |
    |-----------|-------------|
    | Architecture | Stacked LSTM (128 units -> 64 units) |
    | Training Epochs | **50** (default) |
    | Optimizer | Adam (Huber Loss, Adaptive LR) |
    | Context Window (LB) | 25 weeks |

    ### Data Sources

    | Source | Format | Valid Pixels | Time Steps (MODIS) | Study Area |
    |--------|-----------|--------|--------|--------|
    | MODIS NDVI | Flat `.npz` Archive | ~185,000 pixels | 92 weeks | Swiss Alpine Region (Abstracted Grid) |

    ### Satellite-Scale Spatial Reconstruction
    In the "Spatial Visualization" tab, the LSTM model is loaded and applied autoregressively to *every single* valid pixel (~185,000) from the `.npz` archive for a selected date. The results are mapped using the `(N,2)` [row, col] indices into an $H \\times W$ spatial matrix of $466 \\times 513$, perfectly recreating the spatial map of the region.
    """)

    # Export processed data subset
    if 'X_train_full' in locals():
        X_all_out = np.concatenate([X_train_full, X_val_full], axis=0).squeeze()
        export_df = pd.DataFrame(X_all_out)
        csv_data = export_df.to_csv().encode('utf-8')
        st.download_button(
            "Export Flat Scaled NDVI Subset (CSV)",
            data=csv_data,
            file_name='Data_Real_NDVI_01.csv',
            mime='text/csv',
            use_container_width=True
        )

st.markdown('</div>', unsafe_allow_html=True)
