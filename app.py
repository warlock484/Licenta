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
ROI_PATH = "Date_reale/roi2.npy"      # raw NDVI cube (H, W, T), used for preprocessing demos
ROI_PATH_NPZ = "Date_reale/roi2.npz"  # compressed float16 fallback (deploy-friendly, < 100 MB)

# ==========================================
# DATASET FACTS (Sentinel-2)
# ==========================================
# The dataset is a real Sentinel-2 NDVI cube (roi2.npy -> 466 x 513 x 92) from
# the IEEE DataPort "Sentinel-2 super-resolved data cubes" collection (Griparis
# et al., CEOSpaceTech): tile T32TLT, Laegeren / Harth, northern Switzerland.
# The 92 cloud-free L2A scenes (bands CNN super-resolved to 10 m) span 2018-2022
# (5 years) and are NON-uniformly sampled (no winter scenes; concentrated in the
# growing season) -> NOT a fixed weekly cadence. Exact per-scene dates ship as a
# separate metadata CSV with the source dataset, not inside the .npy cube.
SENSOR = "Sentinel-2"
TILE = "T32TLT"
REGION = "Laegeren / Harth, northern Switzerland"
N_ACQ = 92            # temporal acquisitions per pixel
SPAN_YEARS = 5        # 2018-2022, per the source dataset (5 growing seasons)


def acquisition_dates(n=N_ACQ):
    """Display x-axis for the temporal series.

    The exact Sentinel-2 scene dates are not stored in the processed .npy cube
    (they ship as a separate metadata CSV with the source dataset). The 92
    cloud-free scenes span 2018-2022 and are non-uniformly sampled. For plotting
    we spread them evenly across that span; treat the axis as an approximate
    acquisition timeline, not exact calendar dates.
    """
    return pd.date_range(start="2018-01-01", periods=n, freq="20D")


def to_real_ndvi(scaled, scaler):
    """Map [0,1] globally-scaled values back to physical NDVI.

    The preprocessing used global min-max: scaled = (ndvi - lo) / (hi - lo).
    Inverting it recovers the real NDVI range stored in the scaler metadata.
    """
    if not scaler or scaler.get("method") != "global_minmax":
        return scaled
    lo, hi = scaler["lo"], scaler["hi"]
    return scaled * (hi - lo) + lo

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
        st.error(f"Preprocessed data file '{file_path}' not found. Please run ndvi_preprocess.py first.")
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
        st.error(f"Error loading .npz file: {e}")
        st.stop()
        return None, None, None, None, None


# Load real data once
X_train_full, X_val_full, valid_mask_full, pixel_index_full, scaler_real = load_and_preprocess_real_data(NPZ_PATH)

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🌾 Agro-LSTM Predictor")
st.sidebar.markdown(f"Spatio-temporal vegetation monitoring system ({SENSOR})")
st.sidebar.info(f"Operating Mode: *Real {SENSOR} NDVI (tile {TILE})* ✅")

st.sidebar.markdown("---")

menu = st.sidebar.radio(
    "",
    ["Satellite Data Analysis",
     "Preprocessing & Example Scene",
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

st.sidebar.info("*System Status:* Operational ✓")

# ==========================================
# TAB 1: EXPLORATORY DATA ANALYSIS (REAL DATA)
# ==========================================
if menu == "Satellite Data Analysis":
    st.title("Exploratory Satellite Data Analysis")
    st.markdown(
        f"Overview of the preprocessed, spatially and temporally smoothed *{SENSOR}* NDVI time series "
        f"(tile *{TILE}*, {REGION}).")

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
        st.metric("Acquisitions per Pixel", f"{time_steps}", help=f"Cloud-free {SENSOR} layers over ~{SPAN_YEARS} years")
    with col3:
        st.metric("Global Avg Scaled NDVI", f"{global_mean:.3f}")
    with col4:
        st.metric("Global Std (Scaled)", f"{global_std:.3f}")

    # --- Scaling reference: how [0,1] maps to physical NDVI ---
    with st.expander("ℹ️ How the [0,1] scaling maps to physical NDVI"):
        if scaler_real and scaler_real.get("method") == "global_minmax":
            lo, hi = scaler_real["lo"], scaler_real["hi"]
            st.markdown(
                f"All curves are stored as *globally min-max scaled* values in $[0,1]$. "
                f"The mapping is $\\text{{scaled}} = (\\text{{NDVI}} - {lo:.3f}) / ({hi:.3f} - ({lo:.3f}))$, so "
                f"0.0 corresponds to physical NDVI *{lo:.3f}* and 1.0 to *{hi:.3f}*. "
                f"This preserves the relative amplitude of the NDVI signal (no per-pixel re-centring).")
        else:
            st.markdown("Curves are stored as scaled values; no global min-max metadata is available.")

    # Display axis spanning the ~5-year acquisition window
    date_rng = acquisition_dates(time_steps)

    # --- 1. REGIONAL MEAN (scaled AND physical NDVI) ---
    st.markdown("### 📈 Regional Mean Time Series")
    st.info(
        f"Mean NDVI averaged across all {num_valid_pixels:,} valid vegetation pixels. The repeated peaks correspond "
        f"to successive growing seasons captured over the ~{SPAN_YEARS}-year acquisition window.")

    regional_mean_scaled = X_all.squeeze().mean(axis=0)  # (92,)
    regional_mean_real = to_real_ndvi(regional_mean_scaled, scaler_real)

    show_real = st.toggle("Show physical NDVI values instead of [0,1] scaled", value=True)
    series_to_plot = regional_mean_real if show_real else regional_mean_scaled
    y_label = "NDVI (physical)" if show_real else "NDVI (scaled 0-1)"
    df_regional_mean = pd.DataFrame({y_label: series_to_plot}, index=date_rng)
    st.line_chart(df_regional_mean, color="#5aff96" if show_real else "#5a7cff")

    # --- 2. PIXEL VARIANCE VS MEAN ---
    st.markdown("### 📊 Spatial Variability (Pixel vs. Mean)")
    st.info(
        "Five randomly selected pixels overlaid on the regional mean, showing how local phenology "
        "(crop type, management, soil) deviates from the regional average.")

    conv = (lambda a: to_real_ndvi(a, scaler_real)) if show_real else (lambda a: a)
    fig_var, ax_var = plt.subplots(figsize=(12, 5))
    np.random.seed(42)
    indices_random = np.random.choice(X_all.shape[0], 5, replace=False)

    # Plot thick regional mean in background
    ax_var.plot(date_rng, conv(regional_mean_scaled), linewidth=5, color='#ffffff', alpha=0.3, label='Regional Mean')

    # Plot 5 individual pixels
    colors = ['#ff7c5a', '#5a7cff', '#5aff96', '#ff5acc', '#f1c40f']
    for i, idx in enumerate(indices_random):
        ax_var.plot(date_rng, conv(X_all[idx].squeeze()), color=colors[i], alpha=0.8, marker='.', markersize=4,
                    linewidth=1, label=f'Pixel {idx}')

    # Plot Styling
    ax_var.set_facecolor('#0a0a1a')
    fig_var.patch.set_facecolor('#0a0a1a')
    plt.setp(ax_var.get_xticklabels(), color='white')
    plt.setp(ax_var.get_yticklabels(), color='white')
    ax_var.legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white', fontsize=8)
    ax_var.grid(True, linestyle='--', alpha=0.3)
    ax_var.set_ylabel(y_label, color='white')
    st.pyplot(fig_var)


# ==========================================
# TAB 1b: PREPROCESSING & EXAMPLE SCENE
# ==========================================
elif menu == "Preprocessing & Example Scene":
    st.title("Preprocessing Pipeline & Example NDVI Scene")
    st.markdown(
        f"How a raw *{SENSOR}* NDVI cube (roi2.npy, "
        f"{valid_mask_full.shape[0]} × {valid_mask_full.shape[1]} × {N_ACQ}) becomes model-ready sequences: "
        f"cloud/spike flagging → linear gap interpolation → Savitzky-Golay smoothing → global min-max scaling to [0,1].")

    @st.cache_data
    def load_raw_cube(npy_path, npz_path):
        # Prefer the full-resolution .npy locally; fall back to the compressed
        # float16 .npz that is small enough to ship to the cloud (< 100 MB).
        for path in (npy_path, npz_path):
            if not os.path.exists(path):
                continue
            try:
                arr = np.load(path)
                if isinstance(arr, np.lib.npyio.NpzFile):  # .npz archive
                    scale = float(arr["scale"]) if "scale" in arr else 1.0
                    cube_arr = arr["cube"].astype(np.float32)
                    if np.issubdtype(arr["cube"].dtype, np.integer):
                        cube_arr /= scale  # int8-quantized -> NDVI in [-1, 1]
                    return cube_arr  # (H, W, T)
                return arr.astype(np.float32)  # (H, W, T)
            except (ValueError, OSError, KeyError):
                # File present but unreadable (e.g. a Git LFS pointer that wasn't
                # fetched on the deploy host). Try the next candidate / give up.
                continue
        return None

    cube = load_raw_cube(ROI_PATH, ROI_PATH_NPZ)
    if cube is None:
        st.warning(
            f"Raw cube not found (looked for {ROI_PATH} and {ROI_PATH_NPZ}), so the example scene "
            "and before/after demo are unavailable. The processed .npz is still used by the other pages.")
        st.stop()

    H, W, T = cube.shape

    # ---------- 1. EXAMPLE SCENE (single acquisition map) ----------
    st.markdown("### 1. Example NDVI Scene (single acquisition)")
    t_idx = st.slider("Acquisition index", 0, T - 1, T // 2)
    scene = cube[:, :, t_idx].copy()
    scene[~np.isfinite(scene)] = np.nan
    scene[(scene <= -1.0) | (scene > 1.0)] = np.nan
    scene[~valid_mask_full] = np.nan  # blank pixels excluded by the validity mask

    fig_sc, ax_sc = plt.subplots(figsize=(W / 90, H / 90), dpi=100)
    cmap_sc = plt.get_cmap('RdYlGn').copy()
    cmap_sc.set_bad('#0a0a1a')
    cax_sc = ax_sc.imshow(scene, cmap=cmap_sc, vmin=-0.1, vmax=1.0, interpolation='nearest')
    ax_sc.axis('off')
    cbar_sc = fig_sc.colorbar(cax_sc, ax=ax_sc, fraction=0.046, pad=0.02)
    cbar_sc.set_label('Raw NDVI', color='white')
    cbar_sc.ax.yaxis.set_tick_params(color='white')
    plt.setp(plt.getp(cbar_sc.ax.axes, 'yticklabels'), color='white')
    fig_sc.patch.set_facecolor('#0a0a1a')
    st.pyplot(fig_sc)
    st.caption(f"Acquisition {t_idx}/{T - 1}. Dark = no-data / non-vegetation pixels excluded by the validity mask.")

    # ---------- 2. BEFORE / AFTER PER-PIXEL ----------
    st.markdown("### 2. Per-pixel preprocessing: raw vs cleaned")
    st.info(
        "Pick valid pixels to compare the raw NDVI samples (red dots; clouds appear as sharp downward drops) "
        "with the cleaned curve after cloud flagging, linear gap-fill and Savitzky-Golay smoothing.")

    valid_rows, valid_cols = np.where(valid_mask_full)
    n_demo = st.slider("How many random pixels", 1, 4, 3)
    np.random.seed(0)
    sel = np.random.choice(len(valid_rows), n_demo, replace=False)

    def preprocess_pixel(series, cloud_thresh=0.0, spike_drop=0.30, window=7, poly=2):
        """Single-pixel version of the ndvi_preprocess.py pipeline (steps 2-4)."""
        from scipy.signal import savgol_filter
        s = series.astype(float).copy()
        s[(series <= cloud_thresh) | (series > 1.0) | ~np.isfinite(series)] = np.nan
        pad = np.pad(np.nan_to_num(s, nan=np.nan), 1, mode='edge')
        med = np.array([np.nanmedian(pad[i:i + 3]) for i in range(len(s))])
        s[(med - s) > spike_drop] = np.nan
        x = np.arange(len(s))
        nan = np.isnan(s)
        if nan.any() and (~nan).any():
            s[nan] = np.interp(x[nan], x[~nan], s[~nan])
        w = min(window, len(s) // 2 * 2 + 1)
        if w > poly:
            s = savgol_filter(s, w, poly)
        return np.clip(s, -1.0, 1.0)

    dates = acquisition_dates(T)
    fig_ba, axes = plt.subplots(n_demo, 1, figsize=(12, 2.6 * n_demo), squeeze=False)
    for ax, k in zip(axes[:, 0], sel):
        r, c = valid_rows[k], valid_cols[k]
        raw = cube[r, c, :]
        clean = preprocess_pixel(raw)
        ax.plot(dates, raw, '.', color='#ff5a7c', ms=5, alpha=0.5, label='raw NDVI (clouds = drops)')
        ax.plot(dates, clean, '-', color='#5aff96', lw=2, label='cleaned + Savitzky-Golay')
        ax.set_title(f'pixel (row {r}, col {c})', color='white', fontsize=10)
        ax.set_ylim(-0.2, 1.05)
        ax.set_ylabel('NDVI', color='white')
        ax.set_facecolor('#0a0a1a')
        ax.grid(True, ls='--', alpha=0.3)
        plt.setp(ax.get_xticklabels(), color='white')
        plt.setp(ax.get_yticklabels(), color='white')
    axes[0, 0].legend(facecolor='#1a1a2e', edgecolor='#5a7cff', labelcolor='white', fontsize=8)
    fig_ba.patch.set_facecolor('#0a0a1a')
    fig_ba.tight_layout()
    st.pyplot(fig_ba)

    # ---------- 3. VALIDITY MASK ----------
    st.markdown("### 3. Validity mask (kept vs discarded pixels)")
    st.caption(
        f"{int(valid_mask_full.sum()):,} pixels kept out of {H * W:,} "
        f"({100 * valid_mask_full.sum() / (H * W):.1f}%); the rest are no-data / over-clouded and excluded from training.")
    fig_mask, ax_mask = plt.subplots(figsize=(W / 110, H / 110), dpi=100)
    ax_mask.imshow(valid_mask_full, cmap='Greens', interpolation='nearest')
    ax_mask.axis('off')
    fig_mask.patch.set_facecolor('#0a0a1a')
    st.pyplot(fig_mask)


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
        look_back = col_p1.slider("Context Window (Look-back / acquisitions)", min_value=5, max_value=40, value=25,
                                   step=1)
        epochs = col_p2.slider("Training Epochs", min_value=10, max_value=200, value=50, step=10)

        st.markdown("---")
        st.markdown("### 2. Future Forecast Configuration")
        st.info(
            "🌦️ *Note:* The forecast is *autoregressive* (the network uses its own predictions). "
            "The model predicts natural phenology based on its memory of historical patterns.")

        days_to_predict = st.number_input("Acquisition steps to forecast", min_value=1, max_value=100, value=60)

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
                    def _init_(self, epoch_text, progress_bar, total_epochs, **kwargs):
                        super()._init_(**kwargs)
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
                            f"🔄 *Epoch {epoch_num}/{self.total_epochs}* | "
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

                # Date range for forecast: continue right after the last acquisition,
                # keeping the same ~3-week (20-day) display step as the historical series.
                last_timestamp = acquisition_dates(N_ACQ)[-1]
                future_dates = pd.date_range(
                    start=last_timestamp + pd.Timedelta(days=20),
                    periods=days_to_predict,
                    freq='20D'
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

        st.success(f"✅ Training completed in the background and model saved to {MODEL_PATH} successfully.")

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
        ax_pred.set_xlabel(f'Actual {SENSOR} NDVI (Scaled 0-1)', color='white')
        ax_pred.set_ylabel('Predicted NDVI (Scaled 0-1)', color='white')
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




# ==========================================
# TAB 3: SATELLITE IMAGERY (REAL SPATIAL MAPS)
# ==========================================
elif menu == "Spatial NDVI Visualization":
    st.title(f"Spatial Visualization of {SENSOR} NDVI Predictions")
    st.markdown(
        "Generate $H \\times W$ spatial maps in real-time using the trained LSTM model applied to all valid pixels.")

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

        # Slider to select the acquisition index (0..91) used as the prediction target.
        # We need look_back previous acquisitions as context to predict the next one,
        # so the slider ranges from look_back to 91.
        selected_week = col_s1.slider(f"Select {SENSOR} Acquisition Index (target)", min_value=look_back_config,
                                      max_value=91, value=look_back_config, step=1)

        # Map type
        map_type = col_s2.selectbox("Visual Map Type:",
                                    [f"Raw {SENSOR} Data (Actual)", "Model Prediction (1-Step Ahead)",
                                     "Spatial Error Map"])

        col_map_btn1, col_map_btn2 = st.columns([3, 1])
        with col_map_btn1:
            load_map_btn = st.button("🖼️ Generate Spatial Visualization", type="primary", use_container_width=True)

        if load_map_btn:
            with st.spinner(
                    f"Executing prediction on {X_train_full.shape[0] + X_val_full.shape[0]:,} pixels for {SENSOR} acquisition {selected_week}..."):
                # --- A) PREPARE CONTEXT FOR ALL PIXELS ---
                # Look at all valid pixels (concatenate)
                X_all_full = np.concatenate([X_train_full, X_val_full], axis=0)  # Shape: (TotalPix, 92, 1)

                # Extract spatial context H x W x look_back x 1 for selected_week
                all_X_context = X_all_full[
                    :, selected_week - look_back_config: selected_week, :]  # Shape: (TotalPix, LB, 1)

                # Real target for comparison (week selected_week)
                target_real = X_all_full[:, selected_week, 0]  # Shape (TotalPix,)

                # --- B) EXECUTE SATELLITE-SCALE PREDICTION ---
                preds_raw = model.predict(all_X_context, verbose=0)  # Shape: (TotalPix, 1)

                # --- C) RECONSTRUCT SPATIAL MAPS ---
                # Fill with NaN so that masked-out (no-data) pixels are NOT confused
                # with a valid NDVI value of 0. Only valid pixels receive a number;
                # the rest stay NaN and are drawn as transparent/background.
                reconstructed_preds_map = np.full((H, W), np.nan)
                reconstructed_actual_map = np.full((H, W), np.nan)
                reconstructed_error_map = np.full((H, W), np.nan)

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
            st.markdown(f"### 🖼️ Map: {map_type} ({SENSOR} acquisition {map_results['week']})")

            fig_sat_final, ax_sat_final = plt.subplots(figsize=(W / 80, H / 80), dpi=100)  # Maintain aspect ratio

            # Configure colormap and limits depending on map type
            if "Error" in map_type:
                data_plot = map_results['error_map']
                norm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0, vmax=0.3)  # Errors centered on 0
                cmap = plt.get_cmap('coolwarm_r').copy()  # Red=Prediction < Actual, Blue=Prediction > Actual
                label_cbar = 'NDVI Difference (Actual - Predicted)'
            else:
                norm = mcolors.Normalize(vmin=0, vmax=1)  # Scaled NDVI in [0,1]
                cmap = plt.get_cmap('RdYlGn').copy()
                if "Actual" in map_type:
                    data_plot = map_results['actual_map']
                    label_cbar = f'Raw {SENSOR} NDVI (Scaled 0-1)'
                else:
                    data_plot = map_results['preds_map']
                    label_cbar = 'LSTM Model NDVI (Scaled 0-1)'

            # No-data (NaN) pixels render in the dark background colour, so the colour
            # scale is reserved strictly for the [0,1] valid-NDVI range.
            cmap.set_bad('#0a0a1a')
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
                f"💡 *Note:* The {H} × {W} pixel grid (~{H * 10 / 1000:.1f} × {W * 10 / 1000:.1f} km at 10 m "
                f"{SENSOR} resolution) is a clip from tile *{TILE}*, over the {REGION} agricultural area "
                f"north-west of Zurich.")

            # Laegeren / Harth study area within Sentinel-2 tile T32TLT
            laegeren = [47.48, 8.40]
            m = folium.Map(location=laegeren, zoom_start=11, tiles=None)
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri', name='Satellite View', overlay=False, control=True).add_to(m)

            # Approximate footprint of the ~5 x 5 km ROI clip (centred on the Laegeren ridge).
            # 0.045 deg lat ~ 5 km; 0.066 deg lon ~ 5 km at this latitude.
            half_lat = (H * 10 / 1000) / 222.0   # km -> degrees latitude (1 deg ~ 111 km)
            half_lon = (W * 10 / 1000) / (222.0 * np.cos(np.radians(laegeren[0])))
            folium.Rectangle(
                bounds=[[laegeren[0] - half_lat, laegeren[1] - half_lon],
                        [laegeren[0] + half_lat, laegeren[1] + half_lon]],
                color='#5a7cff',
                fill=True,
                fill_opacity=0.25,
                tooltip=f"ROI clip in tile {TILE} ({REGION})",
            ).add_to(m)
            folium.Marker(laegeren, tooltip=f"Tile {TILE} - Laegeren / Harth").add_to(m)
            st_folium(m, width=500, height=450)


# ==========================================
# TAB 4: TECHNICAL DOCUMENTATION
# ==========================================
elif menu == "Technical Documentation":
    st.title("Technical Documentation")

    st.markdown(f"""
    ### Spatio-Temporal Forecasting Architecture

    The system uses a stacked LSTM network trained on univariate *{SENSOR}* NDVI time series
    (cloud-masked, gap-filled, Savitzky-Golay smoothed, and globally min-max scaled to [0,1]).
    The model predicts the next acquisition from its memory of a context window (look-back) of
    25 acquisitions, roughly one and a half years of phenological history given the irregular
    ~3-week effective sampling.

    ### Model Serialization
    The network is saved in the native TensorFlow Keras format (.keras) at agro_lstm_real_spatial.keras,
    so it can be loaded for inference (e.g. in the Spatial Visualization tab) without retraining.

    ### Model Specifications

    | Component | Value |
    |-----------|-------------|
    | Architecture | Stacked LSTM (64 units -> 32 units) -> Dense(16, LeakyReLU) -> Dense(1, sigmoid) |
    | Regularization | L2 (1e-4) on LSTM weights + Dropout 0.2 after each LSTM |
    | Training Epochs | *50* (default) |
    | Optimizer | Adam (Huber loss, ReduceLROnPlateau) |
    | Context Window (LB) | 25 acquisitions |

    ### Data Source

    | Source | Format | Valid Pixels | Acquisitions | Study Area |
    |--------|--------|--------------|--------------|------------|
    | {SENSOR} NDVI (10 m) | Flat .npz archive | ~232,000 pixels | {N_ACQ} over ~{SPAN_YEARS} yr | Tile {TILE} ({REGION}) |

    ### Satellite-Scale Spatial Reconstruction
    In the Spatial Visualization tab, the LSTM model is loaded and applied to every valid pixel
    from the .npz archive for a selected acquisition. The predictions are mapped through the
    (N,2) [row, col] index back into an $H \\times W$ matrix of $466 \\times 513$, reconstructing
    the spatial NDVI map of the region.
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
