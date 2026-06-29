"""
Per-pixel pure-LSTM NDVI forecasting (no clustering).

Exactly the protocol requested:
    - each pixel is a vector of 92 NDVI values
    - the model learns from the first 80 values and predicts the next 12
    - the error is computed between the predicted 12 and the actual last 12
      of the same pixel vector

Methodology
-----------
A single shared LSTM (this is the standard per-pixel approach: one model
trained across many pixels, then applied to each pixel individually) maps the
first 80 acquisitions -> the next 12 in one direct multi-output step. The model
is trained on the training pixels and evaluated on the held-out validation
pixels, so the reported per-pixel error reflects genuine generalization and
there is no train/test leakage (a validation pixel's last 12 values are never
seen during training). No K-means, no clustering, no spatial/CNN component.

Outputs (saved to ./Pixel_LSTM_Results/):
    - example_pixels.png      predicted vs actual last 12, for a few pixels
    - error_distribution.png  histogram of per-pixel RMSE
    - error_map.png           per-pixel RMSE folded onto the 466x513 grid
    - training_loss.png
    - metrics.txt             overall + per-pixel error summary

Run (in the environment that has tensorflow):
    python pixel_lstm_forecast.py
"""

import os
import ast
import random

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
CONFIG = dict(
    npz_path     = "ndvi_lstm_processed.npz",
    out_dir      = "Pixel_LSTM_Results",

    n_acq        = 92,    # values per pixel
    train_steps  = 80,    # learn from the first 80 ...
    test_steps   = 12,    # ... and predict the next 12  (80 + 12 = 92)

    # train the shared model on this many randomly drawn training pixels
    # (each pixel = one 80 -> 12 example); keeps training fast. Set to None
    # to use every available training pixel.
    train_sample = 50000,

    # full-grid predicted-vs-actual images: which held-out acquisitions to
    # render (each must be in [train_steps, n_acq-1], i.e. one of the last 12).
    # The first is saved as image_pred_vs_actual.png; the rest as
    # image_pred_vs_actual_<acq>.png. 82 / 86 / 90 span early / mid / late
    # of the 12-step forecast horizon.
    image_acqs   = [86, 82, 90],

    lstm_units_1 = 100,
    lstm_units_2 = 50,
    activation   = "tanh",   # standard, stable recurrent activation
    epochs       = 50,
    batch_size   = 128,

    seed         = 42,
)


# --------------------------------------------------------------------------- #
def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def load_data(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"'{npz_path}' not found. Run ndvi_preprocess.py first.")
    d = np.load(npz_path)
    X_train = d["X_train"][..., 0]   # (Ntr, 92) training pixels
    X_val   = d["X_val"][..., 0]     # (Nval, 92) validation pixels
    mask    = d["valid_mask"]        # (H, W)
    val_pix = d["val_pixel_index"] if "val_pixel_index" in d else None
    tr_pix  = d["train_pixel_index"] if "train_pixel_index" in d else None
    scaler  = ast.literal_eval(str(d["scaler"][0]))
    lo, hi  = float(scaler.get("lo", 0.0)), float(scaler.get("hi", 1.0))
    return X_train, X_val, mask, tr_pix, val_pix, lo, hi


def to_ndvi(scaled, lo, hi):
    return scaled * (hi - lo) + lo


def build_model(cfg):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    act = cfg["activation"]
    model = Sequential([
        LSTM(cfg["lstm_units_1"], activation=act, return_sequences=True,
             input_shape=(cfg["train_steps"], 1)),
        LSTM(cfg["lstm_units_2"], activation=act),
        Dense(cfg["test_steps"]),   # predict all 12 future steps at once
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


# --------------------------------------------------------------------------- #
def plot_examples(X_val_ndvi, preds_ndvi, rmse, ts, hz, path, n=6):
    rng = np.random.default_rng(0)
    # pick a spread of pixels: best, worst, and random middle ones
    order = np.argsort(rmse)
    picks = [order[0], order[len(order) // 2], order[-1]]
    picks += list(rng.choice(len(rmse), size=max(0, n - 3), replace=False))
    picks = picks[:n]

    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, picks):
        full = X_val_ndvi[p]
        x_ctx = np.arange(ts - 12, ts + hz)
        ax.plot(x_ctx, full[ts - 12:], "-o", color="#1f77b4", ms=3, label="Actual")
        ax.plot(np.arange(ts, ts + hz), preds_ndvi[p], "--o", color="#ff7f0e",
                ms=3, label="Predicted")
        ax.axvline(ts - 0.5, color="gray", ls=":", lw=1)
        ax.set_title(f"pixel #{p}  (RMSE={rmse[p]:.3f})", fontsize=10)
        ax.set_xlabel("Acquisition index")
        ax.set_ylabel("NDVI")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for j in range(len(picks), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Per-pixel forecast: predicted vs actual last {hz} acquisitions",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_hist(rmse, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rmse, bins=60, color="#4c72b0", edgecolor="white")
    ax.axvline(rmse.mean(), color="red", ls="--",
               label=f"mean = {rmse.mean():.3f}")
    ax.axvline(np.median(rmse), color="green", ls="--",
               label=f"median = {np.median(rmse):.3f}")
    ax.set_xlabel("Per-pixel RMSE (NDVI units)")
    ax.set_ylabel("Number of pixels")
    ax.set_title("Distribution of per-pixel forecast error (held-out pixels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_error_map(rmse, val_pix, mask, path):
    if val_pix is None:
        print("  (skipping error map: validation pixel index not in npz)")
        return
    H, W = mask.shape
    emap = np.full((H, W), np.nan)
    emap[val_pix[:, 0], val_pix[:, 1]] = rmse
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(emap, cmap=cmap, vmin=0, vmax=np.nanpercentile(emap, 98),
                   interpolation="nearest")
    ax.set_title("Per-pixel forecast RMSE over the held-out validation pixels")
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="RMSE (NDVI units)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_image_comparison(actual_map, pred_map, acq, path):
    """Side-by-side actual / predicted / difference NDVI maps (one acquisition)."""
    import matplotlib.colors as mcolors
    valid = np.isfinite(actual_map)
    vmin = float(np.percentile(actual_map[valid], 2))
    vmax = float(np.percentile(actual_map[valid], 98))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    ndvi_cmap = plt.get_cmap("RdYlGn").copy()
    ndvi_cmap.set_bad("#dddddd")

    for ax, data, title in [
            (axes[0], actual_map, f"Actual NDVI (acq {acq})"),
            (axes[1], pred_map,   f"Predicted NDVI (acq {acq})")]:
        im = ax.imshow(data, cmap=ndvi_cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    diff = actual_map - pred_map
    dcmap = plt.get_cmap("coolwarm").copy()
    dcmap.set_bad("#dddddd")
    dnorm = mcolors.TwoSlopeNorm(vmin=-0.3, vcenter=0.0, vmax=0.3)
    im = axes[2].imshow(diff, cmap=dcmap, norm=dnorm, interpolation="nearest")
    axes[2].set_title("Difference (actual - predicted)")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Per-pixel pure-LSTM forecast vs. actual NDVI "
                 f"(acquisition {acq}, all valid pixels)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], color="#1f77b4", label="Training loss")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], color="#ff7f0e", label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("Model loss over time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(cfg=CONFIG):
    set_seeds(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    ts, hz = cfg["train_steps"], cfg["test_steps"]

    print("1. Loading preprocessed NDVI ...")
    X_train, X_val, mask, tr_pix, val_pix, lo, hi = load_data(cfg["npz_path"])
    print(f"   train pixels: {X_train.shape[0]:,}   val pixels: {X_val.shape[0]:,}"
          f"   (NDVI range {lo:.3f}..{hi:.3f})")

    # Build supervised examples: first `ts` values -> next `hz` values
    Xtr_in = X_train[:, :ts]            # (Ntr, 80)
    Ytr    = X_train[:, ts:ts + hz]     # (Ntr, 12)
    if cfg["train_sample"] and cfg["train_sample"] < Xtr_in.shape[0]:
        rng = np.random.default_rng(cfg["seed"])
        sel = rng.choice(Xtr_in.shape[0], size=cfg["train_sample"], replace=False)
        Xtr_in, Ytr = Xtr_in[sel], Ytr[sel]
    Xtr_in = Xtr_in[..., None].astype(np.float32)   # (n, 80, 1)
    Ytr = Ytr.astype(np.float32)
    print(f"2. Training shared LSTM on {Xtr_in.shape[0]:,} pixels "
          f"({ts} in -> {hz} out, {cfg['epochs']} epochs) ...")

    model = build_model(cfg)
    history = model.fit(Xtr_in, Ytr, epochs=cfg["epochs"],
                        batch_size=cfg["batch_size"], validation_split=0.1,
                        shuffle=True, verbose=2)

    print("3. Forecasting the last "
          f"{hz} acquisitions for every held-out validation pixel ...")
    Xva_in = X_val[:, :ts][..., None].astype(np.float32)   # (Nval, 80, 1)
    preds_scaled = model.predict(Xva_in, batch_size=512, verbose=0)  # (Nval, 12)
    actual_scaled = X_val[:, ts:ts + hz]                              # (Nval, 12)

    preds_ndvi  = to_ndvi(preds_scaled, lo, hi)
    actual_ndvi = to_ndvi(actual_scaled, lo, hi)
    X_val_ndvi  = to_ndvi(X_val, lo, hi)

    # per-pixel RMSE over the 12 forecast steps
    per_pixel_rmse = np.sqrt(np.mean((preds_ndvi - actual_ndvi) ** 2, axis=1))

    # overall metrics across all pixel-steps
    err = (preds_ndvi - actual_ndvi).ravel()
    rmse_all = float(np.sqrt(np.mean(err ** 2)))
    mae_all  = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actual_ndvi.ravel() - actual_ndvi.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"   overall RMSE = {rmse_all:.4f}   MAE = {mae_all:.4f}   R2 = {r2:.4f}")
    print(f"   per-pixel RMSE: mean = {per_pixel_rmse.mean():.4f}   "
          f"median = {np.median(per_pixel_rmse):.4f}")

    print("4. Writing figures and metrics ...")
    od = cfg["out_dir"]
    plot_examples(X_val_ndvi, preds_ndvi, per_pixel_rmse, ts, hz,
                  os.path.join(od, "example_pixels.png"))
    plot_error_hist(per_pixel_rmse, os.path.join(od, "error_distribution.png"))
    plot_error_map(per_pixel_rmse, val_pix, mask, os.path.join(od, "error_map.png"))
    plot_loss(history, os.path.join(od, "training_loss.png"))

    # Full-grid predicted-vs-actual NDVI images for the selected held-out
    # acquisitions, over ALL valid pixels (train + val) so the maps are dense.
    if tr_pix is not None and val_pix is not None:
        print("5. Rendering full-grid predicted-vs-actual images ...")
        X_all = np.concatenate([X_train, X_val], axis=0)
        pix_all = np.concatenate([tr_pix, val_pix], axis=0)
        preds_all = model.predict(X_all[:, :ts][..., None],
                                  batch_size=512, verbose=0)
        H, W = mask.shape
        for i, acq in enumerate(cfg["image_acqs"]):
            pos = acq - ts                  # column within the 12-step output
            if not (0 <= pos < hz):
                print(f"   (skipping acq {acq}: outside the {hz}-step horizon)")
                continue
            actual_map = np.full((H, W), np.nan)
            pred_map = np.full((H, W), np.nan)
            actual_map[pix_all[:, 0], pix_all[:, 1]] = to_ndvi(X_all[:, acq], lo, hi)
            pred_map[pix_all[:, 0], pix_all[:, 1]] = to_ndvi(preds_all[:, pos], lo, hi)
            fname = ("image_pred_vs_actual.png" if i == 0
                     else f"image_pred_vs_actual_{acq}.png")
            plot_image_comparison(actual_map, pred_map, acq,
                                  os.path.join(od, fname))
            print(f"   acq {acq} -> {fname}")

    with open(os.path.join(od, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write("Per-pixel pure-LSTM NDVI forecast (first 80 -> next 12)\n")
        f.write(f"evaluated on {X_val.shape[0]:,} held-out validation pixels\n\n")
        f.write(f"overall RMSE        : {rmse_all:.4f} (NDVI units)\n")
        f.write(f"overall MAE         : {mae_all:.4f}\n")
        f.write(f"overall R2          : {r2:.4f}\n")
        f.write(f"per-pixel RMSE mean : {per_pixel_rmse.mean():.4f}\n")
        f.write(f"per-pixel RMSE med  : {np.median(per_pixel_rmse):.4f}\n")
        f.write(f"per-pixel RMSE p90  : {np.percentile(per_pixel_rmse, 90):.4f}\n")

    print(f"Done. Results saved to ./{od}/")


if __name__ == "__main__":
    main()
