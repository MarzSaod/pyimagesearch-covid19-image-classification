"""
report_utils.py
Helper code for the chest X ray project.

It does three jobs:
1. Image preprocessing helpers that training and the reports both use, so they always match.
2. A neat one page report for every trained model.
3. Extra analysis that needs no retraining: example images, Grad-CAM heatmaps,
   a decision threshold test, a model comparison table, and an image enhancement preview.

Everything reads the files that train_covid19.py saves into outputs/<model>_<run name>/.
"""
import hashlib
import json
import os
import textwrap
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

TEAL = "#065A82"
LIGHT_TEAL = "#7FB3CB"
ORANGE = "#E8871E"
GRAY = "#9AA5AD"
RED = "#B23A3A"
GREEN = "#2E7D32"
INK = "#1A1A1A"
RUN_COLORS = [TEAL, ORANGE, GRAY, "#1C7293", "#B8860B", "#5B6B76"]

IMG_SIZE = (224, 224)

# Last convolution layer of each model, used by Grad-CAM
LAST_CONV = {
    "vgg16": "block5_conv3",
    "vgg19": "block5_conv4",
    "inceptionv3": "mixed10",
    "resnet50": "conv5_block3_out",
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def apply_clahe(x, clip=2.0, tile=8):
    """Boost local contrast of an X ray. x is a float array (H, W, 3) with values 0 to 255."""
    gray = cv2.cvtColor(np.clip(x, 0, 255).astype("uint8"), cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)
    return np.repeat(gray[..., None], 3, axis=2).astype("float32")


def get_preprocess(model_name, mode="rescale", clahe=False):
    """
    Returns the function that turns a raw image array (0 to 255) into model input.
    mode "rescale": divide by 255, which is what the project has used so far.
    mode "imagenet": the scaling each pretrained model was originally trained with.
    """
    if mode == "imagenet":
        from tensorflow.keras.applications import inception_v3, resnet50, vgg16, vgg19

        base = {
            "vgg16": vgg16.preprocess_input,
            "vgg19": vgg19.preprocess_input,
            "inceptionv3": inception_v3.preprocess_input,
            "resnet50": resnet50.preprocess_input,
        }[model_name]
    else:
        def base(x):
            return x / 255.0

    if clahe:
        return lambda x: base(apply_clahe(x).copy())
    return lambda x: base(x.copy())


# ---------------------------------------------------------------------------
# Loading saved runs
# ---------------------------------------------------------------------------
def load_run(run_dir):
    d = Path(run_dir)
    with open(d / "config.json") as f:
        cfg = json.load(f)
    with open(d / "history.json") as f:
        hist = json.load(f)
    with open(d / "val_files.json") as f:
        files = json.load(f)
    with open(d / "class_labels.json") as f:
        classes = json.load(f)
    return {
        "dir": d,
        "config": cfg,
        "history": hist,
        "files": files,
        "classes": classes,
        "probs": np.load(d / "val_probs.npy"),
        "labels": np.load(d / "val_labels.npy"),
    }


def pretty_name(cfg):
    return f"{cfg['model'].upper()} ({cfg['run_name']})"


def find_runs(outdir):
    """List every finished run folder inside outdir."""
    base = Path(outdir)
    return sorted(str(p.parent) for p in base.glob("*/val_probs.npy"))


def target_index(classes, target_class="pneumonia"):
    return classes.index(target_class) if target_class in classes else len(classes) - 1


def run_metrics(run, target_class="pneumonia"):
    classes, y = run["classes"], run["labels"]
    pred = run["probs"].argmax(axis=1)
    t = target_index(classes, target_class)
    n = len(classes)
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    rep = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
    other = [i for i in range(n) if i != t]
    # where do the missed target images go
    missed_to = {classes[i]: cm[t, i] / max(cm[t].sum(), 1) for i in other}
    return {"cm": cm, "report": rep, "pred": pred, "t": t, "missed_to": missed_to}


# ---------------------------------------------------------------------------
# One page report for a single model
# ---------------------------------------------------------------------------
def make_model_report(run_dir, out_png=None, show=False, target_class="pneumonia"):
    run = load_run(run_dir)
    classes = run["classes"]
    m = run_metrics(run, target_class)
    cm, rep, t = m["cm"], m["report"], m["t"]
    acc = rep["accuracy"]
    cfg, hist = run["config"], run["history"]
    tname = classes[t]

    fig = plt.figure(figsize=(15, 8.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
    fig.suptitle(f"{pretty_name(cfg)}: {acc:.1%} accuracy", fontsize=19, fontweight="bold", color=TEAL, y=0.98)

    # confusion matrix
    ax = fig.add_subplot(gs[0, 0])
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("teal", ["#FFFFFF", TEAL])
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else INK, fontsize=12)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("What the model said")
    ax.set_ylabel("What the image really is")
    ax.set_title("Confusion matrix (percent of each true class)", fontsize=12)

    # per class scores
    ax = fig.add_subplot(gs[0, 1])
    metrics = [("precision", "Precision", GRAY), ("recall", "Recall", TEAL), ("f1-score", "F1 score", LIGHT_TEAL)]
    width = 0.26
    xs = np.arange(len(classes))
    for k, (key, label, color) in enumerate(metrics):
        vals = [rep[c][key] for c in classes]
        bar_colors = [ORANGE if (key == "recall" and i == t) else color for i in range(len(classes))]
        bars = ax.bar(xs + (k - 1) * width, vals, width, color=bar_colors, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.25)
    ax.set_title(f"Scores per class (orange bar: {tname} recall)", fontsize=12)
    ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # accuracy curves
    epochs = np.arange(1, len(hist["accuracy"]) + 1)
    best_ep = int(np.argmax(hist["val_accuracy"])) + 1
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(epochs, hist["accuracy"], color=GRAY, label="Training")
    ax.plot(epochs, hist["val_accuracy"], color=TEAL, label="Validation")
    ax.scatter([best_ep], [max(hist["val_accuracy"])], color=ORANGE, zorder=5, label=f"Best epoch {best_ep}")
    ax.set_title("Accuracy while training", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # plain language summary
    ax = fig.add_subplot(gs[1, 0:2])
    ax.axis("off")
    missed_txt = ", ".join(f"{v:.0%} as {k}" for k, v in m["missed_to"].items())
    lines = [
        ("What this model did", True),
        (f"It labeled {acc:.1%} of the {int(cm.sum())} validation images correctly.", False),
        (f"It caught {rep[tname]['recall']:.0%} of the real {tname} images. The misses were labeled: {missed_txt}.", False),
        (f"When it said {tname}, it was right {rep[tname]['precision']:.0%} of the time.", False),
        (f"Average score across classes (macro F1): {rep['macro avg']['f1-score']:.0%}.", False),
        (f"Best epoch was {best_ep} of {len(epochs)}. Training time: {cfg.get('train_minutes', 0):.0f} minutes.", False),
    ]
    y0 = 0.98
    for text, bold in lines:
        wrapped = textwrap.fill(text, width=78)
        ax.text(0.0, y0, wrapped, fontsize=13 if not bold else 15, fontweight="bold" if bold else "normal",
                color=TEAL if bold else INK, va="top")
        y0 -= 0.075 * (wrapped.count("\n") + 1) + 0.075

    # loss curves
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(epochs, hist["loss"], color=GRAY, label="Training")
    ax.plot(epochs, hist["val_loss"], color=TEAL, label="Validation")
    ax.set_title("Loss while training (lower is better)", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out_png = out_png or str(Path(run_dir) / "report.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Compare several runs side by side
# ---------------------------------------------------------------------------
def compare_runs(run_dirs, out_png=None, show=True, target_class="pneumonia"):
    rows, names, recalls = [], [], []
    classes = None
    for rd in run_dirs:
        run = load_run(rd)
        classes = run["classes"]
        m = run_metrics(run, target_class)
        rep, t = m["report"], m["t"]
        tname = classes[t]
        row = {
            "Run": pretty_name(run["config"]),
            "Accuracy": rep["accuracy"],
            "Macro F1": rep["macro avg"]["f1-score"],
        }
        for c in classes:
            row[f"{c.capitalize()} recall"] = rep[c]["recall"]
        row[f"{tname.capitalize()} precision"] = rep[tname]["precision"]
        for k, v in m["missed_to"].items():
            row[f"{tname.capitalize()} called {k}"] = v
        # held out test results (only present when a run used --test_frac)
        tm_path = Path(rd) / "test_metrics.json"
        if tm_path.exists():
            with open(tm_path) as f:
                tm = json.load(f)
            row["Test accuracy"] = tm["accuracy"]
            row[f"Test {tname} recall"] = tm["recall"][tname]
        rows.append(row)
        names.append(pretty_name(run["config"]))
        recalls.append([rep[c]["recall"] for c in classes])
    df = pd.DataFrame(rows).set_index("Run")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    w = 0.8 / max(len(names), 1)
    for i, nm in enumerate(names):
        vals = [df.loc[nm, "Accuracy"], df.loc[nm, "Macro F1"]]
        bars = ax.bar(np.arange(2) + i * w, vals, w, color=RUN_COLORS[i % len(RUN_COLORS)], label=nm)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(2) + w * (len(names) - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Macro F1"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Overall scores", fontsize=13)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    for i, nm in enumerate(names):
        bars = ax.bar(np.arange(len(classes)) + i * w, recalls[i], w, color=RUN_COLORS[i % len(RUN_COLORS)], label=nm)
        for b, v in zip(bars, recalls[i]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(np.arange(len(classes)) + w * (len(names) - 1) / 2)
    ax.set_xticklabels([f"{c} recall" for c in classes])
    ax.set_ylim(0, 1.1)
    ax.set_title("How many real images of each type were caught", fontsize=13)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    out_png = out_png or "comparison.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return df


# ---------------------------------------------------------------------------
# Example images: caught versus missed
# ---------------------------------------------------------------------------
def pick_examples(run, target_class="pneumonia", n=4, seed=0):
    y, pred = run["labels"], run["probs"].argmax(axis=1)
    t = target_index(run["classes"], target_class)
    rng = np.random.RandomState(seed)
    caught = np.where((y == t) & (pred == t))[0]
    missed = np.where((y == t) & (pred != t))[0]
    caught = rng.choice(caught, min(n, len(caught)), replace=False) if len(caught) else caught
    missed = rng.choice(missed, min(n, len(missed)), replace=False) if len(missed) else missed
    return caught, missed


def _load_rgb(path):
    from tensorflow.keras.utils import img_to_array, load_img

    return img_to_array(load_img(path, target_size=IMG_SIZE, interpolation="nearest"))


def _grid_figure(n, headers):
    """Two rows of pictures. headers is a list of (text, color) for the two rows."""
    fig, axes = plt.subplots(2, n, figsize=(3.1 * n, 6.8))
    axes = np.atleast_2d(axes)
    fig.subplots_adjust(top=0.84, bottom=0.03, left=0.03, right=0.97, hspace=0.38, wspace=0.12)
    for row, (text, color) in enumerate(headers):
        y = axes[row, 0].get_position().y1 + 0.055
        fig.text(0.03, y, text, fontsize=14, fontweight="bold", color=color)
        for ax in axes[row]:
            ax.axis("off")
    return fig, axes


def show_predictions(run_dir, target_class="pneumonia", n=4, seed=0, out_png=None, show=True):
    run = load_run(run_dir)
    classes, probs = run["classes"], run["probs"]
    caught, missed = pick_examples(run, target_class, n, seed)
    fig, axes = _grid_figure(n, [(f"Real {target_class} images the model CAUGHT", GREEN),
                                 (f"Real {target_class} images the model MISSED", RED)])
    for row, (idxs, color) in enumerate([(caught, GREEN), (missed, RED)]):
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            ax.imshow(_load_rgb(run["files"][i]).astype("uint8"))
            p = probs[i].argmax()
            ax.set_title(f"Said: {classes[p]} ({probs[i, p]:.0%})", fontsize=11, color=color)
    fig.suptitle(f"{pretty_name(run['config'])}: good and bad predictions side by side", fontsize=16,
                 fontweight="bold", color=TEAL, y=0.98)
    out_png = out_png or str(Path(run_dir) / "examples.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Grad-CAM heatmaps
# ---------------------------------------------------------------------------
def _gradcam(model, x, layer_name, class_idx):
    import tensorflow as tf

    grad_model = tf.keras.models.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model([tf.convert_to_tensor(x)], training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)
    cam = tf.nn.relu(cam)[0].numpy()
    return cam / (cam.max() + 1e-8)


def gradcam_grid(run_dir, target_class="pneumonia", n=4, seed=0, out_png=None, show=True, model_file="best_model.h5"):
    from tensorflow.keras.models import load_model

    run = load_run(run_dir)
    cfg, classes, probs = run["config"], run["classes"], run["probs"]
    model = load_model(Path(run_dir) / model_file, compile=False)
    layer = LAST_CONV.get(cfg["model"])
    if layer is None or layer not in [l.name for l in model.layers]:
        convs = [l.name for l in model.layers if len(l.output.shape) == 4 and "conv" in l.name]
        layer = convs[-1]
    pre = get_preprocess(cfg["model"], cfg.get("preprocess", "rescale"), cfg.get("clahe", False))
    caught, missed = pick_examples(run, target_class, n, seed)

    fig, axes = _grid_figure(n, [("CAUGHT: where the model looked", GREEN),
                                 ("MISSED: where the model looked", RED)])
    for row, (idxs, color) in enumerate([(caught, GREEN), (missed, RED)]):
        for col, i in enumerate(idxs):
            ax = axes[row, col]
            raw = _load_rgb(run["files"][i])
            x = pre(raw)[None, ...]
            p = int(probs[i].argmax())
            cam = _gradcam(model, x, layer, p)
            cam = cv2.resize(cam, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            overlay = (0.55 * raw + 0.45 * heat).astype("uint8")
            ax.imshow(overlay)
            ax.set_title(f"Said: {classes[p]} ({probs[i, p]:.0%})", fontsize=11, color=color)
    fig.suptitle(f"{pretty_name(cfg)}: Grad-CAM heatmaps (red means the model paid more attention)",
                 fontsize=15, fontweight="bold", color=TEAL, y=0.98)
    out_png = out_png or str(Path(run_dir) / "gradcam.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Decision threshold test (no retraining)
# ---------------------------------------------------------------------------
def threshold_analysis(run_dir, target_class="pneumonia", seed=42, out_png=None, show=True):
    """
    Adds a bias to the target class score so the model says it more easily.
    The bias is picked on one half of the validation images and judged on the other half.
    """
    run = load_run(run_dir)
    classes, y, probs = run["classes"], run["labels"], run["probs"]
    t = target_index(classes, target_class)
    logp = np.log(probs + 1e-9)
    onehot = np.zeros(len(classes))
    onehot[t] = 1.0
    biases = np.round(np.arange(0, 4.01, 0.1), 2)

    idx = np.arange(len(y))
    tune_idx, hold_idx = train_test_split(idx, test_size=0.5, stratify=y, random_state=seed)

    def predict(ids, b):
        return (logp[ids] + b * onehot).argmax(axis=1)

    def scores(ids, b):
        pr = predict(ids, b)
        rep = classification_report(y[ids], pr, target_names=classes, output_dict=True, zero_division=0)
        return rep

    tune_f1 = [f1_score(y[tune_idx], predict(tune_idx, b), average="macro") for b in biases]
    best_b = float(biases[int(np.argmax(tune_f1))])

    curves = {"acc": [], "target_recall": [], "target_precision": []}
    for b in biases:
        rep = scores(tune_idx, b)
        curves["acc"].append(rep["accuracy"])
        curves["target_recall"].append(rep[classes[t]]["recall"])
        curves["target_precision"].append(rep[classes[t]]["precision"])

    before, after = scores(hold_idx, 0.0), scores(hold_idx, best_b)
    table = pd.DataFrame({
        "Before (bias 0)": {
            "Accuracy": before["accuracy"],
            f"{classes[t].capitalize()} recall": before[classes[t]]["recall"],
            f"{classes[t].capitalize()} precision": before[classes[t]]["precision"],
            "Macro F1": before["macro avg"]["f1-score"],
        },
        f"After (bias {best_b})": {
            "Accuracy": after["accuracy"],
            f"{classes[t].capitalize()} recall": after[classes[t]]["recall"],
            f"{classes[t].capitalize()} precision": after[classes[t]]["precision"],
            "Macro F1": after["macro avg"]["f1-score"],
        },
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(biases, curves["target_recall"], color=ORANGE, lw=2.5, label=f"{classes[t]} recall")
    ax.plot(biases, curves["target_precision"], color=GRAY, lw=2, label=f"{classes[t]} precision")
    ax.plot(biases, curves["acc"], color=TEAL, lw=2.5, label="Overall accuracy")
    ax.axvline(best_b, color=INK, ls="--", lw=1)
    ax.text(best_b + 0.05, 0.42, f"chosen bias {best_b}", fontsize=10)
    ax.set_xlabel(f"Bias added to the {classes[t]} score (0 means no change)")
    ax.set_title("Tuning half of the validation images", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    ax = axes[1]
    labels = ["Accuracy", f"{classes[t].capitalize()} recall", f"{classes[t].capitalize()} precision"]
    b_vals = table["Before (bias 0)"].values[:3]
    a_vals = table[f"After (bias {best_b})"].values[:3]
    xs = np.arange(3)
    r1 = ax.bar(xs - 0.2, b_vals, 0.4, color=GRAY, label="Before")
    r2 = ax.bar(xs + 0.2, a_vals, 0.4, color=TEAL, label="After")
    for bars in (r1, r2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.0%}", ha="center", fontsize=10)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.set_title("Judged on the other half (images not used to pick the bias)", fontsize=13)
    ax.legend(frameon=False)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.suptitle(f"{pretty_name(run['config'])}: what if the model said {classes[t]} more easily?",
                 fontsize=15, fontweight="bold", color=TEAL)
    fig.tight_layout()
    out_png = out_png or str(Path(run_dir) / "threshold.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return {"best_bias": best_b, "table": table, "png": out_png}


# ---------------------------------------------------------------------------
# Image enhancement preview
# ---------------------------------------------------------------------------
def clahe_preview(dataset_dir, n_per_class=2, seed=0, out_png=None, show=True):
    root = Path(dataset_dir)
    classes = sorted(p.name for p in root.iterdir() if p.is_dir())
    rng = np.random.RandomState(seed)
    picks = []
    for c in classes:
        files = sorted(str(p) for p in (root / c).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        for f in rng.choice(files, n_per_class, replace=False):
            picks.append((c, f))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.8))
    for k, (c, f) in enumerate(picks):
        raw = _load_rgb(f)
        axes[0, k].imshow(raw.astype("uint8"))
        axes[1, k].imshow(apply_clahe(raw).astype("uint8"))
        axes[0, k].set_title(c, fontsize=11, color=TEAL, fontweight="bold")
        axes[0, k].axis("off")
        axes[1, k].axis("off")
    axes[0, 0].text(-0.05, 0.5, "Original", rotation=90, transform=axes[0, 0].transAxes, va="center", ha="right", fontsize=12)
    axes[1, 0].text(-0.05, 0.5, "Enhanced", rotation=90, transform=axes[1, 0].transAxes, va="center", ha="right", fontsize=12)
    fig.suptitle("Image enhancement preview (CLAHE contrast boost)", fontsize=15, fontweight="bold", color=TEAL)
    fig.tight_layout()
    out_png = out_png or "clahe_preview.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# Dataset listing and image loading (shared by training and the checks below)
# ---------------------------------------------------------------------------
IMG_EXT = (".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".ppm", ".tif", ".tiff")


def list_dataset(dataset_dir):
    """
    Lists images the same way Keras flow_from_directory does:
    class folders in sorted order, files in sorted order, only these image extensions.
    Returns class names, file paths, integer labels, and a count of skipped non image files.
    """
    classes = sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d)))
    files, labels, skipped = [], [], Counter()
    for ci, c in enumerate(classes):
        folder = os.path.join(dataset_dir, c)
        for root, _, fnames in sorted(os.walk(folder), key=lambda x: x[0]):
            for fn in sorted(fnames):
                if fn.lower().endswith(IMG_EXT):
                    files.append(os.path.join(root, fn))
                    labels.append(ci)
                else:
                    skipped[(c, os.path.splitext(fn)[1].lower() or "(no extension)")] += 1
    return classes, files, np.array(labels, dtype=int), skipped


def duplicate_name_count(files, labels):
    """How many extra copies exist: file names that appear more than once inside the same class folder tree."""
    counts = Counter((int(l), os.path.basename(f)) for f, l in zip(files, labels))
    return int(sum(v - 1 for v in counts.values() if v > 1))


def remove_nested_copies(dataset_dir, apply=True):
    """
    Fixes the case where a class folder contains a second copy of itself (for example all/covid/covid).
    A folder inside a class folder is removed only if every file in it also exists directly in the class folder,
    so nothing unique is ever deleted. With apply=False it only reports what it would remove.
    """
    import shutil

    removed, kept = [], []
    for c in sorted(d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))):
        top = os.path.join(dataset_dir, c)
        loose = {fn for fn in os.listdir(top) if os.path.isfile(os.path.join(top, fn))}
        for sub in sorted(d for d in os.listdir(top) if os.path.isdir(os.path.join(top, d))):
            sub_path = os.path.join(top, sub)
            names = [fn for _, _, fns in os.walk(sub_path) for fn in fns]
            if names and all(n in loose for n in names):
                removed.append((sub_path, len(names)))
                if apply:
                    shutil.rmtree(sub_path)
            else:
                kept.append((sub_path, len(names)))
    for path, n in removed:
        print(("Removed" if apply else "Would remove") + f" the copied folder {path} ({n} files)")
    for path, n in kept:
        print(f"Left alone {path} ({n} files), it holds files that are not in the class folder itself")
    if not removed and not kept:
        print("No copied folders found.")
    return removed, kept


def load_image_array(files, size=224, workers=4):
    """Loads every image once as a small uint8 array (about 150 KB each). Uses the same resize Keras uses."""
    X = np.empty((len(files), size, size, 3), dtype=np.uint8)

    def one(i):
        with Image.open(files[i]) as im:
            X[i] = np.asarray(im.convert("RGB").resize((size, size), Image.NEAREST), dtype=np.uint8)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(files))))
    return X


def file_fingerprint(files):
    return hashlib.md5("\n".join(files).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Check 1: does the model sort pneumonia by image type instead of by disease?
# ---------------------------------------------------------------------------
def source_check(run_dir, workers=4):
    """
    Compares simple facts about images the model got right and wrong: original size, brightness, dark borders.
    If caught pneumonia images look different from missed ones (and missed ones look like normal images),
    the model may be using image style instead of disease.
    """
    run = load_run(run_dir)
    names = run["classes"]
    pred = run["probs"].argmax(axis=1)

    def facts(path):
        with Image.open(path) as im:
            w, h = im.size
            g = np.asarray(im.convert("L").resize((224, 224)), dtype=float)
        return w, h, g.mean(), (g[:, :12] < 20).mean() + (g[:, -12:] < 20).mean()

    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(facts, run["files"]))
    df = pd.DataFrame(out, columns=["width", "height", "mean_brightness", "dark_border_share"])
    df["outcome"] = [f"{names[y]} -> {names[p]}" for y, p in zip(run["labels"], pred)]
    df["true"] = [names[y] for y in run["labels"]]
    df["size"] = df["width"].astype(str) + "x" + df["height"].astype(str)

    keep = [o for o in [f"{c} -> {c}" for c in names] + ["pneumonia -> normal"] if o in set(df["outcome"])]
    table = df[df["outcome"].isin(keep)].groupby("outcome")[
        ["width", "height", "mean_brightness", "dark_border_share"]].median().round(2)
    table["count"] = df[df["outcome"].isin(keep)].groupby("outcome").size()
    table = table.loc[keep]

    sizes = []
    for c in names:
        counts = df[df["true"] == c]["size"].value_counts()
        sizes.append({"class": c, "different image sizes": len(counts),
                      "most common size": counts.index[0], "share of images with it": counts.iloc[0] / counts.sum()})
    return table, pd.DataFrame(sizes).set_index("class"), df


# ---------------------------------------------------------------------------
# Check 2: are near copies of training images sitting in the validation set?
# ---------------------------------------------------------------------------
def _thumbs(files, size=32, workers=4):
    def one(p):
        with Image.open(p) as im:
            g = np.asarray(im.convert("L").resize((size, size), Image.BOX), dtype=np.float32)
        g = g.ravel() - g.mean()
        return g / (np.linalg.norm(g) + 1e-8)

    with ThreadPoolExecutor(workers) as ex:
        return np.stack(list(ex.map(one, files)))


def duplicate_check(dataset_dir, run_dir, size=32, thresholds=(0.99, 0.98, 0.95, 0.90),
                    n_show=4, out_png=None, show=True, workers=4):
    """
    For every validation image, finds the most similar training image (also trying a mirrored copy).
    Similarity is the correlation of tiny gray versions, where 1.00 means identical.
    Reports how many validation images have a near copy in training, and whether the model
    scores better on those, which would mean validation accuracy is inflated.
    """
    run = load_run(run_dir)
    names = run["classes"]
    classes, files, labels, _ = list_dataset(dataset_dir)
    idx_of = {f: i for i, f in enumerate(files)}
    missing = [f for f in run["files"] if f not in idx_of]
    if missing:
        raise ValueError(f"{len(missing)} validation files from the run were not found in {dataset_dir}")
    vidx = np.array([idx_of[f] for f in run["files"]])
    excluded = set(vidx)
    test_json = Path(run_dir) / "test_files.json"
    if test_json.exists():
        excluded |= {idx_of[f] for f in json.load(open(test_json)) if f in idx_of}
    tidx = np.array([i for i in range(len(files)) if i not in excluded])

    th = _thumbs(files, size, workers)
    V, T = th[vidx], th[tidx]
    Vf = V.reshape(-1, size, size)[:, :, ::-1].reshape(len(V), -1)
    sim = np.maximum(V @ T.T, Vf @ T.T)
    best, arg = sim.max(axis=1), sim.argmax(axis=1)

    pred_ok = run["probs"].argmax(axis=1) == run["labels"]
    rows = []
    for t in thresholds:
        has = best >= t
        row = {"similarity at least": t, "val images with a near copy in training": int(has.sum()),
               "share of validation": has.mean()}
        row["accuracy on those"] = pred_ok[has].mean() if has.any() else np.nan
        row["accuracy on the rest"] = pred_ok[~has].mean() if (~has).any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("similarity at least")

    by_class = pd.DataFrame({"class": [names[y] for y in run["labels"]], "best": best})
    per_class = by_class.groupby("class")["best"].agg(
        median="median", p90=lambda v: np.quantile(v, 0.9), share_above_098=lambda v: (v >= 0.98).mean())

    # picture: closest pairs, then pairs from the middle of the pack for comparison
    order = np.argsort(-best)
    picks = list(order[:n_show]) + list(order[len(order) // 2: len(order) // 2 + n_show])
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.8))
    for col, k in enumerate(picks):
        for row_i, (path, ttl) in enumerate([(files[vidx[k]], "validation"), (files[tidx[arg[k]]], "training")]):
            with Image.open(path) as im:
                axes[row_i, col].imshow(im.convert("L").resize((224, 224)), cmap="gray")
            axes[row_i, col].axis("off")
            if row_i == 0:
                axes[row_i, col].set_title(f"{'closest' if col < n_show else 'typical'}: {best[k]:.3f}\n"
                                           f"{names[labels[vidx[k]]]}", fontsize=10,
                                           color=RED if col < n_show else INK)
            else:
                axes[row_i, col].set_title(f"nearest {ttl} image\n{names[labels[tidx[arg[k]]]]}", fontsize=9, color=GRAY)
    fig.suptitle("Validation images and their most similar training image", fontsize=13,
                 fontweight="bold", color=TEAL)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.5)
    out_png = out_png or str(Path(run_dir) / "duplicates.png")
    fig.savefig(out_png, dpi=120, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return table, per_class, out_png


# ---------------------------------------------------------------------------
# Lung focus experiment (idea from Bassi and Attux 2022)
# Three versions of every X ray: full image, lungs only, outside the lungs only.
# ---------------------------------------------------------------------------
def _eval_set(run_dir):
    """The held out test set if the run has one, otherwise the validation set."""
    d = Path(run_dir)
    if (d / "test_files.json").exists():
        with open(d / "test_files.json") as f:
            files = json.load(f)
        return files, np.load(d / "test_labels.npy"), np.load(d / "test_probs.npy"), "test"
    with open(d / "val_files.json") as f:
        files = json.load(f)
    return files, np.load(d / "val_labels.npy"), np.load(d / "val_probs.npy"), "validation"


def _last_conv_name(model, cfg):
    layer = LAST_CONV.get(cfg["model"])
    if layer is None or layer not in [l.name for l in model.layers]:
        layer = [l.name for l in model.layers if len(l.output.shape) == 4 and "conv" in l.name][-1]
    return layer


def _stem_key(path, root):
    return os.path.splitext(os.path.relpath(path, root))[0]


def variant_summary(run_map, target_class="pneumonia", out_png=None, show=True):
    """
    run_map: ordered dict such as {"Full image": run_dir, "Lungs only": run_dir, "Outside the lungs only": run_dir}.
    Bar chart of accuracy and recall on the held out test set (or validation set), with a 95 percent range
    worked out from the number of images. The dashed line is what random guessing would score.
    """
    rows = []
    for label, rd in run_map.items():
        files, y, probs, setname = _eval_set(rd)
        with open(Path(rd) / "class_labels.json") as f:
            classes = json.load(f)
        pred = probs.argmax(axis=1)
        n = len(y)
        acc = float((pred == y).mean())
        t = target_index(classes, target_class)
        nt = int((y == t).sum())
        rec = float(((pred == t) & (y == t)).sum() / max(nt, 1))
        rows.append({"Version": label, "Images": n, "Set": setname, "Accuracy": acc,
                     "Accuracy range": 1.96 * np.sqrt(acc * (1 - acc) / n),
                     f"{classes[t].capitalize()} recall": rec,
                     "Recall range": 1.96 * np.sqrt(rec * (1 - rec) / max(nt, 1))})
    df = pd.DataFrame(rows).set_index("Version")
    rec_col = [c for c in df.columns if c.endswith("recall")][0]
    chance = 1.0 / len(classes)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, col, rng_col, title in [(axes[0], "Accuracy", "Accuracy range", "Accuracy (all classes together)"),
                                    (axes[1], rec_col, "Recall range", f"{classes[t].capitalize()} recall (real cases caught)")]:
        vals = df[col].values
        bars = ax.bar(range(len(df)), vals, yerr=df[rng_col].values, capsize=5,
                      color=[RUN_COLORS[i % len(RUN_COLORS)] for i in range(len(df))], edgecolor="none")
        for b, v, e in zip(bars, vals, df[rng_col].values):
            ax.text(b.get_x() + b.get_width() / 2, min(v + e + 0.02, 1.06), f"{v:.0%}", ha="center", fontsize=12,
                    fontweight="bold")
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([textwrap.fill(i, 14) for i in df.index], fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEAL, loc="left")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].axhline(chance, color=RED, ls="--", lw=1.3)
    axes[0].text(-0.45, chance + 0.015, "random guessing", color=RED, fontsize=10, ha="left")
    fig.suptitle(f"Same classifier, three versions of every X ray ({df['Set'].iloc[0]} images, black lines show a "
                 f"95 percent range)", fontsize=13.5, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    out_png = out_png or "lung_focus_summary.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return df


def gradcam_variants(run_map, per_class=1, seed=0, out_png=None, show=True):
    """
    Same X rays through every version. First column is the original image, then one Grad CAM picture per version
    (each model sees its own version of the image). The first entry of run_map must be the full image run.
    """
    from tensorflow.keras.models import load_model

    names = list(run_map)
    runs = {k: load_run(v) for k, v in run_map.items()}
    sets = {k: _eval_set(v) for k, v in run_map.items()}
    ref = names[0]
    ref_root = runs[ref]["config"]["dataset"]
    lookup = {}
    for k in names:
        root = runs[k]["config"]["dataset"]
        lookup[k] = {_stem_key(f, root): f for f in sets[k][0]}
    ref_files, ref_y = sets[ref][0], sets[ref][1]
    keys = [_stem_key(f, ref_root) for f in ref_files]
    for k in names[1:]:
        missing = [x for x in keys if x not in lookup[k]]
        if missing:
            raise ValueError(f"Run '{k}' does not use the same images as '{ref}' (for example {missing[0]}).")
    classes = runs[ref]["classes"]
    rng = np.random.RandomState(seed)
    chosen = []
    for c in range(len(classes)):
        idx = np.where(ref_y == c)[0]
        chosen += [int(i) for i in rng.choice(idx, min(per_class, len(idx)), replace=False)]

    cols = 1 + len(names)
    fig, axes = plt.subplots(len(chosen), cols, figsize=(3.0 * cols, 3.05 * len(chosen) + 0.8))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")
    for j, k in enumerate(["Original X ray"] + names):
        axes[0, j].set_title(k, fontsize=12.5, fontweight="bold", color=INK, pad=10)
    for r, i in enumerate(chosen):
        raw0 = _load_rgb(ref_files[i])
        axes[r, 0].imshow(raw0.astype("uint8"))
        axes[r, 0].text(-0.04, 0.5, f"Real: {classes[ref_y[i]]}", transform=axes[r, 0].transAxes, rotation=90,
                        va="center", ha="right", fontsize=12, fontweight="bold", color=TEAL)
    for j, k in enumerate(names, start=1):
        cfg = runs[k]["config"]
        model = load_model(Path(run_map[k]) / "best_model.h5", compile=False)
        layer = _last_conv_name(model, cfg)
        pre = get_preprocess(cfg["model"], cfg.get("preprocess", "rescale"), cfg.get("clahe", False))
        for r, i in enumerate(chosen):
            raw = _load_rgb(lookup[k][keys[i]])
            x = pre(raw)[None, ...]
            probs = model.predict(x, verbose=0)[0]
            p = int(probs.argmax())
            cam = cv2.resize(_gradcam(model, x, layer, p), IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            axes[r, j].imshow((0.55 * raw + 0.45 * heat).astype("uint8"))
            ok = p == int(ref_y[i])
            axes[r, j].set_title(f"Said: {classes[p]} ({probs[p]:.0%})", fontsize=11, color=GREEN if ok else RED)
        del model
    fig.suptitle("Where each version of the model looked (red means more attention)", fontsize=15,
                 fontweight="bold", color=TEAL, y=1.0)
    fig.tight_layout()
    out_png = out_png or "lung_focus_gradcam.png"
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return out_png


def gradcam_lung_share(run_dir, masks_npz, per_class=100, seed=0, out_png=None, show=True):
    """
    For a model trained on the full image: how much of its Grad CAM attention lands inside the lung mask,
    compared with how much of the picture the lungs fill. A ratio near 1 means the model pays no special
    attention to the lungs. Above 1 means it favors the lungs. Below 1 means it favors the outside.
    Note that the outside area includes the heart, which can matter medically, and that Grad CAM is coarse.
    """
    from tensorflow.keras.models import load_model

    run = load_run(run_dir)
    cfg, classes = run["config"], run["classes"]
    files, y, probs, setname = _eval_set(run_dir)
    z = np.load(masks_npz)
    index = {r: i for i, r in enumerate(z["rel"])}
    masks = z["mask"]
    model = load_model(Path(run_dir) / "best_model.h5", compile=False)
    layer = _last_conv_name(model, cfg)
    pre = get_preprocess(cfg["model"], cfg.get("preprocess", "rescale"), cfg.get("clahe", False))
    rng = np.random.RandomState(seed)
    rows = []
    for c, name in enumerate(classes):
        idx = np.where(y == c)[0]
        idx = np.array([i for i in idx if os.path.relpath(files[i], cfg["dataset"]) in index], dtype=int)
        if len(idx) == 0:
            continue
        idx = rng.choice(idx, min(per_class, len(idx)), replace=False)
        inside, area = [], []
        for i in idx:
            rel = os.path.relpath(files[i], cfg["dataset"])
            lung = masks[index[rel]] > 127
            x = pre(_load_rgb(files[i]))[None, ...]
            cam = cv2.resize(_gradcam(model, x, layer, int(probs[i].argmax())), IMG_SIZE, interpolation=cv2.INTER_LINEAR)
            inside.append(float((cam * lung).sum() / (cam.sum() + 1e-8)))
            area.append(float(lung.mean()))
        rows.append({"Class": name, "Images": len(idx), "Lungs fill": np.mean(area),
                     "Attention inside lungs": np.mean(inside), "Ratio": np.mean(inside) / np.mean(area)})
    if not rows:
        raise ValueError("None of the evaluation images have a lung mask. The masks were made for a different set of images.")
    df = pd.DataFrame(rows).set_index("Class")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    xs = np.arange(len(df))
    b1 = ax.bar(xs - 0.2, df["Lungs fill"], 0.4, color=GRAY, label="Share of the picture the lungs fill")
    b2 = ax.bar(xs + 0.2, df["Attention inside lungs"], 0.4, color=TEAL, label="Share of model attention inside the lungs")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.0%}", ha="center", fontsize=11)
    ax.set_xticks(xs)
    ax.set_xticklabels([c.capitalize() for c in df.index], fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(f"Does the full image model look at the lungs? ({setname} images)", fontsize=13, fontweight="bold",
                 color=TEAL, loc="left")
    fig.tight_layout()
    out_png = out_png or str(Path(run_dir) / "lung_share.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)
    return df
