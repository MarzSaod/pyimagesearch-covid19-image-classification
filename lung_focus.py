"""
lung_focus.py
Adds the idea from Bassi and Attux (2022), "Covid 19 detection using chest X rays: is lung segmentation
important for generalization?", Research on Biomedical Engineering 38(4), 1121 to 1139.

It uses the lung finding network that the paper authors published (GPL 3.0 licence, so we download it at run
time and never copy it into our own repository):
    https://github.com/PedroRASB/COVID-19-detection-with-lung-segmentation

For every X ray in the dataset it makes two new copies:
    lung     the image multiplied by the lung probability, so everything outside the lungs fades to black
    outside  the image multiplied by one minus the lung probability, so only the area outside the lungs stays
The file names and class folders stay the same, so train_covid19.py can train on each copy with no changes
and every run uses exactly the same train, validation and test images.

USAGE
    python lung_focus.py --dataset /content/covid_pneumonia_dataset_wrapped/all --outdir /content/lung_data
Quick test on a few images:
    python lung_focus.py --dataset DATASET --outdir /content/lung_test --limit 20

OUTPUT
    OUTDIR/lung/all/<class>/<file>.png
    OUTDIR/outside/all/<class>/<file>.png
    OUTDIR/masks.npz                 the lung probability of every image, as 0 to 255
    OUTDIR/lung_focus_preview.png    one example per class, ready for a slide
    OUTDIR/lung_focus_more.png       more examples
    OUTDIR/lung_focus_quality.png    mask size per class and the most unusual masks
    OUTDIR/lung_focus_stats.json     the numbers behind the quality figure
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import report_utils as ru

REPO = "PedroRASB/COVID-19-detection-with-lung-segmentation"
CODE_URL = f"https://raw.githubusercontent.com/{REPO}/main/StackedNet.py"
WEIGHTS_URL = f"https://media.githubusercontent.com/media/{REPO}/main/SegmentationDNN.pt"
# The files below are the exact versions that were checked. Any other version stops the script.
CODE_SHA256 = "84ccede2be88a9d5984ceb0bdc3edc937b82e3a08c93d964552a08e6cc869c20"
WEIGHTS_SHA256 = "8a04a99af27624d8723047cf0a9768ede046c34be04bfd4d71be86ee4ae720ad"

SIZE = 224
LOW_AREA, HIGH_AREA = 0.10, 0.55   # our own rule of thumb for masks that look unusual


# ---------------------------------------------------------------------------
# Getting the paper's model
# ---------------------------------------------------------------------------
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, expected):
    if os.path.exists(path) and sha256_of(path) == expected:
        return
    print(f"[INFO] downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    got = sha256_of(path)
    if got != expected:
        os.remove(path)
        raise SystemExit(f"The downloaded file {os.path.basename(path)} does not match the version we checked.\n"
                         f"Expected {expected}\nGot      {got}\nNothing was loaded.")
    print(f"[INFO] {os.path.basename(path)} verified")


def load_segmenter(model_dir, device):
    import torch

    os.makedirs(model_dir, exist_ok=True)
    code_path = os.path.join(model_dir, "StackedNet.py")
    weights_path = os.path.join(model_dir, "SegmentationDNN.pt")
    fetch(CODE_URL, code_path, CODE_SHA256)
    fetch(WEIGHTS_URL, weights_path, WEIGHTS_SHA256)

    sys.path.insert(0, model_dir)
    import StackedNet

    # the saved file refers to its classes by the names they had when it was saved
    sys.modules["unet"] = StackedNet
    main = sys.modules["__main__"]
    for name in ["UNet", "UNetConvBlock", "UNetUpBlock", "StackedNetworks", "Multiply", "TwoLayerOut"]:
        setattr(main, name, getattr(StackedNet, name))
    try:
        stacked = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        stacked = torch.load(weights_path, map_location=device)
    seg = stacked.Segment.to(device).eval()
    return seg


def make_batches(X, batch):
    for s in range(0, len(X), batch):
        yield s, X[s:s + batch]


def prepare_for_unet(rgb_batch):
    """The paper's preprocessing: grayscale, histogram equalization, 0 to 1, three channels."""
    out = np.empty((len(rgb_batch), 3, SIZE, SIZE), dtype=np.float32)
    for i, img in enumerate(rgb_batch):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        eq = cv2.equalizeHist(gray).astype(np.float32) / 255.0
        out[i] = eq[None]
    return out


def lung_probability(seg, X, device, batch):
    import torch

    masks = np.empty((len(X), SIZE, SIZE), dtype=np.uint8)
    t0 = time.time()
    nb = (len(X) + batch - 1) // batch
    with torch.no_grad():
        for b, (s, chunk) in enumerate(make_batches(X, batch)):
            inp = torch.from_numpy(prepare_for_unet(chunk)).to(device)
            out = seg(inp)
            p = torch.softmax(out, dim=1)[:, 1]
            masks[s:s + len(chunk)] = np.round(p.cpu().numpy() * 255).astype(np.uint8)
            if b == 2 or (b + 1) % 20 == 0 or b + 1 == nb:
                done = s + len(chunk)
                rate = done / (time.time() - t0)
                left = (len(X) - done) / rate / 60
                print(f"[INFO] lung finder: {done}/{len(X)} images, {rate:.1f} images per second, "
                      f"about {left:.1f} minutes left")
    return masks


# ---------------------------------------------------------------------------
# Writing the two new copies of the dataset
# ---------------------------------------------------------------------------
def new_relative_names(rels):
    """Same folder and file name, with a png extension. Falls back to adding .png if names would clash."""
    swapped = [os.path.splitext(r)[0] + ".png" for r in rels]
    if len(set(swapped)) == len(swapped):
        return swapped
    return [r + ".png" for r in rels]


def write_variants(X, masks, rels_new, out_root, workers=4):
    lung_root = os.path.join(out_root, "lung", "all")
    out_root_ = os.path.join(out_root, "outside", "all")
    for r in rels_new:
        for root in (lung_root, out_root_):
            os.makedirs(os.path.dirname(os.path.join(root, r)), exist_ok=True)

    def one(i):
        m = (masks[i].astype(np.float32) / 255.0)[..., None]
        img = X[i].astype(np.float32)
        lung = np.round(img * m).astype(np.uint8)
        outside = np.round(img * (1.0 - m)).astype(np.uint8)
        Image.fromarray(lung).save(os.path.join(lung_root, rels_new[i]), compress_level=1)
        Image.fromarray(outside).save(os.path.join(out_root_, rels_new[i]), compress_level=1)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, range(len(X))))
    return lung_root, out_root_


def check_same_order(orig_rels, lung_root, labels):
    """Confirms the new folder lists in the same order as the original, so the split is identical."""
    classes, files, new_labels, _ = ru.list_dataset(lung_root)
    new_rels = [os.path.relpath(f, lung_root) for f in files]
    if len(new_rels) != len(orig_rels) or not np.array_equal(new_labels, labels):
        raise SystemExit("The new folder does not match the original (different count or labels).")
    expected = new_relative_names(orig_rels)
    if new_rels != expected:
        raise SystemExit("The new folder lists files in a different order, so the train and test split would "
                         "not match the original. Stopping.")
    print("[INFO] order check passed: the new folders list the same images in the same order")


# ---------------------------------------------------------------------------
# Figures and numbers
# ---------------------------------------------------------------------------
def draw_row(axes, img, mask, label=None):
    m = (mask.astype(np.float32) / 255.0)[..., None]
    shown = [img, np.repeat(mask[..., None], 3, axis=2),
             np.round(img * m).astype(np.uint8), np.round(img * (1 - m)).astype(np.uint8)]
    for ax, s in zip(axes, shown):
        ax.imshow(s)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    if label:
        axes[0].set_ylabel(label, fontsize=13, fontweight="bold", color=ru.TEAL, labelpad=8)


def preview_figure(X, masks, labels, classes, per_class, seed, path, title):
    rng = np.random.RandomState(seed)
    rows = []
    for c in range(len(classes)):
        idx = np.where(labels == c)[0]
        rows += [(int(i), classes[c]) for i in rng.choice(idx, min(per_class, len(idx)), replace=False)]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.2, 2.35 * len(rows) + 0.9))
    axes = np.atleast_2d(axes)
    for r, (i, name) in enumerate(rows):
        draw_row(axes[r], X[i], masks[i], label=name.capitalize() if r % per_class == 0 else None)
    heads = ["Original X ray", "Lung probability", "Lungs only", "Outside the lungs only"]
    for ax, h in zip(axes[0], heads):
        ax.set_title(h, fontsize=12.5, fontweight="bold", color=ru.INK, pad=8)
    fig.suptitle(title, fontsize=15, fontweight="bold", color=ru.TEAL, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def outline(img, mask):
    out = img.copy()
    cnts, _ = cv2.findContours((mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (230, 60, 60), 2)
    return out


def quality_report(X, masks, labels, classes, files, path_png, path_json):
    area = (masks > 127).reshape(len(masks), -1).mean(axis=1)
    stats = {"thresholds": {"low": LOW_AREA, "high": HIGH_AREA}, "classes": {}}
    for c, name in enumerate(classes):
        a = area[labels == c]
        stats["classes"][name] = {
            "images": int(len(a)),
            "mean_area": float(a.mean()), "median_area": float(np.median(a)),
            "p5": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
            "below_low": int((a < LOW_AREA).sum()), "above_high": int((a > HIGH_AREA).sum()),
        }
    with open(path_json, "w") as f:
        json.dump(stats, f, indent=2)

    colors = [ru.TEAL, ru.ORANGE, ru.GRAY, "#1C7293"]
    fig = plt.figure(figsize=(13, 9.6))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.25, 1, 1], hspace=0.38, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    bins = np.linspace(0, 0.8, 41)
    for c, name in enumerate(classes):
        ax.hist(area[labels == c], bins=bins, alpha=0.55, color=colors[c % 4], label=name.capitalize())
    ax.axvline(LOW_AREA, color=ru.RED, ls="--", lw=1.3)
    ax.text(LOW_AREA + 0.005, ax.get_ylim()[1] * 0.92, "very small mask", color=ru.RED, fontsize=10)
    ax.set_xlabel("Share of the picture that the lung finder marked as lung", fontsize=11)
    ax.set_ylabel("Number of images", fontsize=11)
    ax.set_title("How much of each X ray was marked as lung", fontsize=14, fontweight="bold", color=ru.TEAL, loc="left")
    ax.legend(frameon=False)
    fig.text(0.5, 0.045, "Red outline: what the lung finder marked. Small masks may mean the lungs were missed. "
             "Large masks may mean too much was marked.", ha="center", fontsize=10, color=ru.GRAY)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    order = np.argsort(area)
    for row, (idxs, text) in enumerate([(order[:6], "Smallest masks"),
                                        (order[::-1][:6], "Largest masks")]):
        for col, i in enumerate(idxs):
            a2 = fig.add_subplot(gs[row + 1, col])
            a2.imshow(outline(X[i], masks[i]))
            a2.set_xticks([])
            a2.set_yticks([])
            a2.set_title(f"{classes[labels[i]]}, {area[i]:.0%}", fontsize=10)
            if col == 0:
                a2.set_ylabel(text, fontsize=11, fontweight="bold", color=ru.TEAL, labelpad=6)
    fig.savefig(path_png, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return stats, area


def main():
    ap = argparse.ArgumentParser(description="Make lung only and outside lung only copies of the dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--model_dir", default=None, help="where the paper's files are stored (default: inside outdir)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto", help="auto, cuda or cpu")
    ap.add_argument("--limit", type=int, default=0, help="only use this many images per class (quick test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    classes, files, labels, skipped = ru.list_dataset(args.dataset)
    if skipped:
        print(f"[INFO] files skipped because they are not a supported image type: {dict(skipped)}")
    keep = np.arange(len(files))
    if args.limit:
        keep = np.concatenate([np.where(labels == c)[0][:args.limit] for c in range(len(classes))])
        print(f"[WARNING] limited test run with {len(keep)} images. Do not train on this output.")
    files = [files[i] for i in keep]
    labels = labels[keep]
    rels = [os.path.relpath(f, args.dataset) for f in files]
    print(f"[INFO] {len(files)} images in {len(classes)} classes: {classes}")
    if device == "cpu" and len(files) > 500:
        print("[WARNING] no GPU found. On a normal CPU this takes about 1 hour for the whole dataset. "
              "In Colab choose Runtime, Change runtime type, and pick a GPU.")

    os.makedirs(args.outdir, exist_ok=True)
    X = ru.load_image_array(files, size=SIZE)
    seg = load_segmenter(args.model_dir or os.path.join(args.outdir, "paper_model"), device)
    t0 = time.time()
    masks = lung_probability(seg, X, device, args.batch)
    print(f"[INFO] lung finding took {(time.time() - t0) / 60:.1f} minutes")
    del seg
    if device == "cuda":
        torch.cuda.empty_cache()

    rels_new = new_relative_names(rels)
    np.savez_compressed(os.path.join(args.outdir, "masks.npz"), mask=masks, rel=np.array(rels),
                        rel_new=np.array(rels_new), labels=labels, classes=np.array(classes))
    t1 = time.time()
    lung_root, outside_root = write_variants(X, masks, rels_new, args.outdir)
    print(f"[INFO] wrote the two new copies in {(time.time() - t1) / 60:.1f} minutes")
    if not args.limit:
        check_same_order(rels, lung_root, labels)

    preview_figure(X, masks, labels, classes, 1, args.seed, os.path.join(args.outdir, "lung_focus_preview.png"),
                   "What the lung finder does to a chest X ray")
    preview_figure(X, masks, labels, classes, 4, args.seed + 1, os.path.join(args.outdir, "lung_focus_more.png"),
                   "More examples of the three image versions")
    stats, area = quality_report(X, masks, labels, classes, files,
                                 os.path.join(args.outdir, "lung_focus_quality.png"),
                                 os.path.join(args.outdir, "lung_focus_stats.json"))

    print("\n[SUMMARY] share of each picture marked as lung")
    for name, s in stats["classes"].items():
        print(f"  {name:10s} images {s['images']:5d}  average {s['mean_area']:.0%}  middle {s['median_area']:.0%}  "
              f"below {LOW_AREA:.0%}: {s['below_low']}  above {HIGH_AREA:.0%}: {s['above_high']}")
    print(f"\nTrain on these folders next:\n  {lung_root}\n  {outside_root}")


if __name__ == "__main__":
    main()
