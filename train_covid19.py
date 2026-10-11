# USAGE
# Baseline run for one model (same split and settings as the earlier runs):
#   python train_covid19.py --dataset dataset --models vgg16 --epochs 25 --run_name baseline
# Same run but the classifier head looks at the whole image:
#   python train_covid19.py --dataset dataset --models vgg16 --epochs 15 --run_name globalpool --pool global
# Fine tuning run that starts from a saved model:
#   python train_covid19.py --dataset dataset --models vgg16 --epochs 5 --run_name finetune \
#       --init_from outputs/vgg16_baseline/best_model.h5 --lr 1e-5
# Clean evaluation protocol (random split, 65 percent train, 20 percent validation, 15 percent test):
#   python train_covid19.py --dataset dataset --models vgg16 --split random --test_frac 0.15 --run_name final

# import the necessary packages
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

import report_utils

# initial learning rate and number of epochs (defaults match the earlier runs so results can be compared)
INIT_LR = 1e-3
EPOCHS = 25

MODEL_BUILDERS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inceptionv3": InceptionV3,
    "resnet50": ResNet50,
}

# first layer to unfreeze when fine tuning (everything after it becomes trainable)
UNFREEZE_FROM = {
    "vgg16": "block5_conv1",
    "vgg19": "block5_conv1",
    "inceptionv3": "mixed9",
    "resnet50": "conv5_block1_1_conv",
}


# ---------------------------------------------------------------------------
# Data: every image is loaded once into RAM as a small uint8 array (about 1 GB in total)
# ---------------------------------------------------------------------------
def get_data(dataset_dir, cache_file):
    classes, files, labels, skipped = report_utils.list_dataset(dataset_dir)
    print(f"[INFO] found {len(files)} images in {len(classes)} classes: {classes}")
    if skipped:
        print(f"[INFO] files skipped because they are not a supported image type: {dict(skipped)}")
    extra = report_utils.duplicate_name_count(files, labels)
    if extra > 0.02 * len(files):
        raise SystemExit(f"STOP: {extra} of {len(files)} files have the same name as another file in the same class. "
                         "The dataset folder probably holds a copied folder, for example all/covid/covid. "
                         "Training on it would put the same images in both the training and validation sets. "
                         "Run the dataset check cell in the notebook, then start again.")
    fingerprint = report_utils.file_fingerprint(files)

    X = None
    if cache_file and os.path.exists(cache_file):
        z = np.load(cache_file)
        if str(z["fingerprint"]) == fingerprint:
            print(f"[INFO] loading images from the cache file {cache_file}")
            X = z["X"]
        else:
            print("[INFO] cache file does not match this dataset, rebuilding it")
    if X is None:
        print("[INFO] reading and resizing every image (a few minutes, only done once)...")
        t0 = time.time()
        X = report_utils.load_image_array(files)
        print(f"[INFO] loaded {len(files)} images in {(time.time() - t0) / 60:.1f} minutes, "
              f"array size {X.nbytes / 1e9:.2f} GB")
        if cache_file:
            np.savez(cache_file, X=X, fingerprint=np.array(fingerprint))
            print(f"[INFO] saved cache file {cache_file}")
    return classes, files, labels, X


def make_split(labels, mode, val_frac, test_frac, seed):
    """
    filename: for each class, the first val_frac of the sorted file names become validation and the
              last test_frac become test. With test_frac 0 this is exactly what the earlier runs used.
    random:   stratified random split, so every set has the same class mix.
    """
    if mode == "filename":
        tr, va, te = [], [], []
        for c in range(labels.max() + 1):
            idx = np.where(labels == c)[0]
            n = len(idx)
            nv, nt = int(val_frac * n), int(test_frac * n)
            va += list(idx[:nv])
            te += list(idx[n - nt:])
            tr += list(idx[nv:n - nt])
        return np.array(tr), np.array(va), np.array(te)

    everything = np.arange(len(labels))
    te = np.array([], dtype=int)
    rest = everything
    if test_frac > 0:
        rest, te = train_test_split(everything, test_size=test_frac, stratify=labels, random_state=seed)
    tr, va = train_test_split(rest, test_size=val_frac / (1 - test_frac), stratify=labels[rest], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


def build_generators(X, Y, tr, va, te, preprocess_fn, batch_size, seed):
    """
    Training images get a small random rotation.
    Validation and test images are never rotated, so every evaluation sees the exact same images.
    """
    train_aug = ImageDataGenerator(rotation_range=15, fill_mode="nearest", preprocessing_function=preprocess_fn)
    plain = ImageDataGenerator(preprocessing_function=preprocess_fn)
    train_gen = train_aug.flow(X[tr], Y[tr], batch_size=batch_size, shuffle=True, seed=seed)
    val_gen = plain.flow(X[va], Y[va], batch_size=batch_size, shuffle=False)
    test_gen = plain.flow(X[te], Y[te], batch_size=batch_size, shuffle=False) if len(te) else None
    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_new_model(model_name, weights, pool):
    base = MODEL_BUILDERS[model_name](weights=weights, include_top=False,
                                      input_tensor=Input(shape=(224, 224, 3)))
    head = base.output
    if pool == "global":
        # averages the whole final feature map, so every part of the image counts
        head = GlobalAveragePooling2D(name="global_pool")(head)
    else:
        # original head: a 4 by 4 average that only covers the top left corner of the final feature map
        head = AveragePooling2D(pool_size=(4, 4))(head)
        head = Flatten(name="flatten")(head)
    head = Dense(64, activation="relu")(head)
    head = Dropout(0.5)(head)
    head = Dense(3, activation="softmax")(head)

    model = Model(inputs=base.input, outputs=head)
    for layer in base.layers:
        layer.trainable = False
    return model


def unfreeze_from(model, start_name):
    """Make the named layer and every layer after it trainable (batch norm layers stay frozen)."""
    names = [l.name for l in model.layers]
    if start_name not in names:
        raise ValueError(f"Layer {start_name} not found. Some layer names: {names[-30:]}")
    found, count = False, 0
    for layer in model.layers:
        if layer.name == start_name:
            found = True
        if found and not isinstance(layer, BatchNormalization):
            layer.trainable = True
            count += 1
    return count


def evaluate(model, gen, y_true, classes, title):
    gen.reset()
    probs = model.predict(gen, verbose=0)
    pred = np.argmax(probs, axis=1)
    print(f"{title}")
    print(classification_report(y_true, pred, target_names=classes))
    return probs, pred


def run_model(model_name, args, data):
    classes, files, labels, X = data
    tag = f"{model_name}_{args.run_name}"
    run_dir = os.path.join(args.outdir, tag)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_model.h5")

    tf.keras.utils.set_random_seed(args.seed)
    preprocess_fn = report_utils.get_preprocess(model_name, args.preprocess, args.clahe)

    tr, va, te = make_split(labels, args.split, args.val_frac, args.test_frac, args.seed)
    Y = to_categorical(labels, num_classes=len(classes))
    train_gen, val_gen, test_gen = build_generators(X, Y, tr, va, te, preprocess_fn, args.batch_size, args.seed)
    print(f"[INFO] split mode {args.split}: {len(tr)} train, {len(va)} validation, {len(te)} test images")
    print(f"Class Labels: {classes}")

    if args.init_from:
        print(f"[INFO] loading starting model from {args.init_from}")
        model = load_model(args.init_from)
        start = args.unfreeze_from or UNFREEZE_FROM[model_name]
        n = unfreeze_from(model, start)
        print(f"[INFO] unfroze {n} layers starting at {start}")
    else:
        weights = None if args.weights == "none" else args.weights
        model = build_new_model(model_name, weights, args.pool)

    print("[INFO] compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=args.lr),
                  metrics=["accuracy"])

    checkpoint = ModelCheckpoint(best_path, monitor="val_accuracy", mode="max",
                                 save_best_only=True, verbose=1)

    print("[INFO] training...")
    start_time = time.time()
    H = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        verbose=2,
        callbacks=[checkpoint])
    minutes = (time.time() - start_time) / 60

    print("[INFO] evaluating network...")
    best_model = load_model(best_path)
    true_labels = labels[va]
    probs, pred_idxs = evaluate(best_model, val_gen, true_labels, classes, "Validation results")
    class_report = classification_report(true_labels, pred_idxs, target_names=classes)
    cm = confusion_matrix(true_labels, pred_idxs)

    # save everything the reports need
    np.save(os.path.join(run_dir, "val_probs.npy"), probs)
    np.save(os.path.join(run_dir, "val_labels.npy"), true_labels)
    with open(os.path.join(run_dir, "val_files.json"), "w") as f:
        json.dump([files[i] for i in va], f)
    with open(os.path.join(run_dir, "class_labels.json"), "w") as f:
        json.dump(classes, f)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in H.history.items()}, f)

    # held out test set: never used for training or for picking the best epoch
    if test_gen is not None:
        test_labels = labels[te]
        t_probs, t_pred = evaluate(best_model, test_gen, test_labels, classes, "HELD OUT TEST results")
        rep = classification_report(test_labels, t_pred, target_names=classes, output_dict=True)
        np.save(os.path.join(run_dir, "test_probs.npy"), t_probs)
        np.save(os.path.join(run_dir, "test_labels.npy"), test_labels)
        with open(os.path.join(run_dir, "test_files.json"), "w") as f:
            json.dump([files[i] for i in te], f)
        with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
            json.dump({"accuracy": rep["accuracy"], "n": int(len(te)),
                       "recall": {c: rep[c]["recall"] for c in classes},
                       "precision": {c: rep[c]["precision"] for c in classes}}, f, indent=2)

    config = {
        "model": model_name, "run_name": args.run_name, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "preprocess": args.preprocess, "clahe": args.clahe,
        "pool": args.pool, "split": args.split, "val_frac": args.val_frac, "test_frac": args.test_frac,
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "init_from": args.init_from, "dataset": args.dataset, "seed": args.seed,
        "weights": args.weights, "train_minutes": minutes,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # neat one page report (training results are already saved, so a report problem cannot lose them)
    try:
        png = report_utils.make_model_report(run_dir)
        print(f"[INFO] report picture saved to {png}")
    except Exception as e:
        print(f"[WARN] could not make the report picture: {e}")

    return [cm, class_report, tag, run_dir, minutes]


if __name__ == '__main__':
    print('Running Train COVID19 Models')
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, default='./dataset/0402',
                    help="path to input dataset")
    ap.add_argument("--epochs", required=False, type=int, default=25,
                    help="Number of training epochs")
    ap.add_argument("--models", nargs="+", default=["vgg16", "vgg19", "inceptionv3"],
                    choices=list(MODEL_BUILDERS.keys()), help="which models to train")
    ap.add_argument("--run_name", default="baseline", help="name for this run, used in output folder names")
    ap.add_argument("--outdir", default="./outputs", help="folder where all run results are saved")
    ap.add_argument("--lr", type=float, default=INIT_LR, help="learning rate")
    ap.add_argument("--batch_size", type=int, default=8, help="images per training step")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--pool", default="avg4", choices=["avg4", "global"],
                    help="avg4 is the original head, global averages the whole final feature map")
    ap.add_argument("--split", default="filename", choices=["filename", "random"],
                    help="filename matches the earlier runs, random is a stratified random split")
    ap.add_argument("--val_frac", type=float, default=0.20, help="share of images used for validation")
    ap.add_argument("--test_frac", type=float, default=0.0,
                    help="share of images held out as a final test set (0 means no test set)")
    ap.add_argument("--cache_file", default=None,
                    help="where to keep the loaded images (default: next to the dataset folder, use none to disable)")
    ap.add_argument("--init_from", default=None,
                    help="start from a saved model and fine tune it (use with a single model)")
    ap.add_argument("--unfreeze_from", default=None, help="layer name to start unfreezing from")
    ap.add_argument("--preprocess", default="rescale", choices=["rescale", "imagenet"],
                    help="rescale divides pixels by 255, imagenet uses each model's own original scaling")
    ap.add_argument("--clahe", action="store_true", help="boost image contrast before training")
    ap.add_argument("--weights", default="imagenet", help="imagenet, or none for a quick code test")
    ap.add_argument("--max_per_class", type=int, default=0,
                    help="use only the first N images of each class (0 means all). For quick tests of the whole pipeline")
    args = ap.parse_args()

    if args.init_from and len(args.models) != 1:
        raise SystemExit("--init_from works with exactly one model, for example --models vgg16")
    if args.val_frac + args.test_frac >= 0.9:
        raise SystemExit("val_frac plus test_frac must leave most images for training")

    EPOCHS = args.epochs
    print(f"Using dataset directory: {args.dataset}")

    if args.cache_file is None:
        parent = os.path.dirname(os.path.abspath(args.dataset.rstrip("/")))
        args.cache_file = os.path.join(parent, "image_cache_224.npz")
    elif args.cache_file.lower() == "none":
        args.cache_file = None
    data = get_data(args.dataset, args.cache_file)
    if args.max_per_class:
        classes_, files_, labels_, X_ = data
        keep = np.concatenate([np.where(labels_ == c)[0][:args.max_per_class] for c in range(len(classes_))])
        data = (classes_, [files_[i] for i in keep], labels_[keep], X_[keep])
        print(f"[INFO] quick test: using only the first {args.max_per_class} images of each class ({len(keep)} images)")

    results = []
    for name in args.models:
        print("---------------------------------------------------")
        print(f"Running Model: {name}")
        res = run_model(name, args, data)
        results.append(res)
        print(f"Finished Model: {name} took {res[4]} minutes")

    print("------------  Summary -------------")
    for cm, report, tag, run_dir, minutes in results:
        print(f"\n--------- Model: {tag} -------------")
        print("Classification Report:")
        print(report)
        print("Confusion Matrix")
        print(pd.DataFrame(cm, index=['covid', 'normal', 'pneumonia'],
                           columns=['covid', 'normal', 'pneumonia']))
