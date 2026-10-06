# USAGE
# Baseline run for one model:
#   python train_covid19.py --dataset dataset --models vgg16 --epochs 25 --run_name baseline
# Fine tuning run that starts from the baseline model:
#   python train_covid19.py --dataset dataset --models vgg16 --epochs 10 --run_name finetune \
#       --init_from outputs/vgg16_baseline/best_model.h5 --lr 1e-5

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
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import report_utils

# initialize the initial learning rate, number of epochs to train for,
# and batch size (these stay the same as the earlier runs so results can be compared)
INIT_LR = 1e-3
EPOCHS = 25
BS = 8
SEED = 42

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


def build_generators(dataset_dir, preprocess_fn):
    """
    Training images get a small random rotation.
    Validation images are NOT rotated, so every evaluation sees the exact same images.
    Both use the same 80/20 split, so no image is in both sets.
    """
    train_aug = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest",
        preprocessing_function=preprocess_fn,
        validation_split=0.20)
    val_aug = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        validation_split=0.20)

    train_generator = train_aug.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=SEED)

    val_generator = val_aug.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED)
    return train_generator, val_generator


def build_new_model(model_name, weights):
    base = MODEL_BUILDERS[model_name](weights=weights, include_top=False,
                                      input_tensor=Input(shape=(224, 224, 3)))
    head = base.output
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


def run_model(model_name, args, dataset_dir):
    tag = f"{model_name}_{args.run_name}"
    run_dir = os.path.join(args.outdir, tag)
    os.makedirs(run_dir, exist_ok=True)
    best_path = os.path.join(run_dir, "best_model.h5")

    tf.keras.utils.set_random_seed(SEED)
    preprocess_fn = report_utils.get_preprocess(model_name, args.preprocess, args.clahe)
    train_generator, val_generator = build_generators(dataset_dir, preprocess_fn)
    class_labels = list(train_generator.class_indices.keys())
    print(f"Class Labels: {class_labels}")

    if args.init_from:
        print(f"[INFO] loading starting model from {args.init_from}")
        model = load_model(args.init_from)
        start = args.unfreeze_from or UNFREEZE_FROM[model_name]
        n = unfreeze_from(model, start)
        print(f"[INFO] unfroze {n} layers starting at {start}")
    else:
        weights = None if args.weights == "none" else args.weights
        model = build_new_model(model_name, weights)

    print("[INFO] compiling model...")
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=args.lr),
                  metrics=["accuracy"])

    checkpoint = ModelCheckpoint(best_path, monitor="val_accuracy", mode="max",
                                 save_best_only=True, verbose=1)

    print("[INFO] training...")
    start_time = time.time()
    H = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        verbose=2,
        callbacks=[checkpoint])
    minutes = (time.time() - start_time) / 60

    print("[INFO] evaluating network...")
    best_model = load_model(best_path)
    val_generator.reset()
    probs = best_model.predict(val_generator, verbose=0)
    pred_idxs = np.argmax(probs, axis=1)
    true_labels = val_generator.classes

    class_report = classification_report(true_labels, pred_idxs, target_names=class_labels)
    print(class_report)
    cm = confusion_matrix(true_labels, pred_idxs)

    # save everything the reports need
    np.save(os.path.join(run_dir, "val_probs.npy"), probs)
    np.save(os.path.join(run_dir, "val_labels.npy"), true_labels)
    with open(os.path.join(run_dir, "val_files.json"), "w") as f:
        json.dump([os.path.join(val_generator.directory, p) for p in val_generator.filenames], f)
    with open(os.path.join(run_dir, "class_labels.json"), "w") as f:
        json.dump(class_labels, f)
    with open(os.path.join(run_dir, "history.json"), "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in H.history.items()}, f)
    config = {
        "model": model_name, "run_name": args.run_name, "epochs": args.epochs, "lr": args.lr,
        "batch_size": BS, "preprocess": args.preprocess, "clahe": args.clahe,
        "init_from": args.init_from, "dataset": dataset_dir, "seed": SEED,
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
    ap.add_argument("--init_from", default=None,
                    help="start from a saved model and fine tune it (use with a single model)")
    ap.add_argument("--unfreeze_from", default=None, help="layer name to start unfreezing from")
    ap.add_argument("--preprocess", default="rescale", choices=["rescale", "imagenet"],
                    help="rescale divides pixels by 255, imagenet uses each model's own original scaling")
    ap.add_argument("--clahe", action="store_true", help="boost image contrast before training")
    ap.add_argument("--weights", default="imagenet", help="imagenet, or none for a quick code test")
    args = ap.parse_args()

    if args.init_from and len(args.models) != 1:
        raise SystemExit("--init_from works with exactly one model, for example --models vgg16")

    EPOCHS = args.epochs
    print(f"Using dataset directory: {args.dataset}")

    results = []
    for name in args.models:
        print("---------------------------------------------------")
        print(f"Running Model: {name}")
        res = run_model(name, args, args.dataset)
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
