# USAGE
# python train.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16,VGG19, ResNet50V2, ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import cv2
import os
import time

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 32



def run_model(baseModel, model_name, dataset_dir_name, train_generator, val_generator, class_labels):
    saved_model_path = os.path.sep.join(['.', 'models', f'{model_name}-{dataset_dir_name}-model.h5'])

    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)

    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in baseModel.layers:
        layer.trainable = False

    print("[INFO] compiling model...")
    opt = Adam(learning_rate=INIT_LR)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    fname = os.path.sep.join(['.', 'models', f"best-{model_name}-{dataset_dir_name}-model.h5"])
    checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)
    callbacks = [checkpoint]

    print("[INFO] training head...")
    H = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks)

    print("[INFO] evaluating network...")
    best_model = load_model(fname)
    val_generator.reset()
    predIdxs = best_model.predict(val_generator)
    predIdxs = np.argmax(predIdxs, axis=1)

    true_labels = val_generator.classes
    class_report = classification_report(true_labels, predIdxs, target_names=class_labels)
    print(class_report)

    cm = confusion_matrix(true_labels, predIdxs)

    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title(f"Model: {model_name} Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.sep.join(['.', 'model_performance', f'{model_name}-{dataset_dir_name}-plot.png']))

    print("[INFO] saving COVID-19 detector model...")
    model.save(saved_model_path, save_format="h5")

    return [cm, class_report, model_name]

def train_covid_models(dataset_dir, models=None):
    print("[INFO] setting up data generators...")

    trainAug = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest",
        rescale=1./255,
        validation_split=0.20)

    train_generator = trainAug.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42)

    val_generator = trainAug.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=BS,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42)

    class_labels = list(train_generator.class_indices.keys())
    print(f"Class Labels: {class_labels}")

    if models is None:
        MODELS = [
            {
                "base_model": VGG16(weights="imagenet", include_top=False,
                                    input_tensor=Input(shape=(224, 224, 3))),
                "name": "vgg16"
            },
            {
                "base_model": VGG19(weights="imagenet", include_top=False,
                                    input_tensor=Input(shape=(224, 224, 3))),
                "name": "vgg19"
            }
        ]
    else:
        MODELS = models

    all_model_run_results = []
    dataset_dir_name = dataset_dir.rstrip("/").split("/")[-1]
    for model in MODELS:
        print("---------------------------------------------------")
        print(f"Running Model: {model['name']}")
        start = time.time()
        model_results = run_model(model['base_model'], model['name'], dataset_dir_name,
                                   train_generator, val_generator, class_labels)
        end = time.time()
        all_model_run_results.append(model_results)
        print(f"Finished Model: {model['name']} took {(end-start)/60} minutes")

    return all_model_run_results


if __name__ == '__main__':
    print('Running Train COVID19 Models')
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, default='./dataset/0402',
                    help="path to input dataset")
    ap.add_argument("--epochs", required=False, type=int, default=25,
                    help="Number of training epochs")
    args = vars(ap.parse_args())

    dataset_dir = args["dataset"]
    EPOCHS = args['epochs']

    print(f"Using dataset directory: {dataset_dir}")

    results = train_covid_models(dataset_dir)
    print("------------  Summary -------------")
    for result in results:
        print(f"\n--------- Model: {result[2]} -------------")
        print("Classification Report:")
        print(result[1])
        print("Confusion Matrix")
        df_cm = pd.DataFrame(
            result[0], index=['covid', 'normal', 'pneumonia'], columns=['covid', 'normal', 'pneumonia'],
        )
        print(df_cm.head())

