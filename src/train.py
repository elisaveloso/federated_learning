import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from tensorflow.keras import mixed_precision

# Adjust path to import from src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.model import get_efficientnet_model, get_resnet_model, get_vit_model
from utils.training_utils import F1Score, plot_training_history, plot_comparison_boxplots

# Configuration
DATASET_DIR = '/home/elisaveloso/federated_learning/src/data/datasets/daninhas__25-12-03'
IMG_SIZE = (224, 224)
BATCH_SIZE = 4
EPOCHS = 10 
NUM_RUNS = 3 # repeats for boxplots
RESULTS_DIR = 'results'
SEED = 123
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15

# Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

def collect_original_image_paths(dataset_dir, class_names):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    file_paths = []
    labels = []

    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in os.listdir(class_dir):
            lower_name = file_name.lower()
            if not lower_name.endswith(image_extensions):
                continue
            # Ignore previously generated augmented images.
            if 'aug' in lower_name:
                continue

            file_paths.append(os.path.join(class_dir, file_name))
            labels.append(label_idx)

    return file_paths, labels


def stratified_split(file_paths, labels, test_split=0.15, val_split=0.15, seed=123):
    rng = random.Random(seed)
    class_to_indices = {}

    for idx, label in enumerate(labels):
        class_to_indices.setdefault(label, []).append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for class_indices in class_to_indices.values():
        rng.shuffle(class_indices)
        n_total = len(class_indices)

        n_test = int(round(n_total * test_split))
        n_test = min(max(n_test, 1), n_total - 2) if n_total >= 3 else max(0, n_total - 2)

        remaining_after_test = n_total - n_test
        val_ratio_on_remaining = val_split / (1.0 - test_split) if (1.0 - test_split) > 0 else 0.0
        n_val = int(round(remaining_after_test * val_ratio_on_remaining))
        n_val = min(max(n_val, 1), remaining_after_test - 1) if remaining_after_test >= 2 else 0

        test_part = class_indices[:n_test]
        val_part = class_indices[n_test:n_test + n_val]
        train_part = class_indices[n_test + n_val:]

        test_indices.extend(test_part)
        val_indices.extend(val_part)
        train_indices.extend(train_part)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    def gather(indices):
        return [file_paths[i] for i in indices], [labels[i] for i in indices]

    train_paths, train_labels = gather(train_indices)
    val_paths, val_labels = gather(val_indices)
    test_paths, test_labels = gather(test_indices)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def build_dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return img, label

    if training:
        ds = ds.shuffle(buffer_size=max(len(paths), BATCH_SIZE), seed=SEED)

    ds = ds.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    return ds


def mirror_augment_training_dataset(train_ds):
    mirrored_ds = train_ds.map(
        lambda images, labels: (tf.image.flip_left_right(images), labels),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.concatenate(mirrored_ds)
    train_ds = train_ds.shuffle(buffer_size=100, seed=SEED)
    return train_ds


def load_data():
    # Labels: 'nao_daninha' 0, 'daninha' 1
    class_names = ['nao_daninha', 'daninha']

    print(f"Loading data from {DATASET_DIR}...")
    print(f"Class mapping: {class_names}")

    file_paths, labels = collect_original_image_paths(DATASET_DIR, class_names)
    if not file_paths:
        raise ValueError(f"No original images found in {DATASET_DIR}.")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_split(
        file_paths,
        labels,
        test_split=TEST_SPLIT,
        val_split=VAL_SPLIT,
        seed=SEED,
    )

    print(f"Total original images: {len(file_paths)}")
    print(f"Train images (pre-augmentation): {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Test images: {len(test_paths)}")

    train_ds = build_dataset(train_paths, train_labels, training=True)
    train_ds = mirror_augment_training_dataset(train_ds)
    val_ds = build_dataset(val_paths, val_labels, training=False)
    test_ds = build_dataset(test_paths, test_labels, training=False)

    # Autotune for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

def run_experiment():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    train_ds, val_ds, test_ds = load_data()

  #  models_to_test = {
  #      'EfficientNetB0': get_efficientnet_model,
  #      'ResNet50': get_resnet_model,
  #      'ViT': get_vit_model
  #  }
    
    models_to_test = {
        'ViT': get_vit_model
    }

    all_results = []
    
    for model_name, model_fn in models_to_test.items():
        print(f"\nTraining Model: {model_name}")
        model_histories = []
        
        for run_idx in range(NUM_RUNS):
            print(f"  Run {run_idx + 1}/{NUM_RUNS}")
            
            # Re-instantiate model for each run
            model = model_fn(input_shape=(224, 224, 3))
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Recall(name='recall'), F1Score(name='f1_score')]
            )

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                verbose=1
            )
            model_histories.append(history)

            # Evaluate on Test Set
            print("  Evaluating on Test Set...")
            test_loss, test_acc, test_recall, test_f1 = model.evaluate(test_ds, verbose=0)
            
            # Store results
            all_results.append({
                'Model': model_name,
                'Run': run_idx,
                'Metric': 'Accuracy',
                'Value': test_acc,
                'Set': 'Test'
            })
            all_results.append({
                'Model': model_name,
                'Run': run_idx,
                'Metric': 'Recall',
                'Value': test_recall,
                'Set': 'Test'
            })
            all_results.append({
                'Model': model_name,
                'Run': run_idx,
                'Metric': 'F1-Score',
                'Value': test_f1,
                'Set': 'Test'
            })
            

        # Plot training history (averaged over runs for the line graph)
        plot_training_history(model_histories, model_name, RESULTS_DIR)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save raw results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'final_metrics.csv'), index=False)
    print("\nResults saved to results/final_metrics.csv")
    
    # Plot Boxplots
    plot_comparison_boxplots(results_df, RESULTS_DIR)
    print("Plots saved to results/ directory.")

if __name__ == '__main__':
    run_experiment()
