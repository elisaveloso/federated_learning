import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
NUM_RUNS = 3 # Number of repeats for boxplots
RESULTS_DIR = 'results'

# Mixed Precision
mixed_precision.set_global_policy('mixed_float16')

def load_data():
    # Labels: 'nao_daninha' will be 0, 'daninha' will be 1
    class_names = ['nao_daninha', 'daninha']
    
    print(f"Loading data from {DATASET_DIR}...")
    print(f"Class mapping: {class_names}")

    # Load full dataset
    full_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        labels='inferred',
        class_names=class_names,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=123
    )

    # Split into Train (70%), Val (15%), Test (15%)
    # Getting the size to split manually by batches might be tricky if not cached.
    # Simpler approach: Use validation_split feature twice? No, that relies on seed.
    # Let's take batches directly.
    
    full_ds = full_ds.shuffle(50, seed=42)
    ds_batches = tf.data.experimental.cardinality(full_ds).numpy()
    
    if ds_batches < 0:
        # If cardinality is unknown, iterate to count (slow but safe)
        ds_batches = len(list(full_ds))

    train_size = int(0.7 * ds_batches)
    val_size = int(0.15 * ds_batches)
    test_size = ds_batches - train_size - val_size

    train_ds = full_ds.take(train_size)
    remaining = full_ds.skip(train_size)
    val_ds = remaining.take(val_size)
    test_ds = remaining.skip(val_size)

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

    models_to_test = {
        'EfficientNetB0': get_efficientnet_model,
        'ResNet50': get_resnet_model,
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
            
            # Save Train metrics for boxplot? Usually boxplot is for Test metrics.
            # We already stored Test metrics.

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
