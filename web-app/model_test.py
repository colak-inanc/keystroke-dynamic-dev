
import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from app.services import DATA_DIR, MODEL_PATH, get_stable_key_id
from app.training import process_file_for_training
from app.imposter_generator import generate_imposter_data
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_model_comprehensive():
    print("DEBUG: Starting test_model_comprehensive...")
    print(f"DEBUG: Model Path: {MODEL_PATH}")
    
    if not MODEL_PATH.exists():
        logging.error(f"Model file not found at {MODEL_PATH}")
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return

    try:
        model = tf.keras.models.load_model(str(MODEL_PATH))
        logging.info(f"Model loaded successfully from {MODEL_PATH}")
        print("DEBUG: Model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        print(f"ERROR: Failed to load model: {e}")
        return

    label_map_path = MODEL_PATH.parent / "label_map.json"
    if not label_map_path.exists():
        logging.error("Label map not found.")
        return
        
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    idx_to_label = {v: k for k, v in label_map.items()}
    logging.info(f"Label Map: {label_map}")

    json_files = list(DATA_DIR.rglob("*.json"))
    logging.info(f"Found {len(json_files)} data files.")
    
    X_time_list = []
    X_key_list = []
    y_true_list = []
    
    # 1. Load Real Data
    for f in json_files:
        if "login_" in f.name and "raw_data" not in f.name: 
            continue 
            
        xt, xk, username = process_file_for_training(f)
        
        if xt is None or username is None:
            continue
            
        if username not in label_map:
            continue
            
        label_id = label_map[username]
        
        for i in range(len(xt)):
            X_time_list.append(xt[i])
            X_key_list.append(xk[i])
            y_true_list.append(label_id)

    if not X_time_list:
        logging.error("No valid data found to test.")
        return
    
    real_sample_count = len(X_time_list)

    # 2. Generate Synthetic Imposters for Test (1:0.5 Ratio)
    if "_IMPOSTER_" in label_map:
        imposter_label = label_map["_IMPOSTER_"]
        # Matches training logic to reduce False Rejection bias
        n_imposter_samples = int(real_sample_count * 0.5)
        
        # Ensure at least some imposters if real data is tiny
        if n_imposter_samples < 50: 
            n_imposter_samples = 50

        logging.info(f"Generating {n_imposter_samples} synthetic imposters for testing...")
        X_time_imp, X_key_imp = generate_imposter_data(n_samples=n_imposter_samples)
        
        for i in range(len(X_time_imp)):
            X_time_list.append(X_time_imp[i])
            X_key_list.append(X_key_imp[i])
            y_true_list.append(imposter_label)
    else:
        logging.warning("'_IMPOSTER_' class not in label map. Skipping synthetic test generation.")

    X_time = np.array(X_time_list, dtype=np.float32)
    X_key = np.array(X_key_list, dtype=np.int32)
    y_true = np.array(y_true_list, dtype=np.int32)
    
    logging.info(f"Testing on {len(X_time)} samples (Real: {real_sample_count} + Imposters).")

    y_pred_probs = model.predict([X_time, X_key], verbose=1)
    
    if y_pred_probs.shape[1] == 1:
        y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_probs, axis=1)

    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    present_target_names = [idx_to_label.get(i, f"Class {i}") for i in unique_labels]

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred, labels=unique_labels, target_names=present_target_names))
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    print(cm)
    
    print("\n" + "="*60)
    print("PER-USER ACCURACY")
    print("="*60)
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    
    for i, idx in enumerate(unique_labels):
        name = idx_to_label.get(idx, f"Class {idx}")
        total = cm_sum[i]
        correct = cm_diag[i]
        if total > 0:
            acc = correct / total * 100
            print(f"{name:<15}: {correct}/{total} ({acc:.2f}%)")
        else:
            print(f"{name:<15}: No samples")

    correct_indices = np.where(y_pred == y_true)[0]
    incorrect_indices = np.where(y_pred != y_true)[0]
    
    max_confidences = np.max(y_pred_probs, axis=1)
    
    avg_conf_correct = np.mean(max_confidences[correct_indices]) if len(correct_indices) > 0 else 0
    avg_conf_wrong = np.mean(max_confidences[incorrect_indices]) if len(incorrect_indices) > 0 else 0
    
    print("\n" + "="*60)
    print("CONFIDENCE ANALYSIS")
    print("="*60)
    print(f"Avg Confidence (Correct Predictions):   {avg_conf_correct:.4f}")
    print(f"Avg Confidence (Incorrect Predictions): {avg_conf_wrong:.4f}")

    print("\n" + "="*60)
    print("ROC CURVE GENERATION")
    print("="*60)

    n_classes = len(label_map)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Handle binary case specifically for label_binarize which returns (n_samples, 1) instead of (n_samples, 2)
    if n_classes == 2:
        y_true_bin = np.hstack((1 - y_true_bin, y_true_bin))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        # If binary classification output is 1D, handle it
        if y_pred_probs.shape[1] == 1:
            if i == 0:
                score = 1 - y_pred_probs.ravel()
            else:
                score = y_pred_probs.ravel()
        else:
            score = y_pred_probs[:, i]
            
        # Only calculate if class is present in y_true to avoid errors, or force calculation if we want full map
        # Check if class exists in truth to avoid UndefinedMetricWarning or errors
        if np.sum(y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], score)
            roc_auc[i] = auc(fpr[i], tpr[i])
            name = idx_to_label.get(i, f"Class {i}")
            print(f"Class {name:<15} AUC: {roc_auc[i]:.4f}")
        else:
             print(f"Class {idx_to_label.get(i, f'Class {i}'):<15} AUC: N/A (No samples)")

    plt.figure()
    # Fixed deprecation warning: Use matplotlib.colormaps
    import matplotlib
    try:
        colors = matplotlib.colormaps['tab10']
    except:
        colors = plt.cm.get_cmap('tab10')
    
    for i in range(n_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                     label=f'{idx_to_label.get(i, i)} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Multi-Class')
    plt.legend(loc="lower right")
    
    save_path = MODEL_PATH.parent / "roc_curve.png"
    plt.savefig(save_path)
    print(f"\nROC Curve saved to: {save_path}")

    # -------------------------------------------------------------------------
    # 2.5 REALISTIC ACCURACY (FROM K-FOLD VALIDATION)
    # -------------------------------------------------------------------------
    metrics_path = MODEL_PATH.parent / "validation_metrics.json"
    if metrics_path.exists():
        print("\n" + "="*60)
        print("REALISTIC ACCURACY ESTIMATE (5-FOLD CROSS VALIDATION)")
        print("="*60)
        try:
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
            
            acc = metrics.get("k_fold_accuracy", 0.0)
            n_folds = metrics.get("n_splits", 5)
            print(f"Mean Validation Accuracy: {acc:.2%} (Averaged over {n_folds} folds)")
            print("Note: This score is calculated on 'unseen' data during training.")
            print("      It represents the true expected performance.")
        except Exception as e:
            print(f"Could not read validation metrics: {e}")


    # -------------------------------------------------------------------------
    # 3. SPECIFIC TEST: FAST RANDOM TYPING (The specific failure case)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print("SPECIFIC TEST: FAST/ERRATIC RANDOM TYPING REJECTION")
    print("="*60)
    
    # Generate manually "Fast" and "Erratic" samples that might have confused the old model
    n_stress_test = 20
    X_stress_time = []
    X_stress_key = []
    
    vocab_pool = list("abcdefghijklmnopqrstuvwxyz 1234567890")
    
    # Generate FAST typist behavior (Low Dwell/Flight)
    for _ in range(n_stress_test):
        seq_time = []
        seq_key = []
        for _ in range(50):
            # Very fast: 30-50ms dwell, 10-30ms flight
            dwell = np.random.normal(40, 5)
            flight = np.random.normal(20, 5)
            delay = dwell + flight
            
            norm_dwell = np.log1p(max(10, dwell)) / 7.0
            norm_flight = np.log1p(max(5, flight)) / 7.0
            norm_delay = np.log1p(max(10, delay)) / 7.0
            
            time_vec = [norm_dwell, norm_flight, norm_delay, 0.2, norm_delay] # Low pressure
            
            key_id = get_stable_key_id(np.random.choice(vocab_pool))
            seq_time.append(time_vec)
            seq_key.append(key_id)
        X_stress_time.append(seq_time)
        X_stress_key.append(seq_key)
        
    X_stress_time = np.array(X_stress_time, dtype=np.float32)
    X_stress_key = np.array(X_stress_key, dtype=np.int32)
    
    stress_probs = model.predict([X_stress_time, X_stress_key], verbose=0)
    
    if "_IMPOSTER_" in label_map:
        imposter_idx = label_map["_IMPOSTER_"]
        stress_preds = np.argmax(stress_probs, axis=1)
        
        n_caught = np.sum(stress_preds == imposter_idx)
        print(f"Fast/Random Samples Tested: {n_stress_test}")
        print(f"Correctly Classified as IMPOSTER: {n_caught}/{n_stress_test}")
        
        # Check confidences for non-imposters (false positives)
        fp_indices = np.where(stress_preds != imposter_idx)[0]
        if len(fp_indices) > 0:
            fp_confidences = np.max(stress_probs[fp_indices], axis=1)
            print(f"False Positives Avg Confidence: {np.mean(fp_confidences):.4f}")
            print("WARNING: Some fast random inputs are still leaking through!")
        else:
            print("SUCCESS: All fast random inputs rejected.")
            
    else:
        print("Skipping stress test explanation (No _IMPOSTER_ class).")


if __name__ == "__main__":
    test_model_comprehensive()
