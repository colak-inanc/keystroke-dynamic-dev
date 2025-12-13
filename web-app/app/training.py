import json
import logging
import threading
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.utils import class_weight
from sklearn.model_selection import GroupShuffleSplit
from app.services import DATA_DIR, MODEL_PATH, VOCAB_SIZE, get_stable_key_id, reload_model
from app.imposter_generator import generate_imposter_data

training_lock = threading.Lock()

TIME_SCALE = 7.0
MAX_LATENCY = 3000.0
TARGET_LENGTH = 50
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001

def process_file_for_training(filepath):
    """
    Processes a single JSON file to extract feature vectors and key sequences.
    MUST MATCH app.services.prepare_sequence_for_model EXACTLY.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        keystrokes = data.get('keystrokes', [])
        if data.get('keystrokes_2'):
            keystrokes.extend(data['keystrokes_2'])
        if data.get('keystrokes_3'):
            keystrokes.extend(data['keystrokes_3'])
        username = data.get('username')
        
        if not keystrokes or not username:
            return None, None, None

        sequence_time = []
        sequence_keys = []
        
        keydown_events = {}
        previous_keyup_time = None
        previous_keydown_time = None

        # Pre-normalization (Space and Single Char Lowercase)
        for k in keystrokes:
            if k['key'] == "Space": k['key'] = " "
            if len(k['key']) == 1: k['key'] = k['key'].lower()

        for idx, event in enumerate(keystrokes):
            key = event['key']
            event_type = event['event_type']
            timestamp = event['timestamp']

            if event_type == "keydown":
                curr_dd = 0.0
                if previous_keydown_time is not None:
                    curr_dd = timestamp - previous_keydown_time
                
                keydown_events.setdefault(key, []).append((timestamp, curr_dd))
                previous_keydown_time = timestamp
                if key in {"Backspace", "Shift"}: continue

            elif event_type == "keyup":
                if key in {"Shift", "Backspace"}:
                    if key in keydown_events: 
                       keydown_events[key].pop()
                       if not keydown_events[key]: del keydown_events[key]
                    continue

                dwell_time = 0.0
                curr_dd = 0.0
                
                # Fix: Stack (LIFO) pop
                kd_data = None
                if key in keydown_events and keydown_events[key]:
                    kd_data = keydown_events[key].pop()
                    if not keydown_events[key]:
                        del keydown_events[key]
                
                keydown_time = None
                if kd_data:
                    keydown_time, curr_dd = kd_data
                    dwell_time = timestamp - keydown_time
                
                flight_time = (keydown_time - previous_keyup_time) if (previous_keyup_time and keydown_time) else 0.0
                inter_key_delay = (timestamp - keystrokes[idx - 1]['timestamp']) if idx > 0 else 0.0

                # Clamping
                dwell_time = min(dwell_time, MAX_LATENCY)
                flight_time = min(flight_time, MAX_LATENCY)
                inter_key_delay = min(inter_key_delay, MAX_LATENCY)

                # Log1p Normalization
                norm_dwell = np.log1p(dwell_time) / TIME_SCALE
                norm_flight = np.log1p(flight_time) / TIME_SCALE if flight_time > 0 else 0.0
                norm_delay = np.log1p(inter_key_delay) / TIME_SCALE
                
                # Base features: [Dwell, Flight, Delay]
                time_vector = [float(norm_dwell), float(norm_flight), float(norm_delay)]
                
                # Feature 4: Pressure (Simulated)
                time_vector.append(min(dwell_time / 200.0, 1.0))
                
                # Feature 5: Down-Down Latency
                dd_lat = max(0.0, min(curr_dd, MAX_LATENCY))
                time_vector.append(np.log1p(dd_lat) / TIME_SCALE)

                # KEY ID (Stable Mapping)
                key_id = get_stable_key_id(key)

                sequence_time.append(time_vector)
                sequence_keys.append(key_id)
                
                previous_keyup_time = timestamp

        # Sliding Window / Padding to TARGET_LENGTH
        final_time_sequences = []
        final_key_sequences = []

        def get_padded_window(seq, is_key=False):
            res = list(seq)
            if len(res) < TARGET_LENGTH:
                pad_val = 0 if is_key else [0.0] * 5
                while len(res) < TARGET_LENGTH:
                    res.append(pad_val)
            return res[:TARGET_LENGTH]

        if len(sequence_time) < TARGET_LENGTH:
            final_time_sequences.append(get_padded_window(sequence_time, is_key=False))
            final_key_sequences.append(get_padded_window(sequence_keys, is_key=True))
        else:
            step = max(1, TARGET_LENGTH // 2)
            for i in range(0, len(sequence_time) - TARGET_LENGTH + 1, step):
                final_time_sequences.append(sequence_time[i : i + TARGET_LENGTH])
                final_key_sequences.append(sequence_keys[i : i + TARGET_LENGTH])

            remaining = len(sequence_time)
            if remaining > TARGET_LENGTH and (remaining - TARGET_LENGTH) % step != 0:
                 final_time_sequences.append(sequence_time[-TARGET_LENGTH:])
                 final_key_sequences.append(sequence_keys[-TARGET_LENGTH:])
        
        return np.array(final_time_sequences), np.array(final_key_sequences), username

    except Exception as e:
        logging.error(f"Error processing file {filepath}: {e}")
        return None, None, None

def build_model(num_classes):
    """
    Robust, Balanced Architecture for Production.
    - Uses GaussianNoise for input robustness.
    - Uses Bi-LSTM / LSTM with L2 Regularization.
    - Uses Dropout (0.4) for regularization.
    """
    input_time = layers.Input(shape=(TARGET_LENGTH, 5), name="input_time")
    
    masked_time = layers.Masking(mask_value=0.0)(input_time)
    
    noisy_time = layers.GaussianNoise(0.01)(masked_time)
    
    # Bi-LSTM for time
    bilstm_time = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))
    )(noisy_time)
    bilstm_time = layers.BatchNormalization()(bilstm_time)
    bilstm_time = layers.Dropout(0.5)(bilstm_time)

    # 2. Key Sequence Input (Categorical)
    input_key = layers.Input(shape=(TARGET_LENGTH,), dtype="int32", name="input_key")
    
    # Embedding for key IDs
    embedding = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32, mask_zero=True)(input_key)
    # Add minimal regular dropout to embedding
    embedding = layers.SpatialDropout1D(0.2)(embedding)
    
    # LSTM for keys
    lstm_key = layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))(embedding)
    lstm_key = layers.BatchNormalization()(lstm_key)
    lstm_key = layers.Dropout(0.4)(lstm_key)

    # 3. Merge & Classify
    merged = layers.Concatenate()([bilstm_time, lstm_key])
    
    dense = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.001))(merged)
    dense = layers.Dropout(0.5)(dense)
    
    output = layers.Dense(num_classes, activation="softmax")(dense)

    model = models.Model(inputs=[input_time, input_key], outputs=output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipvalue=0.5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_model():
    if not training_lock.acquire(blocking=False):
        logging.warning("Training already in progress, skipping request.")
        return

    try:
        logging.info("Starting PRODUCTION model training...")
        
        # Sort files to ensure stable session ordering (s1, s2, s3)
        json_files = sorted(list(DATA_DIR.rglob("*.json")))
        logging.info(f"Found {len(json_files)} total data files.")
        
        all_X_time = []
        all_X_key = []
        all_y = []
        
        label_map = {}
        current_label = 0
        
        # 1. Load Real Data
        groups = [] # Track sessions for GroupKFold
        
        for file_idx, f in enumerate(json_files):
            if "login_" in f.name and "raw_data" not in f.name: continue 
            
            xt, xk, user = process_file_for_training(f)
            if xt is not None and user is not None:
                if user not in label_map:
                    label_map[user] = current_label
                    current_label += 1
                
                label_id = label_map[user]
                for i in range(len(xt)):
                    all_X_time.append(xt[i])
                    all_X_key.append(xk[i])
                    all_y.append(label_id)
                    groups.append(file_idx)

        real_sample_count = len(all_X_time)
        if real_sample_count == 0:
            logging.error("No real data found for training.")
            return

        logging.info(f"Total REAL samples: {real_sample_count}")

        # 2. Dynamic Imposter Data Generation
        # User requested 1:0.5 ratio (Real:Imposter) to reduce False Rejection bias
        n_imposter_samples = int(real_sample_count * 0.5)
        
        # Minimum safe amount for stability
        if n_imposter_samples < 50:
            n_imposter_samples = 50

        logging.info(f"Generating {n_imposter_samples} synthetic imposter samples (Ratio 1:0.5)...")
        X_time_imp, X_key_imp = generate_imposter_data(n_samples=n_imposter_samples)
        
        imposter_label = current_label
        label_map["_IMPOSTER_"] = imposter_label
        
        for i in range(len(X_time_imp)):
            all_X_time.append(X_time_imp[i])
            all_X_key.append(X_key_imp[i])
            all_y.append(imposter_label)
            # Assign unique group ID to each imposter sample to allow random splitting
            groups.append(100000 + i)
            
        logging.info(f"Final Dataset Size: {len(all_X_time)} samples. Classes: {len(label_map)}")

        if len(label_map) < 2:
            logging.warning("Not enough classes to train (Need at least 1 user + Imposters).")
            return

        X_time_all = np.array(all_X_time, dtype=np.float32)
        X_key_all = np.array(all_X_key, dtype=np.int32)
        y_all = np.array(all_y, dtype=np.int32)
        groups_all = np.array(groups)

        # 3. Stratified Group Splitting (Per-User)
        logging.info("Splitting Train/Val using Stratified Group Logic (per user)...")
        
        train_indices_list = []
        val_indices_list = []
        
        unique_classes = np.unique(y_all)
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        for cls in unique_classes:
            cls_mask = (y_all == cls)
            cls_indices = np.where(cls_mask)[0]
            cls_groups = groups_all[cls_indices]
            
            # If a user has only 1 group (session), we must put it in train 
            # (or val, but train is better for learning).
            # GroupShuffleSplit might fail or put 0 in val if n_groups=1.
            n_groups = len(np.unique(cls_groups))
            if n_groups < 2:
                # If only 1 session, we MUST split it internally (e.g. 80/20) to have SOME validation data.
                # This risks slight leakage due to sliding windows, but is better than NO validation for the user.
                logging.warning(f"Class {cls} has only 1 session. Forcing internal split (Warning: Potential leakage).")
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                # Dummy 'y' for split
                dummy_y = np.zeros(len(cls_indices))
                tr_i, val_i = next(sss.split(cls_indices, dummy_y))
                train_indices_list.extend(cls_indices[tr_i])
                val_indices_list.extend(cls_indices[val_i])
                continue

            try:
                tr_i, val_i = next(gss.split(cls_indices, groups=cls_groups))
                # Map back to global indices
                train_indices_list.extend(cls_indices[tr_i])
                val_indices_list.extend(cls_indices[val_i])
            except Exception:
                # Fallback to random split if group split fails
                logging.warning(f"Group split failed for class {cls}. Falling back to random split.")
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                dummy_y = np.zeros(len(cls_indices))
                tr_i, val_i = next(sss.split(cls_indices, dummy_y))
                train_indices_list.extend(cls_indices[tr_i])
                val_indices_list.extend(cls_indices[val_i])
                
        # Convert to numpy arrays
        train_idx = np.array(train_indices_list)
        val_idx = np.array(val_indices_list)
        
        # Shuffle the resulting sets to mix classes
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)

        X_time_train, X_time_val = X_time_all[train_idx], X_time_all[val_idx]
        X_key_train, X_key_val = X_key_all[train_idx], X_key_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]
        
        logging.info(f"Stratified Split Result -> Train: {len(y_train)} samples, Val: {len(y_val)} samples.")
        
        # Verify coverage
        covered_classes = len(np.unique(y_val))
        total_classes = len(unique_classes)
        logging.info(f"Validation Class Coverage: {covered_classes}/{total_classes}")

        # SAFETY CHECK: If validation set is empty (e.g. all users have only 1 session),
        # force a standard shuffle split to avoid crash. Leakage risk is acceptable vs crash.
        if len(val_idx) == 0:
             logging.warning("Validation set is empty! (Single sessions detected). Falling back to random split.")
             # Take 20% of training data for validation
             indices = np.arange(len(train_idx))
             np.random.shuffle(indices)
             split_point = int(len(indices) * 0.8)
             
             train_idx_new = train_idx[indices[:split_point]]
             val_idx_new = train_idx[indices[split_point:]]
             
             X_time_train, X_time_val = X_time_all[train_idx_new], X_time_all[val_idx_new]
             X_key_train, X_key_val = X_key_all[train_idx_new], X_key_all[val_idx_new]
             y_train, y_val = y_all[train_idx_new], y_all[val_idx_new]
             
             logging.info(f"Fallback Split -> Train: {len(y_train)}, Val: {len(y_val)}")

        # --- FINAL FULL TRAINING ---
        logging.info("Training Final Production Model (Groupled Split)...")
        
        classes = np.unique(y_train)
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, weights))
        
        model = build_model(num_classes=len(label_map))
        
        # Enhanced Callback Strategy
        callbacks_list = [
            # Stop if validation loss doesn't improve for 15 epochs (Prevents Overfitting)
            callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=15, 
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce LR if validation loss plateaus (Fine-tuning)
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=0.00001,
                verbose=1
            ),
            # Save best model based on validation loss
            callbacks.ModelCheckpoint(
                str(MODEL_PATH), 
                save_best_only=True, 
                monitor='val_loss'
            )
        ]
        
        # Train with explicit Group-based Validation Data
        history = model.fit(
            [X_time_train, X_key_train],
            y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1,
            validation_data=([X_time_val, X_key_val], y_val)
        )
        
        if not MODEL_PATH.exists():
             model.save(MODEL_PATH)
        
        # Ensure label map matches current training
        with open(MODEL_PATH.parent / "label_map.json", "w", encoding='utf-8') as f:
            json.dump(label_map, f)
            
        logging.info("Training completed. Label map saved.")
        
        reload_model()

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
    finally:
        training_lock.release()
