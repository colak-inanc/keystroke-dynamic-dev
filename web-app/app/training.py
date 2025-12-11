import json
import logging
import threading
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from app.services import DATA_DIR, MODEL_PATH, get_stable_key_id, reload_model
from app.imposter_generator import generate_imposter_data

training_lock = threading.Lock()

# --- CONFIGURATION (Hyperparameters) ---
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
                keydown_events[key] = timestamp
                previous_keydown_time = timestamp
                if key in {"Backspace", "Shift"}: continue

            elif event_type == "keyup":
                if key in {"Shift", "Backspace"}:
                    if key in keydown_events: del keydown_events[key]
                    continue

                dwell_time = 0.0
                keydown_time = keydown_events.get(key)
                if keydown_time:
                    dwell_time = timestamp - keydown_time
                    del keydown_events[key]
                
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
                if previous_keydown_time and keydown_time:
                    dd_lat = keydown_time - previous_keydown_time
                    dd_lat = max(0.0, min(dd_lat, MAX_LATENCY))
                    time_vector.append(np.log1p(dd_lat) / TIME_SCALE)
                else:
                    time_vector.append(0.0)

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
    - Uses 32 units to balance capacity and overfitting.
    - Uses Dropout (0.4) for regularization.
    - Uses L2 Regularization.
    """
    # 1. Time Features Input (Continuous)
    input_time = layers.Input(shape=(TARGET_LENGTH, 5), name="input_time")
    masked_time = layers.Masking(mask_value=0.0)(input_time)
    
    # Bi-LSTM for time
    bilstm_time = layers.Bidirectional(
        layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))
    )(masked_time)
    bilstm_time = layers.BatchNormalization()(bilstm_time)
    bilstm_time = layers.Dropout(0.4)(bilstm_time)

    # 2. Key Sequence Input (Categorical)
    input_key = layers.Input(shape=(TARGET_LENGTH,), dtype="int32", name="input_key")
    
    # Embedding for key IDs
    embedding = layers.Embedding(input_dim=1000, output_dim=32, mask_zero=True)(input_key)
    
    # LSTM for keys
    lstm_key = layers.LSTM(32, return_sequences=False, kernel_regularizer=regularizers.l2(0.001))(embedding)
    lstm_key = layers.BatchNormalization()(lstm_key)
    lstm_key = layers.Dropout(0.4)(lstm_key)

    # 3. Merge & Classify
    merged = layers.Concatenate()([bilstm_time, lstm_key])
    
    dense = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(0.001))(merged)
    dense = layers.Dropout(0.4)(dense)
    
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
        
        json_files = list(DATA_DIR.rglob("*.json"))
        logging.info(f"Found {len(json_files)} data files.")
        
        all_X_time = []
        all_X_key = []
        all_y = []
        
        label_map = {}
        current_label = 0
        
        # 1. Load Real Data
        for f in json_files:
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

        real_sample_count = len(all_X_time)
        if real_sample_count == 0:
            logging.error("No real data found for training.")
            return

        logging.info(f"Total REAL samples: {real_sample_count}")

        # 2. Dynamic Imposter Data Generation
        # Strategy: Strict 1:1 Ratio. Generate exactly as many imposters as real samples.
        # This prevents the "Imposter" class from dominating class weights or the loss function.
        n_imposter_samples = real_sample_count
        
        # Safety floor: Ensure at least some imposters if real data is tiny (for testing)
        if n_imposter_samples < 50:
            n_imposter_samples = 50

        logging.info(f"Generating {n_imposter_samples} synthetic imposter samples (Balanced with Real Data)...")
        X_time_imp, X_key_imp = generate_imposter_data(n_samples=n_imposter_samples)
        
        imposter_label = current_label
        label_map["_IMPOSTER_"] = imposter_label
        
        for i in range(len(X_time_imp)):
            all_X_time.append(X_time_imp[i])
            all_X_key.append(X_key_imp[i])
            all_y.append(imposter_label)
            
        logging.info(f"Final Dataset Size: {len(all_X_time)} samples. Classes: {len(label_map)}")

        if len(label_map) < 2:
            logging.warning("Not enough classes to train (Need at least 1 user + Imposters).")
            return

        X_time_train = np.array(all_X_time, dtype=np.float32)
        X_key_train = np.array(all_X_key, dtype=np.int32)
        y_train = np.array(all_y, dtype=np.int32)

        # Stratified Shuffle Split
        from sklearn.model_selection import StratifiedShuffleSplit
        # Use StratifiedShuffleSplit for better handling of small/imbalanced sets
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        
        for train_index, val_index in sss.split(X_time_train, y_train):
            X_time_tr, X_time_val = X_time_train[train_index], X_time_train[val_index]
            X_key_tr, X_key_val = X_key_train[train_index], X_key_train[val_index]
            y_tr, y_val = y_train[train_index], y_train[val_index]

        from sklearn.utils import class_weight
        classes = np.unique(y_tr)
        weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_tr)
        class_weight_dict = dict(zip(classes, weights))
        
        logging.info(f"Class Weights: {class_weight_dict}")

        # 3. Build & Train
        model = build_model(num_classes=len(label_map))
        
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
            callbacks.ModelCheckpoint(str(MODEL_PATH), save_best_only=True, monitor='val_loss')
        ]

        history = model.fit(
            [X_time_tr, X_key_tr],
            y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=([X_time_val, X_key_val], y_val),
            class_weight=class_weight_dict,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # 4. Save Final State
        # ModelCheckpoint handles the best model save, but we ensure label map is saved.
        # Check if ModelCheckpoint saved anything (it should have), if not, save current.
        if not MODEL_PATH.exists():
             model.save(MODEL_PATH)

        with open(MODEL_PATH.parent / "label_map.json", "w", encoding='utf-8') as f:
            json.dump(label_map, f)
            
        logging.info("Training completed. Label map saved.")
        
        reload_model()

    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
    finally:
        training_lock.release()
