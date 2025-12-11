import numpy as np
import random
from app.services import get_stable_key_id

def generate_imposter_data(n_samples=100, target_length=50):
    time_sequences = []
    key_sequences = []

    TIME_SCALE = 7.0
    
    # Common keys to sample from (lowercase + space + common special)
    common_chars = list("abcdefghijklmnopqrstuvwxyz 1234567890")
    special_keys = ["Backspace", "Enter", "Shift", "Period", "Comma"]
    vocab_pool = common_chars + special_keys

    for _ in range(n_samples):
        seq_time = []
        seq_key = []

        base_dwell = np.random.normal(100, 20)
        base_flight = np.random.normal(150, 40)
        
        base_dwell = max(30, base_dwell)
        base_flight = max(10, base_flight)

        for _ in range(target_length):
            # 1. Dwell Time
            dwell = np.random.normal(base_dwell, 15)
            dwell = max(10, min(dwell, 500)) # Clamp
            
            # 2. Flight Time
            flight = np.random.normal(base_flight, 30)
            flight = max(10, min(flight, 1000))
            
            # 3. Inter-Key Delay (roughly Dwell + Flight)
            delay = dwell + flight
            if delay < 0: delay = 10
            
            # Normalize (Log1p / Scale)
            norm_dwell = np.log1p(dwell) / TIME_SCALE
            norm_flight = np.log1p(flight) / TIME_SCALE
            norm_delay = np.log1p(delay) / TIME_SCALE
            
            # 4. Pressure (Simulated)
            pressure = min(dwell / 200.0, 1.0)
            
            # 5. Down-Down Latency (Same as delay approx)
            dd_lat = np.log1p(delay) / TIME_SCALE # approximation

            # Construct Feature Vector
            time_vector = [
                float(norm_dwell), 
                float(norm_flight), 
                float(norm_delay), 
                float(pressure), 
                float(dd_lat)
            ]
            
            # Random Key
            random_key = random.choice(vocab_pool)
            key_id = get_stable_key_id(random_key)

            seq_time.append(time_vector)
            seq_key.append(key_id)

        time_sequences.append(seq_time)
        key_sequences.append(seq_key)

    return np.array(time_sequences, dtype=np.float32), np.array(key_sequences, dtype=np.int32)
