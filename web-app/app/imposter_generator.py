import numpy as np
import random
from app.services import get_stable_key_id, VOCAB_LIST

def generate_imposter_data(n_samples=100, target_length=50):
    time_sequences = []
    key_sequences = []

    TIME_SCALE = 7.0
    
    # Use the shared VOCAB_LIST to ensure imposters use all valid characters (including Turkish)
    vocab_str = [k for k in VOCAB_LIST if k not in ["<PAD>", "<UNK>", "", None]]
    vocab_pool = vocab_str

    
    # ---------------------------------------------------------
    # DIVERSE IMPOSTER PROFILES
    # We generate a mix of different typing styles to force the model
    # to learn a tighter boundary around the real user.
    # ---------------------------------------------------------
    
    samples_per_profile = n_samples // 4
    remainder = n_samples % 4
    
    profiles = [
        {"name": "fast",    "dwell_mean": 60,  "dwell_std": 10, "flight_mean": 50,  "flight_std": 20, "variance_mult": 1.0},
        {"name": "slow",    "dwell_mean": 200, "dwell_std": 30, "flight_mean": 250, "flight_std": 50, "variance_mult": 1.0},
        {"name": "erratic", "dwell_mean": 120, "dwell_std": 50, "flight_mean": 150, "flight_std": 80, "variance_mult": 2.5}, # High variance
        {"name": "average", "dwell_mean": 100, "dwell_std": 20, "flight_mean": 150, "flight_std": 40, "variance_mult": 1.0}, # Original
    ]

    for i, profile in enumerate(profiles):
        # Calculate how many samples for this profile
        count = samples_per_profile
        if i == len(profiles) - 1:
            count += remainder
            
        if count <= 0: continue
            
        dwell_mean = profile["dwell_mean"]
        dwell_std = profile["dwell_std"]
        flight_mean = profile["flight_mean"]
        flight_std = profile["flight_std"]
        var_mult = profile["variance_mult"]

        for _ in range(count):
            seq_time = []
            seq_key = []

            # Per-session base variations (some sessions are naturally faster/slower than the profile mean)
            session_dwell_base = np.random.normal(dwell_mean, dwell_std * 0.5) 
            session_flight_base = np.random.normal(flight_mean, flight_std * 0.5)
            
            session_dwell_base = max(20, session_dwell_base)
            session_flight_base = max(5, session_flight_base)

            for _ in range(target_length):
                # 1. Dwell Time
                dwell = np.random.normal(session_dwell_base, dwell_std * var_mult)
                dwell = max(10, min(dwell, 800)) # Loose clamp
                
                # 2. Flight Time
                flight = np.random.normal(session_flight_base, flight_std * var_mult)
                flight = max(5, min(flight, 2000))
                
                # 3. Inter-Key Delay
                delay = dwell + flight
                if delay < 10: delay = 10
                
                # Normalize (Log1p / Scale)
                norm_dwell = np.log1p(dwell) / TIME_SCALE
                norm_flight = np.log1p(flight) / TIME_SCALE
                norm_delay = np.log1p(delay) / TIME_SCALE
                
                # 4. Pressure (Simulated)
                pressure = min(dwell / 200.0, 1.0)
                
                # 5. Down-Down Latency (Approximation for synthetic)
                dd_lat_val = delay + np.random.normal(0, 10) 
                dd_lat_val = max(10, dd_lat_val)
                norm_dd_lat = np.log1p(dd_lat_val) / TIME_SCALE

                # Construct Feature Vector
                time_vector = [
                    float(norm_dwell), 
                    float(norm_flight), 
                    float(norm_delay), 
                    float(pressure), 
                    float(norm_dd_lat)
                ]
                
                # Random Key
                random_key = random.choice(vocab_pool)
                key_id = get_stable_key_id(random_key)

                seq_time.append(time_vector)
                seq_key.append(key_id)

            time_sequences.append(seq_time)
            key_sequences.append(seq_key)

    # Shuffle the resulting lists to mix profiles in the batch
    combined = list(zip(time_sequences, key_sequences))
    random.shuffle(combined)
    time_sequences, key_sequences = zip(*combined)

    return np.array(time_sequences, dtype=np.float32), np.array(key_sequences, dtype=np.int32)
