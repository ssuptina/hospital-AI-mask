import json
import os
import numpy as np

def compute_class_weights_from_flow(flow):
    counts = np.zeros(len(flow.class_indices))
    seen = 0
    for _, y in flow:
        counts += y.sum(axis=0)
        seen += y.shape[0]
        if seen >= flow.samples:
            break
    total = counts.sum()
    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    class_weights = {i: float(total / (len(counts) * c)) for i, c in enumerate(counts)}
    return class_weights

def save_class_index_map(flow, save_path="../saved_model/class_indices.json"):
    inv = {int(v): k for k, v in flow.class_indices.items()}
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(inv, f, indent=2)

def load_class_index_map(path="../saved_model/class_indices.json"):
    with open(path, "r") as f:
        return json.load(f)
