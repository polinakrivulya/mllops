import numpy as np
from typing import List, Dict

def logits_to_label(logits: np.ndarray, id2label: Dict[int, str]) -> List[str]:
    probs = softmax(logits, axis=-1)
    ids = probs.argmax(axis=-1)
    return [id2label[int(i)] for i in ids]

def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
