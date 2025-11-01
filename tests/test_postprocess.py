import numpy as np

from src.utils.postprocess import logits_to_label, softmax

def test_softmax_properties():
    x = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    p = softmax(x, axis=-1)

    # Строки — распределения вероятностей
    assert p.shape == x.shape
    np.testing.assert_allclose(p.sum(axis=-1), np.ones(x.shape[0]), rtol=1e-6, atol=1e-6)
    assert np.all(p >= 0.0)

    # Инвариантность к добавлению константы
    x_shift = x + 10.0
    p_shift = softmax(x_shift, axis=-1)
    np.testing.assert_allclose(p, p_shift, rtol=1e-6, atol=1e-6)


def test_logits_to_label_mapping():
    logits = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [5.0, 1.0, 0.0, -1.0],
        [-2.0, 7.0, 3.0, 1.0],
    ])
    id2label = {0: "a", 1: "b", 2: "c", 3: "d"}
    labels = logits_to_label(logits, id2label)
    assert labels == ["d", "a", "b"]
