import numpy as np

def get_sample_signals():
    t = np.linspace(0, 10, 100)  # uniform 100 points
    return {
        'Exponential Decay ': np.exp(-t),
        'Sine Wave ': np.sin(2 * np.pi * t),
        'Cosine Wave ': np.cos(2 * np.pi * t),
        'Unit Step Signal ': np.where(t >= 0, 1, 0),
        'Unit Impulse Signal': np.array([1] + [0]*99),
        'Ramp Signal ': t,
    }
