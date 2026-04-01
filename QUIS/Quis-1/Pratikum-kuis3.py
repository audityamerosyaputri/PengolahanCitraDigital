import numpy as np
import matplotlib.pyplot as plt

def simulate_digitization(analog_function, sampling_rate, quantization_levels):
    """
    analog_function: fungsi kontinu (misal np.sin)
    sampling_rate: jumlah sampel
    quantization_levels: level kuantisasi
    """
    
    # Sinyal kontinu (analog)
    x_cont = np.linspace(0, 2*np.pi, 1000)
    y_cont = analog_function(x_cont)
    
    # 1. Sampling
    x_sample = np.linspace(0, 2*np.pi, sampling_rate)
    y_sample = analog_function(x_sample)
    
    # 2. Quantization
    y_min = np.min(y_sample)
    y_max = np.max(y_sample)
    
    y_norm = (y_sample - y_min) / (y_max - y_min)
    y_quant = np.round(y_norm * (quantization_levels - 1))
    y_quant = y_quant / (quantization_levels - 1)
    y_quant = y_quant * (y_max - y_min) + y_min
    
    # 3. Visualisasi
    plt.figure(figsize=(10,6))
    plt.plot(x_cont, y_cont, label="Analog Signal", linewidth=2)
    plt.stem(x_sample, y_quant, linefmt='r-', markerfmt='ro',
             basefmt=" ", label="Digital Signal")
    
    plt.title("Simulasi Sampling dan Quantization")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    simulate_digitization(np.sin, sampling_rate=20, quantization_levels=8)