import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter as tk
import pandas as pd

def load_spectral_data():
    """Load spectral data from a CSV or text file."""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_path = filedialog.askopenfilename(
        title="Select spectral data file",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    
    if not file_path:
        return None, None
    
    try:
        # Try reading as CSV first
        data = pd.read_csv(file_path)
        # Assume first column is wavelength, second is intensity
        wavelengths = data.iloc[:, 0].values
        intensities = data.iloc[:, 1].values
    except:
        try:
            # Try reading as space/tab-delimited text
            data = np.loadtxt(file_path)
            wavelengths = data[:, 0]
            intensities = data[:, 1]
        except:
            print(f"Error: Could not read file {file_path}")
            return None, None
    
    return wavelengths, intensities

def plot_spectrum(wavelengths, intensities, title="Spectral Data"):
    """Plot spectral data."""
    plt.figure(figsize=(8, 5))
    plt.semilogy(wavelengths, intensities, 'b.-', label='Spectral Data')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Intensity (log scale)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

def main():
    """Main function to load and plot spectral data."""
    wavelengths, intensities = load_spectral_data()
    
    if wavelengths is not None and intensities is not None:
        plot_spectrum(wavelengths, intensities)

if __name__ == "__main__":
    main()