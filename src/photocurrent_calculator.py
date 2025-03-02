import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting
import matplotlib.pyplot as plt
import numpy as np
from tkinter import filedialog
import tkinter as tk
import pandas as pd

def load_spectral_data():
    """
    Load spectral data from a CSV or text file.
    CSV files should have headers and use standard comma separation for Rainbow CSV compatibility.
    Expected format:
    wavelength_nm,intensity_normalized
    190.0,0.0
    190.5,100.0
    ...
    """
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
        # Try reading as CSV first with explicit column names for Rainbow CSV
        data = pd.read_csv(file_path)
        # Look for standardized column names
        wavelength_col = next(col for col in data.columns if 'wavelength' in col.lower())
        intensity_col = next(col for col in data.columns if 'intensity' in col.lower())
        wavelengths = data[wavelength_col].values
        intensities = data[intensity_col].values
    except:
        try:
            # Fallback: Try reading as space/tab-delimited text
            data = np.loadtxt(file_path)
            wavelengths = data[:, 0]
            intensities = data[:, 1]
            
            # Convert to CSV format for future use with Rainbow CSV
            output_df = pd.DataFrame({
                'wavelength_nm': wavelengths,
                'intensity_normalized': intensities
            })
            csv_path = file_path.rsplit('.', 1)[0] + '.csv'
            output_df.to_csv(csv_path, 
                           index=False,
                           float_format='%.3f',
                           encoding='utf-8')
            print(f"Converted and saved to CSV: {csv_path}")
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