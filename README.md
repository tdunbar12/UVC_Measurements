# UVC Measurements Tools

A collection of tools for analyzing UVC measurements and processing spectral data.

## Tools Included

### 1. Interactive Plot Digitizer
A graphical tool for extracting data points from spectral plots. Features include:

- Multiple interpolation methods:
  - Cubic Spline: Best for smooth, continuous curves
  - LOESS: Local regression for noisy data
  - Local Window: Better for curves with sharp transitions
  - Linear: Simple point-to-point interpolation
- Support for both linear and log-linear plots
- Multi-step calibration process:
  - X-axis (wavelength) calibration
  - Y-axis (intensity) calibration
  - Data point collection
- Interactive features:
  - Real-time visualization of selected points
  - Undo functionality for point selection
  - Verification plotting with original and interpolated data
- Export to standardized 1nm resolution CSV format

#### Usage:
```bash
python src/Interactive_Plot_Digitizer.py
```

Follow the on-screen instructions to:
1. Load and size the plot image
2. Select plot type (linear or log-linear)
3. Calibrate axes with at least 3 points each
4. Collect data points along the curve
5. Choose interpolation method
6. Verify and save results

### 2. Photocurrent Calculator
This tool processes spectral data from UV-C sources, filters, and detectors to calculate photocurrent. It supports:
- Loading spectral data from CSV files
- Interactive plotting of spectral data
- Multiplication of spectral datasets
- Integration for photocurrent calculation

## Installation

```bash
git clone https://github.com/tdunbar12/UVC_Measurements.git
cd UVC_Measurements
conda env create -f environment.yml
conda activate uvc_measure
```

## Dependencies
- Python 3.11+
- numpy>=1.21.0
- pandas>=1.3.0
- matplotlib>=3.4.0
- opencv-python>=4.5.0
- scipy>=1.7.0
- statsmodels>=0.13.0
- tk>=0.1.0

## Project Structure
```
UVC_Measurements/
├── src/
│   ├── __init__.py
│   ├── photocurrent_calculator.py
│   ├── dataset_file_creator.py
│   └── Interactive_Plot_Digitizer.py
├── data/
│   ├── input/
│   │   ├── D2-Lamp.csv
│   │   └── LPMV-Lamp.csv
│   └── output/
├── requirements.txt
├── environment.yml
└── README.md
```

## Development Setup

### VS Code Extensions
For the best development experience, install these VS Code extensions:
1. Rainbow CSV
   - Open VS Code
   - Press `Ctrl+P`
   - Paste: `ext install mechatroner.rainbow-csv`
   - Or search "Rainbow CSV" in the Extensions marketplace

The Rainbow CSV extension provides:
- Colored column visualization
- CSV data validation
- SQL-like queries for CSV data
- Column name tooltips

## Usage
1. Place your spectral data files in the `data/input` directory
2. Run the photocurrent calculator:
```bash
python src/photocurrent_calculator.py
3. Select your input files when prompted
4. View the results in the interactive plot window

## Data Format
Input CSV files should follow this format:
```csv
wavelength_nm,intensity
190.000,0.000
191.000,0.100
...
```

## Interactive Plot Digitizer Features
- Multiple interpolation methods for different curve types
- Support for log-linear plots common in spectral data
- Real-time visualization of selected points
- Verification plotting showing both original and processed data
- Undo functionality for point selection
- Export to standardized 1nm resolution CSV format

### Known Issues
- Edge cases in interpolation methods
- Scale calibration requires careful point selection

### Next Steps
- [ ] Add data smoothing options
- [ ] Implement better error handling
- [ ] Add validation metrics
- [ ] Support for multiple plot types

## License
MIT License
````
