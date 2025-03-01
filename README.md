# UVC Measurements

A Python tool for analyzing UV-C spectral data and calculating photocurrent.

## Project Description
This tool processes spectral data from UV-C sources, filters, and detectors to calculate photocurrent. It supports:
- Loading spectral data from CSV files
- Interactive plotting of spectral data
- Multiplication of spectral datasets
- Integration for photocurrent calculation

## Installation

```bash
git clone https://github.com/tdunbar12/UVC_Measurements.git
cd UVC_Measurements
```

## Dependencies
- Python 3.8+
- pandas
- numpy
- matplotlib

## Project Structure
```
UVC_Measurements/
├── src/
│   ├── __init__.py
│   ├── photocurrent_calculator.py
│   └── dataset_file_creator.py
├── data/
│   ├── input/
│   └── output/
└── tests/
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
```
3. Select your input files when prompted
4. View the results in the interactive plot window

## Data Format
Input CSV files should follow this format:
```csv
wavelength_nm,intensity_normalized
190.000,0.000
190.500,100.000
...
```

## License
MIT License
