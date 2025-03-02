"""
Plot Image to CSV Converter
--------------------------
Extracts data points from graph images and saves to CSV format.
Currently supports:
- Linear and log-linear plots
- Normalized output data
- Dual save locations (Linux project dir and Windows access)

Known issues:
- Manual calibration needed for axes
- Edge detection needs improvement
- Scale accuracy needs verification
"""

import cv2
import numpy as np
import pandas as pd
from tkinter import Tk, filedialog, messagebox
import os

def select_image_file():
    """Open file dialog to select an image file"""
    root = Tk()
    root.withdraw()  # Hide the main window
    
    # Set initial directory for Windows path - update path to match your structure
    initial_dir = "/mnt/c/Users/TomDunbar/Documents/airPhyzx/Plots"
    
    if not os.path.exists(initial_dir):
        print(f"Warning: Directory not found: {initial_dir}")
        print("Falling back to home directory...")
        initial_dir = os.path.expanduser("~")
    else:
        print(f"Found directory: {initial_dir}")
    
    file_path = filedialog.askopenfilename(
        initialdir=initial_dir,
        title="Select Graph Image",
        filetypes=(
            ("JPEG files", "*.jpg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        )
    )
    
    if file_path and file_path.startswith('/mnt/c/'):
        print(f"Selected file: {file_path}")
        return file_path
    return None

def get_plot_type():
    """Get plot type from user"""
    root = Tk()
    root.withdraw()
    plot_type = messagebox.askquestion("Plot Type", 
                                     "Is this a log-linear plot?\n\n" +
                                     "Yes = Log-Linear\n" +
                                     "No = Linear-Linear")
    return plot_type == 'yes'

def ensure_output_directory():
    """Create data/input directory if it doesn't exist"""
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'input')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def extract_graph_data(image_path, is_log_plot=False):
    """Extract data points from the graph image"""
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract points from the largest contour (assumed to be the graph line)
    if contours:
        graph_contour = max(contours, key=cv2.contourArea)
        points = graph_contour.reshape(-1, 2)
        
        # Sort points by x-coordinate
        points = points[points[:, 0].argsort()]
        
        # Convert pixel coordinates to data values
        # Note: You'll need to calibrate these values based on your graph's scale
        x_min, x_max = 190, 250  # wavelength range in nm
        y_min, y_max = 0, 100    # intensity range
        
        height, width = img.shape[:2]
        x_scale = (x_max - x_min) / width
        y_scale = (y_max - y_min) / height
        
        data = []
        for x, y in points:
            x_val = x_min + x * x_scale
            if is_log_plot:
                # Convert from pixel position to log scale
                log_min, log_max = np.log10(0.1), np.log10(100)  # Adjust range as needed
                log_scale = (log_max - log_min) / height
                y_val = 10 ** (log_max - y * log_scale)
            else:
                y_val = y_max - y * y_scale
            data.append([x_val, y_val])
        
        return data
    
    return None

def get_windows_output_dir():
    """Get Windows output directory path"""
    return "/mnt/c/Users/TomDunbar/Documents/airPhyzx/Plots/extracted_data"

def save_to_csv(data, output_path, save_to_windows=True):
    """Save extracted data to CSV file with normalized values"""
    df = pd.DataFrame(data, columns=['wavelength_nm', 'intensity'])
    
    # Normalize intensity values to maximum
    max_intensity = df['intensity'].max()
    df['intensity_normalized'] = df['intensity'] / max_intensity * 100
    
    # Reorder columns and save
    df = df[['wavelength_nm', 'intensity_normalized']]
    
    # Save to project directory
    df.to_csv(output_path, index=False)
    print(f"Data saved to project directory: {output_path}")
    
    # Save to Windows directory if requested
    if save_to_windows:
        windows_dir = get_windows_output_dir()
        os.makedirs(windows_dir, exist_ok=True)
        windows_path = os.path.join(windows_dir, os.path.basename(output_path))
        df.to_csv(windows_path, index=False)
        print(f"Data saved to Windows directory: {windows_path}")
    
    print(f"Data normalized to maximum value: {max_intensity:.2f}")

def main():
    try:
        # Select input image file
        image_path = select_image_file()
        if not image_path:
            print("No file selected. Exiting...")
            return
        
        # Get plot type
        is_log_plot = get_plot_type()
        print(f"Processing {'log-linear' if is_log_plot else 'linear-linear'} plot")
        
        # Extract data from graph
        data = extract_graph_data(image_path, is_log_plot)
        
        # Generate output path in data/input directory
        output_dir = ensure_output_directory()
        output_name = os.path.splitext(os.path.basename(image_path))[0] + "_data.csv"
        output_path = os.path.join(output_dir, output_name)
        
        # Save to CSV (both locations)
        save_to_csv(data, output_path, save_to_windows=True)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()