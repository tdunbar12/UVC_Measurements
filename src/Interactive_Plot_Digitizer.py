import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox, StringVar as tk_StringVar
import tkinter as tk
from tkinter import ttk
import os
import scipy.interpolate as interp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
from statsmodels.nonparametric.smoothers_lowess import lowess


class PlotDigitizer:
    def __init__(self):
        self.image = None
        # Set backend for Qt/OpenCV
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
        self.window_name = "Plot Digitizer"
        self.x_calibration_points = []  # [(pixel_x, wavelength), ...]
        self.y_calibration_points = []  # [(pixel_y, intensity), ...]
        self.data_points = []           # [(pixel_x, pixel_y), ...]
        self.current_mode = "x_cal"     # Modes: x_cal, y_cal, data
        self.is_log_plot = False
        self.min_points_for_calibration = 3  # Minimum points needed for good calibration
        self.calibration_complete = False
        self.display_backup = None  # For restoring image before last point
        self.window_size = None  # Store window size after first resize
        self.original_click_positions = {
            'x_cal': [],  # [(x, y), ...] store original click positions
            'y_cal': [],
            'data': []
        }
        self.interpolated_points = []  # [(wavelength, intensity), ...]
        
    def load_image(self, image_path):
        """Load and display the image"""
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Could not load image")
        self.display_image = self.image.copy()
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_mode == "x_cal":
                self.handle_x_calibration(x, y)
            elif self.current_mode == "y_cal":
                self.handle_y_calibration(x, y)
            elif self.current_mode == "data":
                self.handle_data_point(x, y)
                
    def handle_x_calibration(self, x, y):
        """Handle X-axis calibration point selection"""
        if self.display_backup is None:
            self.display_backup = self.display_image.copy()
            
        # Store original click position
        self.original_click_positions['x_cal'].append((x, y))
        value = float(input(f"Enter wavelength value for point ({x}, {y}): "))
        self.x_calibration_points.append((x, value))
        cv2.circle(self.display_image, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow(self.window_name, self.display_image)
        
        # Store window size after first point
        if self.window_size is None:
            self.window_size = cv2.getWindowImageRect(self.window_name)
            print(f"Window size fixed at: {self.window_size}")
        
    def handle_y_calibration(self, x, y):
        """Handle Y-axis calibration point selection"""
        if self.is_log_plot:
            # For log plots, user enters the actual intensity value (not the log)
            value_str = input(f"Enter intensity value for point ({x}, {y}): ")
            try:
                value = float(value_str)
                # Store the actual value directly
                self.y_calibration_points.append((y, value))
                print(f"Stored intensity value: {value} (appears at log10 position: {np.log10(value)})")
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                return
        else:
            # For linear plots, just store the value directly
            value = float(input(f"Enter intensity value for point ({x}, {y}): "))
            self.y_calibration_points.append((y, value))
        
        cv2.circle(self.display_image, (x, y), 3, (255, 0, 0), -1)
        cv2.imshow(self.window_name, self.display_image)
        
    def start_digitizing(self):
        """Main control flow for digitizing process"""
        # Get plot type
        self.is_log_plot = messagebox.askyesno(
            "Plot Type",
            "Is this a log-linear plot?\n\nYes = Log-Linear\nNo = Linear-Linear"
        )
        
        # Setup window and callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("X-Axis Calibration Mode:")
        print("Click points on X-axis and enter wavelength values")
        print("Press 'n' when done with X calibration")
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Handle mode switching
            if key == ord('n'):  # Next mode
                if self.current_mode == "x_cal":
                    if len(self.x_calibration_points) >= self.min_points_for_calibration:
                        self.current_mode = "y_cal"
                        print("\nY-Axis Calibration Mode:")
                        print("Click points on Y-axis and enter intensity values")
                        print("Press 'n' when done with Y calibration")
                    else:
                        print(f"Need at least {self.min_points_for_calibration} X calibration points")
                
                elif self.current_mode == "y_cal":
                    if len(self.y_calibration_points) >= self.min_points_for_calibration:
                        self.current_mode = "data"
                        self.calibration_complete = True
                        print("\nData Collection Mode:")
                        print("Click points along the curve")
                        print("Press 'f' when finished")
                    else:
                        print(f"Need at least {self.min_points_for_calibration} Y calibration points")
            
            elif key == ord('f'):  # Finish
                if self.current_mode == "data" and len(self.data_points) > 0:
                    break
            
            elif key == ord('q'):  # Quit
                return False
        
        cv2.destroyAllWindows()
        return True

    def handle_data_point(self, x, y):
        """Handle data point selection"""
        if not self.calibration_complete:
            print("Complete calibration first")
            return
        
        self.data_points.append((x, y))
        # Draw point on image
        cv2.circle(self.display_image, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow(self.window_name, self.display_image)

    def test_image_display(self):
        """Test image loading and display"""
        try:
            # Get test image path
            root = Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                initialdir="/mnt/c/Users/TomDunbar/Documents/airPhyzx/Plots",
                title="Select test image",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            
            if not image_path:
                print("No file selected")
                return False
                
            # Load and display image
            self.load_image(image_path)
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Add WINDOW_NORMAL flag
            cv2.imshow(self.window_name, self.display_image)
            print("\nWindow Interaction Instructions:")
            print("1. Resize the plot window to desired size")
            print("2. Click INSIDE the plot window to activate it")
            print("3. Press any key WHILE the plot window is active to continue")
            print("\nWindow Setup Instructions:")
            print("1. RESIZE the plot window to your preferred working size")
            print("2. CLICK INSIDE the plot window to make it active")
            print("3. While plot window is active, press ANY KEY to continue")
            print("Note: Window size will be fixed after this step")
            cv2.waitKey(0)
            
            # Store initial window size
            self.window_size = cv2.getWindowImageRect(self.window_name)
            print(f"Window size fixed at: {self.window_size}")
            
            # Fix window size for future displays
            cv2.resizeWindow(self.window_name, self.window_size[2], self.window_size[3])
            return True
        except Exception as e:
            print(f"Error during test: {str(e)}")
            return False

    def test_plot_type_selection(self):
        """Test plot type selection dialog"""
        root = Tk()
        root.withdraw()
        
        # Show plot type selection dialog
        self.is_log_plot = messagebox.askyesno(
            "Plot Type Selection Test",
            "Is this a log-linear plot?\n\nYes = Log-Linear\nNo = Linear-Linear"
        )
        
        print(f"\nSelected plot type: {'Log-Linear' if self.is_log_plot else 'Linear-Linear'}")
        return True

    def test_x_calibration(self):
        """Test X-axis calibration point selection"""
        if self.image is None:
            print("Load image first")
            return False
        
        print("\nX-Axis Calibration Test")
        print("------------------------")
        print("Click points on X-axis and enter wavelength values")
        print("Press 'n' when done, 'q' to quit")
        print("Press 'u' to undo last point")
        print("\nX-Axis Calibration Instructions:")
        print("1. Click INSIDE plot window to activate it")
        print("2. Left-click points along X-axis")
        print("3. Enter wavelength value in terminal when prompted")
        print("4. Press 'u' WHILE plot window is active to undo last point")
        print("5. Press 'n' to finish or 'q' to quit (while plot window is active)")
        print("\nX-Axis Calibration Instructions:")
        print("1. CLICK INSIDE plot window to make it active")
        print("2. Left-click points along X-axis (need at least 3 points)")
        print("3. Enter wavelength value in TERMINAL when prompted")
        print("4. After typing value, CLICK INSIDE plot window again")
        print("5. While plot window is active:")
        print("   - Press 'u' to undo last point")
        print("   - Press 'n' to finish")
        print("   - Press 'q' to quit")
        
        # Setup window for calibration
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.current_mode = "x_cal"  # Set mode before starting
        
        # Make backup of clean image
        self.display_backup = self.display_image.copy()
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):
                self.undo_last_point()
                if self.x_calibration_points:
                    print("\nLast X calibration point removed")
                    print(f"Remaining points: {len(self.x_calibration_points)}")
            
            elif key == ord('n'):
                if len(self.x_calibration_points) >= self.min_points_for_calibration:
                    print("\nX calibration points:")
                    for x, value in self.x_calibration_points:
                        print(f"Pixel: {x}, Wavelength: {value}nm")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print(f"Need at least {self.min_points_for_calibration} points")
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

    def test_y_calibration(self):
        """Test Y-axis calibration point selection"""
        if self.image is None:
            print("Load image first")
            return False
        
        print("\nY-Axis Calibration Test")
        print("------------------------")
        print(f"Click points on Y-axis and enter {'log ' if self.is_log_plot else ''}intensity values")
        print("Press 'n' when done, 'q' to quit")
        print("Press 'u' to undo last point")
        print("\nY-Axis Calibration Instructions:")
        print("1. CLICK INSIDE plot window to make it active")
        print(f"2. Left-click points along Y-axis ({self.min_points_for_calibration} points minimum)")
        print("3. Enter intensity value in TERMINAL when prompted")
        print("   (Use log values if log-linear plot selected)")
        print("4. After typing value, CLICK INSIDE plot window again")
        print("5. While plot window is active:")
        print("   - Press 'u' to undo last point")
        print("   - Press 'n' to finish")
        print("   - Press 'q' to quit")
        
        # Setup window for calibration
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.current_mode = "y_cal"
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):
                self.undo_last_point()
                if self.y_calibration_points:
                    print("\nLast Y calibration point removed")
                    print(f"Remaining points: {len(self.y_calibration_points)}")
            
            if key == ord('n'):
                if len(self.y_calibration_points) >= self.min_points_for_calibration:
                    print("\nY calibration points:")
                    for y, value in self.y_calibration_points:
                        print(f"Pixel: {y}, {'Log ' if self.is_log_plot else ''}Intensity: {value}")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print(f"Need at least {self.min_points_for_calibration} points")
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

    def test_data_collection(self):
        """Test data point collection along the curve"""
        if self.image is None:
            print("Load image first")
            return False
            
        print("\nData Point Collection Test")
        print("-------------------------")
        print("Click points along the curve")
        print("Press 'n' when done, 'q' to quit")
        print("Press 'u' to undo last point")
        print("Points will be shown in red")
        print("\nData Collection Instructions:")
        print("1. CLICK INSIDE plot window to make it active")
        print("2. Left-click points along the curve")
        print("3. After each point, window must remain active")
        print("4. While plot window is active:")
        print("   - Press 'u' to undo last point")
        print("   - Press 'n' to finish")
        print("   - Press 'q' to quit")
        print("\nPoints collected will be shown in red")
        
        # Setup window for data collection
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        self.current_mode = "data"
        self.calibration_complete = True  # Allow data collection
        
        while True:
            cv2.imshow(self.window_name, self.display_image)
            key = cv2.waitKey(1) & 0xFF
            
            # Show point count every time a point is added
            if len(self.data_points) > 0:
                print(f"\rPoints collected: {len(self.data_points)}", end="")
            
            if key == ord('u'):
                self.undo_last_point()
                if self.data_points:
                    print("\nLast data point removed")
                    print(f"Remaining points: {len(self.data_points)}")
            
            if key == ord('n'):
                if len(self.data_points) > 0:
                    print("\n\nData points collected:")
                    for i, (x, y) in enumerate(self.data_points):
                        print(f"Point {i+1}: Pixel ({x}, {y})")
                    cv2.destroyAllWindows()
                    return True
                else:
                    print("\nNeed at least one data point")
            
            elif key == ord('q'):
                cv2.destroyAllWindows()
                return False

    def undo_last_point(self):
        """Remove the last point added in current mode"""
        if self.current_mode == "x_cal" and self.x_calibration_points:
            self.x_calibration_points.pop()
        elif self.current_mode == "y_cal" and self.y_calibration_points:
            self.y_calibration_points.pop()
        elif self.current_mode == "data" and self.data_points:
            self.data_points.pop()
            
        # Restore image and redraw remaining points
        if self.display_backup is not None:
            self.display_image = self.display_backup.copy()
            self.redraw_points()
            cv2.imshow(self.window_name, self.display_image)
    
    def redraw_points(self):
        """Redraw all points on the image"""
        # Draw X calibration points in green using original positions
        for i, (x, _) in enumerate(self.x_calibration_points):
            if i < len(self.original_click_positions['x_cal']):
                orig_x, orig_y = self.original_click_positions['x_cal'][i]
                cv2.circle(self.display_image, (orig_x, orig_y), 3, (0, 255, 0), -1)
        
        # Draw Y calibration points in blue using original positions
        for i, (y, _) in enumerate(self.y_calibration_points):
            if i < len(self.original_click_positions['y_cal']):
                orig_x, orig_y = self.original_click_positions['y_cal'][i]
                cv2.circle(self.display_image, (orig_x, orig_y), 3, (255, 0, 0), -1)
        
        # Draw data points in red - these keep their exact positions
        for x, y in self.data_points:
            cv2.circle(self.display_image, (x, y), 3, (0, 0, 255), -1)

    def interpolate_data(self, method='spline'):
        """Convert pixel coordinates to actual values and interpolate to 1nm spacing"""
        try:
            # Sort and filter points
            sorted_points = sorted(self.data_points, key=lambda p: p[0])
            x_data, y_data = zip(*sorted_points)
            
            # Sort calibration points
            sorted_x_cal = sorted(self.x_calibration_points, key=lambda p: p[0])
            sorted_y_cal = sorted(self.y_calibration_points, key=lambda p: p[0])
            x_pixels, x_values = zip(*sorted_x_cal)
            y_pixels, y_values = zip(*sorted_y_cal)
            
            # Convert coordinates using calibration
            x_interpolator = interp.interp1d(x_pixels, x_values, kind='linear')
            wavelengths = x_interpolator(x_data)
            y_interpolator = interp.interp1d(y_pixels, y_values, kind='linear')
            intensities = y_interpolator(y_data)
            
            # Handle multiple y-values at same x by averaging
            wavelength_dict = {}
            for wave, intensity in zip(wavelengths, intensities):
                wave = round(wave, 3)  # Round to reduce floating point issues
                if wave in wavelength_dict:
                    wavelength_dict[wave].append(intensity)
                else:
                    wavelength_dict[wave] = [intensity]
            
            # Create unique x,y pairs with averaged y values
            unique_wavelengths = []
            unique_intensities = []
            for wave in sorted(wavelength_dict.keys()):
                unique_wavelengths.append(wave)
                unique_intensities.append(np.mean(wavelength_dict[wave]))
                if len(wavelength_dict[wave]) > 1:
                    print(f"Averaged {len(wavelength_dict[wave])} values at {wave}nm")
            
            # Create output wavelength range
            start_nm = int(min(unique_wavelengths))
            end_nm = int(max(unique_wavelengths)) + 1
            wavelength_range = np.arange(start_nm, end_nm, 1.0)
            
            # Perform interpolation based on chosen method
            if method == 'loess':
                # LOESS smoothing implementation
                bandwidth = 0.3
                smoothed = lowess(unique_intensities, 
                                unique_wavelengths,
                                frac=bandwidth,
                                it=1,
                                return_sorted=True)
                
                loess_interp = interp.interp1d(smoothed[:, 0],
                                             smoothed[:, 1],
                                             bounds_error=False,
                                             fill_value='extrapolate')
                interpolated = loess_interp(wavelength_range)
                
            elif method == 'spline':
                interp_func = interp.CubicSpline(unique_wavelengths, unique_intensities)
                interpolated = interp_func(wavelength_range)
                
            elif method == 'linear':
                interp_func = interp.interp1d(unique_wavelengths, unique_intensities, 
                                            kind='linear',
                                            bounds_error=False,
                                            fill_value='extrapolate')
                interpolated = interp_func(wavelength_range)
                
            elif method == 'local':
                # Local window approach
                window_size = 5
                interpolated = []
                for wave in wavelength_range:
                    mask = np.abs(np.array(unique_wavelengths) - wave) <= window_size
                    if np.sum(mask) >= 2:
                        local_waves = np.array(unique_wavelengths)[mask]
                        local_ints = np.array(unique_intensities)[mask]
                        local_interp = interp.interp1d(local_waves, local_ints,
                                                     bounds_error=False,
                                                     fill_value='extrapolate')
                        interpolated.append(float(local_interp(wave)))
                    else:
                        idx = np.argmin(np.abs(np.array(unique_wavelengths) - wave))
                        interpolated.append(float(unique_intensities[idx]))
        
            # Store interpolated points for all methods
            self.interpolated_points = list(zip(wavelength_range, interpolated))
            print(f"Successfully interpolated {len(self.interpolated_points)} points using {method} method")
            return True
            
        except Exception as e:
            print(f"Interpolation error: {str(e)}")
            traceback.print_exc()
            return False

    def choose_interpolation_method(self, parent=None):
        """Let user choose interpolation method using a dropdown"""
        methods = {
            'Spline - Best for smooth curves': 'spline',
            'LOESS - Best for noisy data': 'loess',
            'Local Window - Better for sharp transitions': 'local',
            'Linear - Simple point-to-point': 'linear'
        }
        
        # Create dialog window
        dialog = tk.Toplevel(parent) if parent else Tk()
        dialog.title("Choose Interpolation Method")
        dialog.geometry("400x250")
        dialog.grab_set()  # Make dialog modal
        
        # Add description
        desc = ttk.Label(dialog, text="Select interpolation method:", 
                         wraplength=350, justify="left")
        desc.pack(pady=10)
        
        # Create combobox
        method_var = tk.StringVar(value='Spline - Best for smooth curves')  # Set initial value
        combo = ttk.Combobox(dialog, 
                            textvariable=method_var,
                            values=list(methods.keys()),
                            state="readonly",
                            width=40)
        combo.pack(pady=10)
        
        # Add method descriptions
        descriptions = {
            'spline': 'Cubic spline interpolation for smooth, continuous curves',
            'loess': 'Local regression smoothing, good for noisy data',
            'local': 'Moving window average for curves with sharp transitions',
            'linear': 'Simple linear interpolation between points'
        }
        
        desc_text = tk.Text(dialog, height=6, width=40, wrap=tk.WORD)
        desc_text.pack(pady=10)
        
        def update_description(*args):
            selected = method_var.get()
            if selected:  # Only update if a value is selected
                method_key = methods[selected]
                desc_text.delete('1.0', tk.END)
                desc_text.insert('1.0', descriptions[method_key])
        
        method_var.trace('w', update_description)
        update_description()  # Show initial description
        
        # Add OK button with proper cleanup
        selected_method = [None]
        def on_ok():
            selected_method[0] = methods[method_var.get()]
            dialog.quit()  # Use quit instead of destroy
            dialog.destroy()
        
        ttk.Button(dialog, text="OK", command=on_ok).pack(pady=10)
        
        # Add protocol for window close button
        dialog.protocol("WM_DELETE_WINDOW", on_ok)
        
        # Make dialog modal and wait for result
        dialog.transient(parent)
        dialog.grab_set()
        dialog.wait_window()
        
        return selected_method[0] or 'spline'

    def plot_verification(self):
        """Show original points and interpolated curve"""
        if not self.interpolated_points:
            print("Run interpolation first")
            return False
            
        try:
            # Close any existing plots and create new figure
            plt.close('all')  # Close all existing plots first
            
            # Create figure without interactive mode
            fig = plt.figure(figsize=(12, 6))
            gs = fig.add_gridspec(1, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            
            # Original image with points
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            ax1.imshow(rgb_image)
            x_data, y_data = zip(*self.data_points)
            ax1.plot(x_data, y_data, 'r.', label='Selected Points')
            ax1.set_title('Original Image with Points')
            
            try:
                # Convert points
                x_to_wave = interp.interp1d([x for x, _ in self.x_calibration_points],
                                          [v for _, v in self.x_calibration_points],
                                          bounds_error=False,
                                          fill_value='extrapolate')
                
                y_to_int = interp.interp1d([y for y, _ in self.y_calibration_points],
                                          [v for _, v in self.y_calibration_points],
                                          bounds_error=False,
                                          fill_value='extrapolate')
                
                x_orig = x_to_wave(x_data)
                y_orig = y_to_int(y_data)
                
                # Get interpolated data
                wavelengths, intensities = zip(*self.interpolated_points)
                wavelengths = np.array(wavelengths)
                intensities = np.array(intensities)
                
                # Plot data
                ax2.plot(wavelengths, intensities, 'b-', label='Interpolated (1nm)')
                ax2.plot(x_orig, y_orig, 'r.', markersize=8, label='Original Points')
                
                # Set axis labels
                ax2.set_xlabel('Wavelength (nm)')
                ax2.set_ylabel('Intensity')
                
                # Handle log scale
                if self.is_log_plot:
                    ax2.set_yscale('log')
                    ax2.grid(True, which='both')
                    ax2.grid(True, which='minor', alpha=0.2)
                else:
                    ax2.grid(True)
                
                ax2.legend()
                ax2.set_title('Interpolated Data')
                
                # Adjust layout
                plt.tight_layout()
                
                # Add close button with proper callback
                def on_close(event):
                    plt.close('all')
                    plt.ioff()  # Turn off interactive mode
                
                button_ax = plt.axes([0.45, 0.02, 0.1, 0.04])
                close_button = plt.Button(button_ax, 'Close')
                close_button.on_clicked(on_close)
                
                # Show plot with block=True and proper cleanup
                plt.show()
                return True
                
            except Exception as e:
                print(f"Error in data conversion: {str(e)}")
                traceback.print_exc()
                plt.close('all')
                return False
                
        except Exception as e:
            print(f"Plot verification error: {str(e)}")
            traceback.print_exc()
            plt.close('all')
            return False

    def save_interpolated_data(self):
        """Save interpolated data to CSV file"""
        try:
            # Create file dialog
            root = Tk()
            root.withdraw()
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Save Interpolated Data"
            )
            
            if not file_path:
                print("Save cancelled")
                return False
                
            # Create DataFrame and save
            wavelengths, intensities = zip(*self.interpolated_points)
            df = pd.DataFrame({
                'wavelength_nm': wavelengths,
                'intensity': intensities
            })
            
            df.to_csv(file_path, index=False)
            print(f"Data saved to: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            traceback.print_exc()
            return False

# Update main to include X calibration test
if __name__ == "__main__":
    root = None
    try:
        # Initialize Tk root window
        root = Tk()
        root.withdraw()
        
        digitizer = PlotDigitizer()
        
        # Clean start
        plt.close('all')
        cv2.destroyAllWindows()
        
        if (digitizer.test_image_display() and 
            digitizer.test_plot_type_selection() and
            digitizer.test_x_calibration() and
            digitizer.test_y_calibration() and
            digitizer.test_data_collection()):
            
            print("\nData collection test passed")
            plt.ioff()  # Turn off interactive mode
            
            while True:
                # Clean up before method selection
                cv2.destroyAllWindows()
                plt.close('all')
                
                # Show root temporarily for dialog
                root.deiconify()
                method = digitizer.choose_interpolation_method(root)
                root.withdraw()
                
                if not method:
                    print("No interpolation method selected")
                    break
                    
                print(f"\nUsing {method} interpolation method...")
                if digitizer.interpolate_data(method=method):
                    print(f"{method.capitalize()} interpolation successful")
                    print("Showing verification plot...")
                    
                    if digitizer.plot_verification():
                        if messagebox.askyesno("Verification", 
                            "Are you satisfied with the interpolation result?"):
                            if messagebox.askyesno("Save Data",
                                "Would you like to save the interpolated data?"):
                                digitizer.save_interpolated_data()
                            break
                        else:
                            print("\nTry a different interpolation method...")
                    else:
                        print("Plot verification failed")
                        if not messagebox.askyesno("Retry?", 
                            "Would you like to try a different method?"):
                            break
                else:
                    print(f"{method.capitalize()} interpolation failed")
                    if not messagebox.askyesno("Retry?", 
                        "Would you like to try a different method?"):
                        break
                        
    except Exception as e:
        print(f"Program error: {str(e)}")
        traceback.print_exc()
    finally:
        plt.close('all')
        cv2.destroyAllWindows()
        if root:
            try:
                root.quit()
                root.destroy()
            except:
                pass