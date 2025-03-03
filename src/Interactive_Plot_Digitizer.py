import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, messagebox
import os
import scipy.interpolate as interp
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import numpy as np

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
            
            # Filter points within calibration ranges
            margin = 1  # Allow slight extrapolation
            valid_points = []
            for x, y in zip(x_data, y_data):
                if (min(x_pixels) - margin <= x <= max(x_pixels) + margin and 
                    min(y_pixels) - margin <= y <= max(y_pixels) + margin):
                    valid_points.append((x, y))
                else:
                    print(f"Point ({x}, {y}) outside calibration range - skipped")
            
            if not valid_points:
                print("No valid points after range filtering")
                return False
            
            x_data, y_data = zip(*valid_points)
            
            # Convert x coordinates to wavelengths
            x_interpolator = interp.interp1d(x_pixels, x_values, kind='linear')
            wavelengths = x_interpolator(x_data)
            
            # Convert y coordinates to intensities
            # For log plots, y_values are already actual intensities
            y_interpolator = interp.interp1d(y_pixels, y_values, kind='linear')
            intensities = y_interpolator(y_data)
            
            # Create wavelength range for output
            start_nm = int(min(wavelengths))
            end_nm = int(max(wavelengths)) + 1
            wavelength_range = np.arange(start_nm, end_nm, 1.0)
            
            # Perform interpolation based on chosen method
            if method == 'spline':
                interp_func = interp.CubicSpline(wavelengths, intensities)
            elif method == 'linear':
                interp_func = interp.interp1d(wavelengths, intensities, 
                                            kind='linear',
                                            bounds_error=False)
            elif method == 'local':
                # Local window approach
                window_size = 5
                interpolated = []
                for wave in wavelength_range:
                    mask = np.abs(wavelengths - wave) <= window_size
                    if np.sum(mask) >= 2:
                        local_waves = wavelengths[mask]
                        local_ints = intensities[mask]
                        local_interp = interp.interp1d(local_waves, local_ints,
                                                     bounds_error=False)
                        interpolated.append(float(local_interp(wave)))
                    else:
                        idx = np.argmin(np.abs(wavelengths - wave))
                        interpolated.append(float(intensities[idx]))
                
                self.interpolated_points = list(zip(wavelength_range, interpolated))
                return True
            
            # For spline and linear methods
            interpolated = interp_func(wavelength_range)
            self.interpolated_points = list(zip(wavelength_range, interpolated))
            print(f"Successfully interpolated {len(self.interpolated_points)} points using {method} method")
            return True
            
        except Exception as e:
            print(f"Interpolation error: {str(e)}")
            traceback.print_exc()
            return False

    def choose_interpolation_method(self):
        """Let user choose interpolation method"""
        root = Tk()
        root.withdraw()
        methods = {
            'Spline': 'spline',
            'Local Window': 'local',
            'Linear': 'linear'
        }
        
        msg = "Choose interpolation method:\n\n" + \
              "Spline - Best for smooth curves\n" + \
              "Local Window - Better for sharp transitions\n" + \
              "Linear - Simple point-to-point"
              
        choice = messagebox.askquestion("Interpolation Method", 
            msg + "\n\nTry spline method first?")
        
        if choice == 'yes':
            return 'spline'
        else:
            choice = messagebox.askquestion("Interpolation Method",
                "Try local window method?\n(No will use linear)")
            return 'local' if choice == 'yes' else 'linear'

    def plot_verification(self):
        """Show original points and interpolated curve with better scaling"""
        if not self.interpolated_points:
            print("Run interpolation first")
            return False
            
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original image with points
            ax1.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            x_data, y_data = zip(*self.data_points)
            ax1.plot(x_data, y_data, 'r.', label='Selected Points')
            ax1.set_title('Original Image with Points')
            
            # Convert original points to wavelengths and intensities
            x_to_wave = interp.interp1d([x for x, _ in self.x_calibration_points],
                                       [v for _, v in self.x_calibration_points],
                                       bounds_error=False,
                                       fill_value='extrapolate')
            
            y_to_int = interp.interp1d([y for y, _ in self.y_calibration_points],
                                      [v for _, v in self.y_calibration_points],
                                      bounds_error=False,
                                      fill_value='extrapolate')
            
            # Convert points and filter out invalid values
            x_orig = x_to_wave(x_data)
            y_orig = y_to_int(y_data)  # Values are already in actual units
            
            # Filter out any invalid points
            valid_mask = np.isfinite(x_orig) & np.isfinite(y_orig)
            if not np.any(valid_mask):
                print("No valid points to plot")
                return False
                
            x_orig = x_orig[valid_mask]
            y_orig = y_orig[valid_mask]
            
            # Plot interpolated data
            wavelengths, intensities = zip(*self.interpolated_points)
            # Filter interpolated points
            valid_interp = np.isfinite(intensities)
            wavelengths = np.array(wavelengths)[valid_interp]
            intensities = np.array(intensities)[valid_interp]
            
            ax2.plot(wavelengths, intensities, 'b-', label='Interpolated (1nm)')
            ax2.plot(x_orig, y_orig, 'r.', markersize=8, label='Original Points')
            
            ax2.set_xlabel('Wavelength (nm)')
            ax2.set_ylabel('Intensity')
            
            if self.is_log_plot:
                # Set log scale with safe limits
                y_min = max(1e-10, min(y_orig.min(), intensities.min()))
                y_max = min(1e10, max(y_orig.max(), intensities.max()))
                ax2.set_yscale('log')
                ax2.set_ylim(y_min, y_max)
                # Add grid with minor lines
                ax2.grid(True, which='both')
                ax2.grid(True, which='minor', alpha=0.2)
            else:
                y_min = min(y_orig.min(), intensities.min())
                y_max = max(y_orig.max(), intensities.max())
                ax2.set_ylim(y_min * 0.9, y_max * 1.1)
                ax2.grid(True)
                
            ax2.legend()
            ax2.set_title('Interpolated Data')
            
            plt.tight_layout()
            plt.show()
            return True
            
        except Exception as e:
            print(f"Plot verification error: {str(e)}")
            traceback.print_exc()
            return False

# Update main to include X calibration test
if __name__ == "__main__":
    digitizer = PlotDigitizer()
    print("Testing image display...")
    if digitizer.test_image_display():
        print("Image display test passed")
        print("\nTesting plot type selection...")
        if digitizer.test_plot_type_selection():
            print("Plot type selection test passed")
            print("\nTesting X-axis calibration...")
            if digitizer.test_x_calibration():
                print("X-axis calibration test passed")
                print("\nTesting Y-axis calibration...")
                if digitizer.test_y_calibration():
                    print("Y-axis calibration test passed")
                    print("\nTesting data point collection...")
                    if digitizer.test_data_collection():
                        print("\nData collection test passed")
                        
                        while True:
                            # Let user choose method
                            method = digitizer.choose_interpolation_method()
                            print(f"\nUsing {method} interpolation method...")
                            
                            if digitizer.interpolate_data(method=method):
                                print(f"{method.capitalize()} interpolation successful")
                                print("Showing verification plot...")
                                
                                try:
                                    digitizer.plot_verification()
                                    
                                    # Ask if result is satisfactory
                                    root = Tk()
                                    root.withdraw()
                                    if messagebox.askyesno("Verification", 
                                        "Are you satisfied with the interpolation result?"):
                                        break
                                    else:
                                        print("\nTry a different interpolation method...")
                                except Exception as e:
                                    print(f"Plot error: {str(e)}")
                                    traceback.print_exc()
                            else:
                                print(f"{method.capitalize()} interpolation failed")
                                if not messagebox.askyesno("Retry?", 
                                    "Would you like to try a different method?"):
                                    break
                    else:
                        print("\nData collection test failed")
                else:
                    print("Y-axis calibration test failed")
            else:
                print("X-axis calibration test failed")
        else:
            print("Plot type selection test failed")
    else:
        print("Image display test failed")