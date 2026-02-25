import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import websocket
import json
import math
from threading import Thread

# Initialize position and calibration offsets
current_x, current_y = 0, 0
current_theta = 0
calibration_offsets = {"x": 0, "y": 0, "z": 0}
calibrated = False
calibration_samples = []

# Thresholds (temporarily set to zero for debugging)
THRESHOLD_X = 0.0
THRESHOLD_Y = 0.0

# Configuration Variables
incomputer = True

camera_ang = 60
x_angle = 0
camera_h = 44.0  # 44 cm
focal = 3.67
h_sensor = 5.7
hfov = 53.0
vfov = 38.0 
hfov_rad = np.radians(hfov)
vfov_rad = np.radians(vfov)
obs_circle_default = 1
obs_circle = obs_circle_default
_resize = 0.4
closest_obs_z = 1000000

mapx = 120
mapz = 250

directionx = 0.0
directiony = 0.0

fig, ax = plt.subplots()

# Camera position initialized
camera_x, camera_z, camera_theta = 0, 0, 0  # Added theta for orientation

# Declare smoothed_gyro_z_rad globally
smoothed_gyro_z = 0
smoothed_gyro_z_rad = 0

# Removed the adjust_for_tilt function as tilt is no longer considered

def calibrate_sensor(x, y, z):
    global calibrated, calibration_offsets, calibration_samples
    if not calibrated:
        calibration_samples.append((x, y, z))
        if len(calibration_samples) >= 100:
            avg_x = sum(sample[0] for sample in calibration_samples) / len(calibration_samples)
            avg_y = sum(sample[1] for sample in calibration_samples) / len(calibration_samples)
            avg_z = sum(sample[2] for sample in calibration_samples) / len(calibration_samples)
            calibration_offsets = {"x": avg_x, "y": avg_y, "z": avg_z}
            calibrated = True
            print(f"Calibration complete. Offsets: {calibration_offsets}")
            calibration_samples = []

def apply_calibration(x, y, z):
    z_corrected = z - calibration_offsets["z"] - 9.81  # Assuming z is vertical
    return (x - calibration_offsets["x"], 
            y - calibration_offsets["y"], 
            z_corrected)

def on_message(ws, message):
    global current_x, current_y, current_theta, smoothed_gyro_z, smoothed_gyro_z_rad
    data = json.loads(message)
    values = data['values']
    
    if len(values) < 3:
        print("Incomplete IMU data received.")
        return
    
    x_accel, y_accel, z_accel = values[0:3]
    gyro = data.get('gyro', [0, 0, 0])
    if len(gyro) < 3:
        print("Incomplete gyroscope data received.")
        return
    x_gyro, y_gyro, z_gyro = gyro

    # Smoothing gyroscope data
    ALPHA = 0.98  # Smoothing factor
    smoothed_gyro_z = ALPHA * smoothed_gyro_z + (1 - ALPHA) * z_gyro
    # Convert gyro_z to radians per second if it's in degrees per second
    smoothed_gyro_z_rad = math.radians(smoothed_gyro_z)
    current_theta += smoothed_gyro_z_rad * 0.033  # Assuming ~30Hz update rate

    print(f"Received IMU Data: accel_x={x_accel}, accel_y={y_accel}, accel_z={z_accel}, gyro_z={z_gyro}")
    print(f"Smoothed Gyro Z (rad/s): {smoothed_gyro_z_rad:.4f}")
    print(f"Updated Theta: {current_theta:.4f} radians")

    if not calibrated:
        calibrate_sensor(x_accel, y_accel, z_accel)
        return

    x_accel, y_accel, z_accel = apply_calibration(x_accel, y_accel, z_accel)
    
    # Removed tilt adjustment
    x_adjusted = x_accel
    y_adjusted = y_accel

    print(f"Adjusted Accelerations: x_adjusted={x_adjusted}, y_adjusted={y_adjusted}")

    if abs(x_adjusted) < THRESHOLD_X:
        x_adjusted = 0
    if abs(y_adjusted) < THRESHOLD_Y:
        y_adjusted = 0

    # Update global variables
    current_x = x_adjusted
    current_y = y_adjusted

    print(f"Current IMU Data: current_x={current_x}, current_y={current_y}, current_theta={current_theta:.4f} radians")

    # Save to file for Kalman filter (optional, can be removed if not used)
    with open('sensor_data.txt', 'w') as f:
        json.dump({
            'timestamp': time.time(),
            'accel_x': x_adjusted,
            'accel_y': y_adjusted,
            'gyro_z': smoothed_gyro_z_rad,
            'theta': current_theta
        }, f)

def on_error(ws, error):
    print("Error:", error)
    
def on_close(ws, close_code, reason):
    print("Connection closed:", reason)
    
def on_open(ws):
    print("Connected")

def connect(url):
    ws = websocket.WebSocketApp(url,
                              on_open=on_open,
                              on_message=on_message,
                              on_error=on_error,
                              on_close=on_close)
    ws.run_forever()

class KalmanFilter:
    def __init__(self):
        # State: [x, y, theta, vel_x, vel_y, angular_vel]
        self.state = np.zeros(6)
        self.P = np.eye(6) * 1000  # Initial uncertainty
        
        # Process noise
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.2  # Position and orientation
        self.Q[3:6, 3:6] *= 0.3  # Velocities
        
        # Measurement noise
        self.R_vision = np.eye(3) * 0.1  # Vision measurements
        
        self.dt = 1.0/30.0  # Assuming 30Hz update rate

    def predict(self, accel_x, accel_y, gyro_z):
        # State transition matrix
        F = np.eye(6)
        F[0, 3] = self.dt  # x += vel_x * dt
        F[1, 4] = self.dt  # y += vel_y * dt
        F[2, 5] = self.dt  # theta += angular_vel * dt
        
        # Control input model
        B = np.array([
            [0.5 * self.dt**2, 0],
            [0, 0.5 * self.dt**2],
            [0, 0],
            [self.dt, 0],
            [0, self.dt],
            [0, self.dt]
        ])
        u = np.array([accel_x, accel_y])

        # Predict state
        self.state = F @ self.state + B @ u
        self.P = F @ self.P @ F.T + self.Q

        print(f"After Prediction: {self.state}")

    def update_vision(self, measurement):
        # Vision measurement matrix (measures x, y, theta)
        H = np.zeros((3, 6))
        H[0:3, 0:3] = np.eye(3)
        
        self._update(measurement, H, self.R_vision)

    def _update(self, measurement, H, R):
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - (H @ self.state)
        self.state = self.state + (K @ y)
        self.P = (np.eye(6) - K @ H) @ self.P

        print(f"After Update: {self.state}")

def integrate_kalman_filter(create_map_func):
    kf = KalmanFilter()
    
    def wrapper(*args, **kwargs):
        global camera_x, camera_z, camera_theta
        
        # Get vision measurements
        px, py = create_map_func(*args, **kwargs)
        
        # Determine if we're in Navigation Mode
        in_navigation = env_memory.mode == "Navigation"
        
        if not in_navigation and px and py:
            # Perform vision update only if not in Navigation Mode
            # Example: Use average of detected points as vision measurement
            vision_x = np.mean(px) if px else kf.state[0]
            vision_z = np.mean(py) if py else kf.state[1]
            vision_theta = camera_theta  # Replace with actual orientation estimation if available
            
            vision_measurement = np.array([vision_x, vision_z, np.radians(vision_theta)])
            print(f"Vision Measurement: x={vision_x:.2f}, z={vision_z:.2f}, theta={vision_theta:.2f} radians")
            kf.update_vision(vision_measurement)
        else:
            if in_navigation:
                print("Navigation Mode: Skipping vision update.")
            else:
                print("No vision data available for update.")

        # Get IMU measurements from websocket data
        try:
            accel_x = current_x  # From websocket data (m/s²)
            accel_y = current_y  # From websocket data (m/s²)
            gyro_z = smoothed_gyro_z_rad  # From on_message, in radians/s
            
            print(f"IMU Measurement: accel_x={accel_x}, accel_y={accel_y}, gyro_z={gyro_z:.4f} rad/s")
            
            # Predict with accelerations and gyro
            kf.predict(accel_x, accel_y, gyro_z)
            
        except NameError:
            print("IMU data not available.")
            pass  # Skip if IMU data isn't available
        
        # Update global position variables
        camera_x = kf.state[0]
        camera_z = kf.state[1]
        camera_theta = np.degrees(kf.state[2])
        
        # Debugging: Print the current state
        print(f"KF State: x={camera_x:.2f}, z={camera_z:.2f}, theta={camera_theta:.2f}")
        
        # Pass updated position to 'update_plot'
        update_plot(px, py, camera_x, camera_z, camera_theta)
        
        return px, py
    
    return wrapper

# Global variables to store the environment
class EnvironmentMemory:
    def __init__(self):
        self.stored_px = []
        self.stored_py = []
        self.is_environment_captured = False
        self.camera_offset_x = 0
        self.camera_offset_z = 0
        self.mode = "Mapping"  # Modes: "Mapping", "Navigation"

env_memory = EnvironmentMemory()

def update_plot(px, py, camera_x, camera_z, camera_theta, color="red"):
    ax.clear()
    ax.set_xlim(-mapx, mapx)
    ax.set_ylim(0, mapz)
    ax.set_aspect('equal')

    # Plot stored environment if captured
    if env_memory.is_environment_captured:
        ax.scatter(
            [p + env_memory.camera_offset_x for p in env_memory.stored_px], 
            [p + env_memory.camera_offset_z for p in env_memory.stored_py], 
            color="blue", 
            s=1
        )
        # Plot 4 yellow holes at the corners
        corner_size = 50  # Size of the yellow holes
        corners_x = [-mapx, mapx, mapx, -mapx]
        corners_z = [0, 0, mapz, mapz]
        for (cx, cz) in zip(corners_x, corners_z):
            ax.scatter(cx, cz, color="yellow", s=corner_size, marker='o')

    # Plot current detected points if in Mapping mode
    if env_memory.mode == "Mapping":
        ax.scatter(px, py, color=color, s=1)

    # Plot camera position and orientation if in Navigation mode
    if env_memory.mode == "Navigation" and env_memory.is_environment_captured:
        ax.scatter(camera_x, camera_z, color="black", s=100, marker="o", label="Camera Position")  # Black circle for camera
        # Draw an arrow for orientation
        arrow_length = 10
        arrow_x = camera_x + arrow_length * np.cos(np.radians(camera_theta))
        arrow_z = camera_z + arrow_length * np.sin(np.radians(camera_theta))
        ax.arrow(camera_x, camera_z, arrow_x - camera_x, arrow_z - camera_z,
                 head_width=5, head_length=5, fc='black', ec='black')
        
        # Plot Kalman Filter estimate using passed parameters
        ax.scatter(camera_x, camera_z, color="green", s=50, marker="x", label="KF Estimate")

    # Add legend if needed
    if env_memory.mode == "Navigation" and env_memory.is_environment_captured:
        ax.legend(loc='upper right')

    plt.draw()
    plt.pause(0.001)

if incomputer:
    def empty(v):
        pass
    cv2.namedWindow("track_bar")
    cv2.resizeWindow("track_bar", 640,320)
    cv2.createTrackbar("x_angle", "track_bar",   0 , 360, empty)
    cv2.createTrackbar("z_angle", "track_bar",   0 , 360, empty)
    cv2.createTrackbar("hfov", "track_bar",   0 , 100, empty)
    cv2.createTrackbar("vfov", "track_bar",   0 , 100, empty)
    cv2.setTrackbarPos("x_angle", "track_bar", 180)
    cv2.setTrackbarPos("z_angle", "track_bar", 180+int(camera_ang))
    cv2.setTrackbarPos("hfov", "track_bar", int(hfov))
    cv2.setTrackbarPos("vfov", "track_bar", int(vfov))
    cap = cv2.VideoCapture(0)  # Default camera

def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)
    translated_point = (point[0] - center[0], point[1] - center[1])
    
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])
    
    rotated_point = np.dot(rotation_matrix, translated_point)
    
    final_point = (rotated_point[0] + center[0], rotated_point[1] + center[1])
    
    return final_point

def recalibrate_if_needed():
    global calibration_offsets, calibration_samples, calibrated
    if calibrated and not env_memory.is_environment_captured:
        # Check if stationary (e.g., IMU readings are close to 0)
        if abs(current_x) < 0.01 and abs(current_y) < 0.01 and abs(current_theta) < 0.01:
            calibrate_sensor(current_x, current_y, 0)
            print("Recalibrated offsets:", calibration_offsets)

def create_map(data):
    global camera_x, camera_z, camera_theta  # Declare as global to modify global variables

    if incomputer:
        try:
            ret, oimg = cap.read()
            if not ret:
                print("Failed to grab frame")
                return [], []
        except Exception as e:
            print("no pic:", e)
            return [], []

    global obs_circle
    oimg = cv2.GaussianBlur(oimg, (5, 5), 0)
    top, bottom, left, right = 1, 1, 1, 1
    oimg = cv2.copyMakeBorder(oimg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img_w = oimg.shape[1]
    img_h = oimg.shape[0]

    hsv_img = cv2.cvtColor(oimg, cv2.COLOR_BGR2HSV)
    
    px = []
    py = []
    list_x = []
    list_y = []
    
    # Color ranges for blue and yellow
    lower = [np.array([100, 150, 0]), np.array([20, 100, 100])]  # Blue and Yellow lower bounds
    upper = [np.array([140, 255, 255]), np.array([30, 255, 255])]  # Blue and Yellow upper bounds

    # Image Processing
    for _ in range(0, len(lower)):
        output = cv2.inRange(hsv_img, lower[_], upper[_])
        output = cv2.Laplacian(output, -1, 1, 1)
        ret, output = cv2.threshold(output, 30, 255, cv2.THRESH_BINARY)
        
        if not incomputer:
            __, contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        else:
            contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for contour in contours:
            if (cv2.contourArea(contour) < 500):
                continue

            for i in range(0, len(contour), 10):
                stapx = contour[i][0][0]
                stapy = contour[i][0][1]
                
                list_x.append(stapx)
                list_y.append(stapy)

        for i in range(len(list_x)):
            if list_x[i] < 100000:
                color_cv = (255, 255, 0) if _ == 1 else (255, 0, 0)  # Yellow for index 1, Blue for index 0
                oimg = cv2.circle(oimg, (list_x[i], list_y[i]), 2, color_cv, 3)

    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)

    for i in range(len(list_x)):
        ztheta = np.degrees(np.arctan((list_y[i] - (img_h / 2)) * np.tan(np.radians(vfov / 2)) / (img_h / 2))) + camera_ang
        z = (camera_h / np.sin(np.radians(ztheta))) * np.sin(np.radians(90 - ztheta))

        rotate_ang = np.radians(abs(camera_ang - ztheta))
        rotate_z = z / np.cos(np.radians(ztheta))
        hfov_rotate = 2 * np.arcsin(np.tan(hfov_rad / 2) / np.sqrt(np.tan(rotate_ang) ** 2 + (1 / (np.cos(hfov_rad / 2))) ** 2))
        hfov_rotate = np.degrees(hfov_rotate)

        xtheta = np.degrees(np.arctan((list_x[i] - (img_w / 2)) * np.tan(np.radians(hfov_rotate / 2)) / (img_w / 2)))
        x = rotate_z * np.tan(np.radians(xtheta))

        x_rotate = rotate_point((x, z), (0, 0), -x_angle)
        x = x_rotate[0]
        z = x_rotate[1]

        px.append(x)
        py.append(z)

    # Print detected points for debugging
    print(f"Detected Points: px={len(px)}, py={len(py)}")

    # Update the environment capture
    update_plot(px, py, camera_x, camera_z, camera_theta)

    # Create map visualization (Optional: You can extend this part as needed)
    map_img = np.zeros((mapz, mapx * 2, 3), np.uint8)
    # (Map drawing code can be added here if needed)

    # Display the image with detected points
    cv2.imshow("Video Feed", oimg)
    cv2.setWindowProperty("Video Feed", cv2.WND_PROP_TOPMOST, 1)  # Keep video feed window on top

    # Check for key press with a dedicated wait
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # 'c' to capture map
        if not env_memory.is_environment_captured:
            env_memory.stored_px = px.copy()
            env_memory.stored_py = py.copy()
            env_memory.is_environment_captured = True
            env_memory.mode = "Mapping"
            # Initialize camera position to the center
            if env_memory.stored_px and env_memory.stored_py:
                camera_x = np.mean(env_memory.stored_px)
                camera_z = np.mean(env_memory.stored_py)
                print(f"Camera initialized to center: x={camera_x:.2f}, z={camera_z:.2f}")
            print("Environment captured! Stored points:", len(env_memory.stored_px))
    elif key == ord('n'):  # 'n' to start navigation
        if env_memory.is_environment_captured:
            env_memory.mode = "Navigation"
            print("Entered Navigation Mode!")
    elif key == ord('q'):
        print("Capture key pressed!")
        env_memory.stored_px = px.copy()
        env_memory.stored_py = py.copy()
        env_memory.is_environment_captured = True
        # Initialize camera position to the center
        if env_memory.stored_px and env_memory.stored_py:
            camera_x = np.mean(env_memory.stored_px)
            camera_z = np.mean(env_memory.stored_py)
            print(f"Camera initialized to center: x={camera_x:.2f}, z={camera_z:.2f}")
        print("Environment captured! Stored points:", len(env_memory.stored_px))
    elif key == 27:  # ESC key to exit
        print("ESC key pressed. Exiting...")
        return [], []

    # Update camera offset if environment is captured and in Mapping mode
    if env_memory.is_environment_captured and env_memory.mode == "Mapping":
        env_memory.camera_offset_x = camera_x
        env_memory.camera_offset_z = camera_z

    return px, py

create_map = integrate_kalman_filter(create_map)

# Define color ranges for blue and yellow
lower_colors = [np.array([100, 150, 0]), np.array([20, 100, 100])]  # Blue and Yellow lower bounds
upper_colors = [np.array([140, 255, 255]), np.array([30, 255, 255])]  # Blue and Yellow upper bounds

def main():
    plt.axis('square')
    plt.xlim(-mapx, mapx)
    plt.ylim(0, mapz)

    while True:
        px, py = create_map(0)

        # Update camera parameters from trackbars
        global camera_ang, x_angle, hfov, vfov, hfov_rad, vfov_rad
        x_angle = cv2.getTrackbarPos("x_angle", "track_bar") - 180
        camera_ang = cv2.getTrackbarPos("z_angle", "track_bar") - 180
        hfov = cv2.getTrackbarPos("hfov", "track_bar")
        vfov = cv2.getTrackbarPos("vfov", "track_bar")
        hfov_rad = np.radians(hfov)
        vfov_rad = np.radians(vfov)

        # Add a short delay
        time.sleep(0.01)

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        accel = Thread(target=connect, args=("ws://192.168.1.108:8080/sensor/connect?type=android.sensor.accelerometer",), daemon=True)
        gyro = Thread(target=connect, args=("ws://192.168.1.108:8080/sensor/connect?type=android.sensor.gyroscope",), daemon=True)
        
        accel.start()
        gyro.start()
        recalibrate_if_needed()

        main()
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        if incomputer:
            cap.release()
        cv2.destroyAllWindows()
        print("Resources released")
