import cv2
import mediapipe as mp # type: ignore
import numpy as np
import time
import math
import json
import threading
import queue
import pyttsx3 # type: ignore
import asyncio
import aiohttp # type: ignore
from datetime import datetime
from enum import Enum
from collections import deque
from typing import Optional, Tuple, List, Dict
import os

# ================= ADVANCED CONFIGURATION =================
class ExerciseType(Enum):
    CENTER = "center"
    LEFT = "left"
    RIGHT = "right"
    UP = "up"
    DOWN = "down"

# Enhanced color palette
COLORS = {
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "red": (0, 0, 255),
    "blue": (255, 100, 0),
    "white": (255, 255, 255),
    "purple": (255, 0, 255),
    "cyan": (255, 255, 0),
    "orange": (0, 165, 255),
    "teal": (255, 128, 0)
}

# ================= 3D HEAD POSE ESTIMATION =================
class HeadPoseEstimator3D:
    """True 3D head pose estimation using facial landmarks"""
    
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Enable 3D landmarks
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 3D facial landmark model points (generic model in millimeters)
        self.model_points = np.array([
            (0.0, 0.0, 0.0),           # Nose tip
            (0.0, -330.0, -65.0),      # Chin
            (-225.0, 170.0, -135.0),   # Left eye left corner
            (225.0, 170.0, -135.0),    # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left mouth corner
            (150.0, -150.0, -125.0)    # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix approximation (webcam)
        self.focal_length = 1000
        self.camera_center = (640, 360)  # HD camera center
        self.camera_matrix = np.array([
            [self.focal_length, 0, self.camera_center[0]],
            [0, self.focal_length, self.camera_center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        
    def estimate_pose(self, frame) -> Tuple[float, float, float, bool]:
        """Estimate true 3D head pose (yaw, pitch, roll)"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return 0, 0, 0, False
            
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]
        
        # Get 2D image points from 3D landmarks
        image_points = np.array([
            (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
            (landmarks[152].x * w, landmarks[152].y * h), # Chin
            (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left corner
            (landmarks[263].x * w, landmarks[263].y * h), # Right eye right corner
            (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth corner
            (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
        ], dtype=np.float64)
        
        try:
            # Solve PnP for 3D pose estimation
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return 0, 0, 0, False
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles (yaw, pitch, roll)
            # Convert to degrees
            yaw = np.degrees(np.arcsin(rotation_matrix[2, 1]))
            pitch = np.degrees(np.arcsin(-rotation_matrix[2, 0]))
            roll = np.degrees(np.arcsin(rotation_matrix[1, 0]))
            
            return yaw, pitch, roll, True
            
        except Exception as e:
            print(f"3D Pose estimation error: {e}")
            return 0, 0, 0, False

# ================= KALMAN FILTER FOR SMOOTHING =================
class KalmanFilter3D:
    """Kalman filter for smooth 3D angle tracking"""
    
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        self.kf = cv2.KalmanFilter(6, 3)  # 6 states, 3 measurements
        
        # State transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        
        # Measurement noise covariance
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        
        # Initial state
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)
        
    def update(self, yaw, pitch, roll):
        """Update filter with new measurements"""
        measurement = np.array([[yaw], [pitch], [roll]], dtype=np.float32)
        
        # Prediction
        predicted = self.kf.predict()
        
        # Correction
        corrected = self.kf.correct(measurement)
        
        return float(corrected[0]), float(corrected[2]), float(corrected[4])
    
    def reset(self):
        """Reset filter state"""
        self.kf.statePost = np.zeros((6, 1), dtype=np.float32)

# ================= ADVANCED VOICE QUEUE SYSTEM =================
class VoiceCoachAdvanced:
    """Non-blocking voice system with message queue"""
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)
        self.engine.setProperty('volume', 0.9)
        self.message_queue = queue.Queue()
        self.is_running = True
        self.voice_thread = threading.Thread(target=self._voice_worker, daemon=True)
        self.voice_thread.start()
        
    def speak(self, text: str, priority: int = 0):
        """Queue speech message (priority 0=highest, 2=lowest)"""
        self.message_queue.put((priority, text, time.time()))
        
    def _voice_worker(self):
        """Background worker for voice synthesis"""
        while self.is_running:
            try:
                # Get message with timeout to allow exit
                priority, text, timestamp = self.message_queue.get(timeout=0.1)
                
                # Check if message is still relevant (not too old)
                if time.time() - timestamp < 3.0:  # 3-second freshness
                    try:
                        self.engine.say(text)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"Voice error: {e}")
                        
                self.message_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice worker error: {e}")
                
    def stop(self):
        """Stop voice system"""
        self.is_running = False
        self.voice_thread.join(timeout=1.0)

# ================= AI BRAIN INTEGRATION =================
class MedicalAIAnalyzer:
    """AI-powered medical analysis using Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
    async def analyze_performance(self, performance_data: Dict) -> Dict:
        """Analyze performance with AI medical insights"""
        
        # If no API key, use local expert system
        if not self.api_key:
            return self._local_analysis(performance_data)
        
        try:
            prompt = self._create_ai_prompt(performance_data)
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Content-Type": "application/json",
                }
                
                payload = {
                    "contents": [{
                        "parts": [{
                            "text": prompt
                        }]
                    }],
                    "generationConfig": {
                        "temperature": 0.7,
                        "topK": 1,
                        "topP": 0.95,
                        "maxOutputTokens": 500,
                    }
                }
                
                url = f"{self.api_url}?key={self.api_key}"
                
                async with session.post(url, json=payload, headers=headers) as response:
                    if response.status == 200:
                        result = await response.json()
                        analysis = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        return self._parse_ai_response(analysis)
                    else:
                        print(f"AI API error: {response.status}")
                        return self._local_analysis(performance_data)
                        
        except Exception as e:
            print(f"AI analysis error: {e}")
            return self._local_analysis(performance_data)
    
    def _create_ai_prompt(self, data: Dict) -> str:
        """Create medical analysis prompt for AI"""
        
        prompt = f"""You are a physical therapist specializing in neck rehabilitation.

Patient Exercise Results:
- Left rotation: {data.get('left_angle', 0):.1f}° (target: 25°)
- Right rotation: {data.get('right_angle', 0):.1f}° (target: 25°)
- Upward gaze: {data.get('up_angle', 0):.1f}° (target: 25°)
- Downward gaze: {data.get('down_angle', 0):.1f}° (target: 25°)
- Symmetry score: {data.get('symmetry', 0):.1f}/100
- Movement quality: {data.get('quality', 0):.1f}/100

Please provide:
1. Medical interpretation of limitations
2. Specific muscles likely affected
3. 3 personalized exercise recommendations
4. Safety warnings if needed

Format response as JSON with: analysis, affected_muscles, recommendations, warnings"""

        return prompt
    
    def _parse_ai_response(self, response: str) -> Dict:
        """Parse AI response into structured data"""
        try:
            # Try to extract JSON from response
            lines = response.strip().split('\n')
            json_start = None
            json_end = None
            
            for i, line in enumerate(lines):
                if '{' in line:
                    json_start = i
                    break
                    
            for i in range(len(lines)-1, -1, -1):
                if '}' in lines[i]:
                    json_end = i + 1
                    break
            
            if json_start is not None and json_end is not None:
                json_str = '\n'.join(lines[json_start:json_end])
                return json.loads(json_str)
            else:
                return {"analysis": response}
                
        except:
            return {"analysis": response}
    
    def _local_analysis(self, data: Dict) -> Dict:
        """Local expert system when AI is unavailable"""
        analysis = ""
        
        left = data.get('left_angle', 0)
        right = data.get('right_angle', 0)
        up = data.get('up_angle', 0)
        down = data.get('down_angle', 0)
        symmetry = data.get('symmetry', 0)
        
        # Symmetry analysis
        if abs(left - right) > 10:
            if left > right:
                analysis += f"⚠️ Limited right rotation ({right:.1f}°) compared to left ({left:.1f}°). "
                analysis += "May indicate tightness in left sternocleidomastoid or right cervical rotators.\n"
            else:
                analysis += f"⚠️ Limited left rotation ({left:.1f}°) compared to right ({right:.1f}°). "
                analysis += "May indicate tightness in right sternocleidomastoid or left cervical rotators.\n"
        else:
            analysis += "✅ Good symmetry between left and right rotation.\n"
        
        # Range of motion analysis
        if up < 15:
            analysis += f"⚠️ Limited upward gaze ({up:.1f}°). Often caused by tight suboccipital muscles from forward head posture.\n"
            analysis += "Try chin tuck exercises and suboccipital stretches.\n"
        
        if down < 15:
            analysis += f"⚠️ Limited downward gaze ({down:.1f}°). May indicate cervical extensor tightness.\n"
            analysis += "Try gentle neck flexor strengthening and chin-to-chest stretches.\n"
        
        # General recommendations
        analysis += "\n🔹 RECOMMENDATIONS:\n"
        analysis += "1. Perform stretches slowly and hold for 30 seconds\n"
        analysis += "2. Avoid jerky movements during exercises\n"
        analysis += "3. Maintain proper posture throughout the day\n"
        analysis += "4. Take frequent breaks from screen time\n"
        
        return {
            "analysis": analysis,
            "affected_muscles": ["Sternocleidomastoid", "Trapezius", "Suboccipitals"],
            "recommendations": [
                "Gentle neck rotations",
                "Chin tuck exercises",
                "Scapular retractions"
            ],
            "warnings": ["Stop if you feel sharp pain"]
        }

# ================= ADVANCED EXERCISE TRACKER =================
class ExerciseAdvanced:
    """Enhanced exercise tracking with velocity monitoring"""
    
    def __init__(self, name: str, target_angle: float, direction: str):
        self.name = name
        self.target_angle = target_angle
        self.direction = direction
        self.best_angle = 0
        self.start_time = 0
        self.hold_time = 0
        self.completed = False
        self.quality = 0
        self.velocity_history = deque(maxlen=10)  # Track velocity
        self.max_safe_velocity = 40  # degrees per second
        self.safe_zone_count = 0
        
    def start(self):
        self.start_time = time.time()
        self.velocity_history.clear()
        self.safe_zone_count = 0
        
    def update(self, current_angle: float, velocity: float = 0) -> Tuple[bool, List[str]]:
        """Update exercise with velocity monitoring"""
        warnings = []
        
        # Track best angle
        if abs(current_angle) > abs(self.best_angle):
            self.best_angle = current_angle
        
        # Velocity safety check
        if abs(velocity) > self.max_safe_velocity:
            warnings.append("⚠️ TOO FAST! Slow down")
        elif abs(velocity) < 5:  # Good control
            self.safe_zone_count += 1
        
        # Store velocity for smoothing
        self.velocity_history.append(velocity)
        
        # Calculate hold progress
        self.hold_time = time.time() - self.start_time
        angle_ratio = abs(current_angle) / abs(self.target_angle) if self.target_angle != 0 else 0
        self.quality = min(1.0, angle_ratio * 0.7 + (self.safe_zone_count / 50) * 0.3)
        
        # Check completion
        if (self.hold_time >= 2.0 and 
            angle_ratio >= 0.7 and 
            self.safe_zone_count >= 20):
            self.completed = True
            return True, warnings
        
        return False, warnings

# ================= MAIN APPLICATION =================
class NeckExerciseCoachAdvanced:
    """Enhanced neck exercise coach with all upgrades"""
    
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator3D()
        self.kalman_filter = KalmanFilter3D()
        self.voice = VoiceCoachAdvanced()
        self.ai_analyzer = MedicalAIAnalyzer()
        self.cap = None
        
        self.exercises = []
        self.current_exercise_index = 0
        self.calibrated = False
        self.neutral_yaw = 0
        self.neutral_pitch = 0
        self.neutral_roll = 0
        
        self.performance_data = {}
        self.safety_warnings = []
        self.last_angles = (0, 0, 0)
        self.last_time = 0
        self.angle_buffer = deque(maxlen=5)  # For additional smoothing
        
        self.velocity_history = deque(maxlen=5)
        self.max_angular_velocity = 45  # Degrees per second
        
    def setup_exercises(self):
        """Create exercise sequence with dynamic targets"""
        self.exercises = [
            ExerciseAdvanced("Center Position", 0, "center"),
            ExerciseAdvanced("Left Turn", -25, "left"),
            ExerciseAdvanced("Right Turn", 25, "right"),
            ExerciseAdvanced("Upward Look", 25, "up"),
            ExerciseAdvanced("Downward Look", -25, "down"),
            ExerciseAdvanced("Return to Center", 0, "center")
        ]
    
    def calibrate(self) -> bool:
        """Advanced 3D calibration"""
        print("\n" + "="*60)
        print("3D CALIBRATION - Look straight at camera")
        print("="*60)
        
        self.voice.speak("Please look straight at the camera for 3D calibration", priority=0)
        
        angles_yaw = []
        angles_pitch = []
        angles_roll = []
        start_time = time.time()
        
        calibration_frames = 30  # Use multiple frames for accuracy
        
        for i in range(calibration_frames):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            frame = cv2.flip(frame, 1)
            yaw, pitch, roll, detected = self.pose_estimator.estimate_pose(frame)
            
            if detected:
                angles_yaw.append(yaw)
                angles_pitch.append(pitch)
                angles_roll.append(roll)
                
                # Update Kalman filter during calibration
                self.kalman_filter.update(yaw, pitch, roll)
            
            # Show calibration progress
            self._draw_calibration_screen(frame, i + 1, calibration_frames)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False
        
        if angles_yaw:
            # Use filtered values for better accuracy
            self.neutral_yaw = np.median(angles_yaw)
            self.neutral_pitch = np.median(angles_pitch)
            self.neutral_roll = np.median(angles_roll)
            self.calibrated = True
            
            print(f"✓ 3D Calibration Complete:")
            print(f"  Neutral Yaw: {self.neutral_yaw:.1f}°")
            print(f"  Neutral Pitch: {self.neutral_pitch:.1f}°")
            print(f"  Neutral Roll: {self.neutral_roll:.1f}°")
            
            self.voice.speak("3D calibration complete. Starting exercises.", priority=0)
            return True
        
        print("✗ Calibration failed - no face detected")
        return False
    
    def _draw_calibration_screen(self, frame, current, total):
        """Draw calibration screen with 3D visualization"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Title
        cv2.putText(frame, "3D CALIBRATION", (w//2 - 150, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS["cyan"], 3)
        
        # Instructions
        cv2.putText(frame, "Look straight ahead at the camera", 
                   (w//2 - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["white"], 2)
        
        # Progress
        progress = current / total
        cv2.putText(frame, f"Calibrating: {int(progress*100)}%", 
                   (w//2 - 100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["yellow"], 2)
        
        # Progress bar
        bar_width = 400
        bar_height = 30
        bar_x = (w - bar_width) // 2
        bar_y = h // 2
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height),
                     COLORS["green"], -1)
        
        # 3D axis visualization
        self._draw_3d_axes(frame, w//2, h//2 + 100)
        
        cv2.imshow("Neck Exercise Coach 3D", frame)
    
    def _draw_3d_axes(self, frame, x, y, size=100):
        """Draw 3D axes for visualization"""
        # X axis (red) - yaw
        cv2.arrowedLine(frame, (x, y), (x + size, y), (0, 0, 255), 3)
        cv2.putText(frame, "Yaw", (x + size + 10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Y axis (green) - pitch
        cv2.arrowedLine(frame, (x, y), (x, y - size), (0, 255, 0), 3)
        cv2.putText(frame, "Pitch", (x, y - size - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Z axis (blue) - roll
        cv2.arrowedLine(frame, (x, y), (x - size//2, y + size//2), (255, 0, 0), 3)
        cv2.putText(frame, "Roll", (x - size//2 - 40, y + size//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    def calculate_angular_velocity(self, current_angles: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate angular velocity in degrees per second"""
        current_time = time.time()
        
        if self.last_time == 0:
            self.last_time = current_time
            self.last_angles = current_angles
            return 0, 0, 0
        
        dt = current_time - self.last_time
        if dt < 0.01:  # Too small time difference
            return 0, 0, 0
        
        # Calculate velocity for each axis
        vel_yaw = (current_angles[0] - self.last_angles[0]) / dt
        vel_pitch = (current_angles[1] - self.last_angles[1]) / dt
        vel_roll = (current_angles[2] - self.last_angles[2]) / dt
        
        # Smooth velocity
        self.velocity_history.append((vel_yaw, vel_pitch, vel_roll))
        
        # Return average velocity
        avg_vel = np.mean(self.velocity_history, axis=0) if self.velocity_history else (0, 0, 0)
        
        self.last_angles = current_angles
        self.last_time = current_time
        
        return avg_vel[0], avg_vel[1], avg_vel[2]
    
    def check_safety(self, yaw: float, pitch: float, roll: float, 
                    vel_yaw: float, vel_pitch: float) -> List[str]:
        """Enhanced safety checks with velocity monitoring"""
        warnings = []
        
        # Velocity warnings
        total_velocity = math.sqrt(vel_yaw**2 + vel_pitch**2)
        if total_velocity > self.max_angular_velocity:
            warnings.append("⚠️ MOVING TOO FAST!")
            self.voice.speak("Too fast, slow down", priority=0)
        
        # Extreme angle warnings
        if abs(yaw) > 50:
            warnings.append("⚠️ Extreme head rotation")
            self.voice.speak("Reduce rotation", priority=1)
        
        if abs(pitch) > 40:
            warnings.append("⚠️ Extreme head tilt")
            self.voice.speak("Reduce tilt", priority=1)
        
        # Sudden movement detection
        if len(self.velocity_history) >= 3:
            recent_vel = np.array(self.velocity_history)[-3:]
            if np.any(np.abs(np.diff(recent_vel, axis=0)) > 30):
                warnings.append("⚠️ Jerky movement detected")
                self.voice.speak("Smooth movement please", priority=1)
        
        return warnings
    
    def run_exercise(self) -> bool:
        """Run enhanced exercise with 3D tracking"""
        exercise = self.exercises[self.current_exercise_index]
        
        if not exercise.start_time:
            exercise.start()
            self.voice.speak(f"Now, {exercise.name}", priority=0)
        
        last_encouragement = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            
            # Get 3D pose
            yaw_raw, pitch_raw, roll_raw, detected = self.pose_estimator.estimate_pose(frame)
            
            if not detected:
                self._draw_face_not_found(frame)
                cv2.imshow("Neck Exercise Coach 3D", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
                continue
            
            # Apply Kalman filter for smooth tracking
            yaw_smooth, pitch_smooth, roll_smooth = self.kalman_filter.update(
                yaw_raw, pitch_raw, roll_raw
            )
            
            # Adjust with calibration
            yaw = yaw_smooth - self.neutral_yaw
            pitch = pitch_smooth - self.neutral_pitch
            roll = roll_smooth - self.neutral_roll
            
            # Calculate angular velocity
            vel_yaw, vel_pitch, vel_roll = self.calculate_angular_velocity((yaw, pitch, roll))
            
            # Safety checks
            safety_warnings = self.check_safety(yaw, pitch, roll, vel_yaw, vel_pitch)
            
            # Determine which angle to track for current exercise
            if exercise.direction == "center":
                current_angle = math.sqrt(yaw**2 + pitch**2)
                current_velocity = math.sqrt(vel_yaw**2 + vel_pitch**2)
            elif exercise.direction in ["left", "right"]:
                current_angle = yaw
                current_velocity = abs(vel_yaw)
            else:  # up or down
                current_angle = pitch
                current_velocity = abs(vel_pitch)
            
            # Update exercise with velocity monitoring
            completed, exercise_warnings = exercise.update(current_angle, current_velocity)
            all_warnings = safety_warnings + exercise_warnings
            
            # Voice encouragement
            current_time = time.time()
            progress = min(1.0, exercise.hold_time / 2.0)
            
            if 0.3 < progress < 0.35 and current_time - last_encouragement > 2:
                self.voice.speak("Good form", priority=2)
                last_encouragement = current_time
            elif 0.7 < progress < 0.75 and current_time - last_encouragement > 2:
                self.voice.speak("Keep holding", priority=2)
                last_encouragement = current_time
            
            # Draw enhanced interface
            frame = self._draw_enhanced_interface(
                frame, exercise, yaw, pitch, roll,
                vel_yaw, vel_pitch, progress,
                exercise.hold_time, exercise.best_angle,
                all_warnings
            )
            
            cv2.imshow("Neck Exercise Coach 3D", frame)
            
            # Check completion
            if completed:
                self.voice.speak("Excellent! Well done", priority=0)
                time.sleep(0.5)
                break
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('s'):
                self.voice.speak("Skipping exercise", priority=0)
                break
            elif key == ord('r'):
                exercise.start()
                self.voice.speak("Restarting exercise", priority=0)
        
        return True
    
    def _draw_face_not_found(self, frame):
        """Draw face not found screen"""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, "FACE NOT DETECTED", (w//2 - 200, h//2 - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS["red"], 3)
        cv2.putText(frame, "Move into camera view", (w//2 - 180, h//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["white"], 2)
    
    def _draw_enhanced_interface(self, frame, exercise, yaw, pitch, roll,
                               vel_yaw, vel_pitch, progress, hold_time,
                               best_angle, warnings):
        """Draw enhanced interface with 3D visualization"""
        h, w = frame.shape[:2]
        
        # Dark overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, int(h*0.35)), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Exercise info
        cv2.putText(frame, f"EXERCISE: {exercise.name}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS["blue"], 2)
        
        # Instructions
        if exercise.direction == "center":
            inst = "LOOK STRAIGHT AHEAD"
        elif exercise.direction == "left":
            inst = "TURN HEAD SLOWLY TO LEFT"
        elif exercise.direction == "right":
            inst = "TURN HEAD SLOWLY TO RIGHT"
        elif exercise.direction == "up":
            inst = "LOOK UP GENTLY"
        else:  # down
            inst = "LOOK DOWN GENTLY"
        
        cv2.putText(frame, inst, (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["cyan"], 2)
        
        # Current angles with velocity
        cv2.putText(frame, f"Yaw: {yaw:6.1f}° ({vel_yaw:4.1f}°/s)", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["yellow"], 2)
        cv2.putText(frame, f"Pitch: {pitch:6.1f}° ({vel_pitch:4.1f}°/s)", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["yellow"], 2)
        cv2.putText(frame, f"Roll: {roll:6.1f}°", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["yellow"], 2)
        
        # Target and best
        cv2.putText(frame, f"Target: {exercise.target_angle}°", (300, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["purple"], 2)
        cv2.putText(frame, f"Best: {best_angle:5.1f}°", (300, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["green"], 2)
        
        # Quality indicator
        quality_color = COLORS["green"] if exercise.quality > 0.7 else COLORS["yellow"] if exercise.quality > 0.4 else COLORS["red"]
        cv2.putText(frame, f"Quality: {exercise.quality*100:.0f}%", (300, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, quality_color, 2)
        
        # Progress bar with time
        bar_width = 400
        bar_height = 25
        bar_x = (w - bar_width) // 2
        bar_y = h - 100
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Progress
        progress_width = int(bar_width * progress)
        bar_color = COLORS["green"] if progress > 0.7 else COLORS["yellow"] if progress > 0.3 else COLORS["red"]
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height),
                     bar_color, -1)
        
        # Border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     COLORS["white"], 2)
        
        # Progress text
        time_text = f"HOLD: {hold_time:.1f}s / 2.0s"
        cv2.putText(frame, time_text, (bar_x, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS["white"], 2)
        
        # 3D visualization
        self._draw_head_pose_visualization(frame, w - 150, 150, yaw, pitch, roll)
        
        # Warnings
        if warnings:
            for i, warning in enumerate(warnings[:3]):  # Show max 3 warnings
                cv2.putText(frame, warning, (20, h - 160 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["red"], 2)
        
        return frame
    
    def _draw_head_pose_visualization(self, frame, x, y, yaw, pitch, roll, size=80):
        """Visualize head pose as a 3D cube"""
        # Draw head circle
        cv2.circle(frame, (x, y), size, COLORS["white"], 2)
        
        # Convert angles to rotation
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Calculate nose position based on angles
        nose_x = int(x + np.sin(yaw_rad) * size * 0.8)
        nose_y = int(y - np.sin(pitch_rad) * size * 0.8)
        
        # Draw nose direction
        cv2.line(frame, (x, y), (nose_x, nose_y), COLORS["green"], 3)
        cv2.circle(frame, (nose_x, nose_y), 8, COLORS["red"], -1)
        
        # Draw rotation indicators
        if abs(yaw) > 5:
            side = "LEFT" if yaw < 0 else "RIGHT"
            color = COLORS["orange"] if abs(yaw) > 20 else COLORS["yellow"]
            cv2.putText(frame, side, (x - 20, y - size - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    async def analyze_performance_ai(self):
        """AI-powered performance analysis"""
        print("\n" + "="*60)
        print("AI MEDICAL ANALYSIS")
        print("="*60)
        
        # Collect performance data
        performance = {
            "left_angle": 0,
            "right_angle": 0,
            "up_angle": 0,
            "down_angle": 0,
            "symmetry": 0,
            "quality": 0
        }
        
        for ex in self.exercises:
            if ex.direction == "left":
                performance["left_angle"] = abs(ex.best_angle)
            elif ex.direction == "right":
                performance["right_angle"] = abs(ex.best_angle)
            elif ex.direction == "up":
                performance["up_angle"] = abs(ex.best_angle)
            elif ex.direction == "down":
                performance["down_angle"] = abs(ex.best_angle)
        
        # Calculate symmetry
        if performance["left_angle"] > 0 and performance["right_angle"] > 0:
            smaller = min(performance["left_angle"], performance["right_angle"])
            larger = max(performance["left_angle"], performance["right_angle"])
            performance["symmetry"] = (smaller / larger) * 100 if larger > 0 else 0
        
        # Calculate overall quality
        qualities = [ex.quality for ex in self.exercises if ex.quality > 0]
        performance["quality"] = np.mean(qualities) * 100 if qualities else 0
        
        # Get AI analysis
        self.voice.speak("Analyzing your performance with AI", priority=0)
        ai_analysis = await self.ai_analyzer.analyze_performance(performance)
        
        # Save to JSON
        self.save_ai_data(performance, ai_analysis)
        
        return ai_analysis
    
    def save_ai_data(self, performance: Dict, ai_analysis: Dict):
        """Save AI analysis to JSON"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "performance": performance,
            "ai_analysis": ai_analysis,
            "exercises": []
        }
        
        for ex in self.exercises:
            data["exercises"].append({
                "name": ex.name,
                "direction": ex.direction,
                "best_angle": float(ex.best_angle),
                "hold_time": float(ex.hold_time),
                "quality": float(ex.quality),
                "completed": ex.completed
            })
        
        filename = f"neck_ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ AI analysis saved to {filename}")
        return filename
    
    def show_results(self, analysis: Dict):
        """Show AI-powered results screen"""
        self.voice.speak("Workout complete! Here is your AI analysis", priority=0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            # Dark overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
            
            # Title
            cv2.putText(frame, "🧠 AI MEDICAL ANALYSIS", (w//2 - 200, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS["cyan"], 3)
            
            # Performance summary
            y_offset = 120
            completed = sum(1 for ex in self.exercises if ex.completed)
            cv2.putText(frame, f"Exercises Completed: {completed}/{len(self.exercises)}", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS["green"], 2)
            y_offset += 50
            
            # AI Analysis
            analysis_text = analysis.get("analysis", "No AI analysis available")
            lines = self._wrap_text(analysis_text, 70)
            
            for line in lines[:12]:  # Show first 12 lines
                color = COLORS["red"] if "⚠️" in line else COLORS["green"] if "✅" in line else COLORS["yellow"]
                cv2.putText(frame, line, (50, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_offset += 30
            
            # Recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                y_offset += 20
                cv2.putText(frame, "Recommended Exercises:", (50, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["purple"], 2)
                y_offset += 40
                
                for i, rec in enumerate(recommendations[:4]):
                    cv2.putText(frame, f"{i+1}. {rec}", (70, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["white"], 2)
                    y_offset += 30
            
            # Controls
            cv2.putText(frame, "Press 'R' to restart or 'Q' to quit", 
                       (w//2 - 200, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["white"], 2)
            
            cv2.imshow("Neck Exercise Coach 3D", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('r'):
                return True
    
    def _wrap_text(self, text: str, max_length: int) -> List[str]:
        """Wrap text to fit on screen"""
        words = text.split()
        lines = []
        current_line = []
        
        for word in words:
            if len(' '.join(current_line + [word])) <= max_length:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    async def run(self):
        """Main async loop"""
        print("\n" + "="*60)
        print("NECK EXERCISE COACH 3D - ADVANCED EDITION")
        print("="*60)
        print("Features:")
        print("  ✓ True 3D Head Pose Tracking")
        print("  ✓ Kalman Filter Smoothing")
        print("  ✓ AI Medical Analysis (Gemini)")
        print("  ✓ Non-blocking Voice Queue")
        print("  ✓ Real-time Velocity Safety")
        print("="*60)
        print("Controls:")
        print("  Q - Quit")
        print("  S - Skip exercise")
        print("  R - Restart exercise / Results screen")
        print("="*60)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ ERROR: Cannot open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        cv2.namedWindow("Neck Exercise Coach 3D", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Neck Exercise Coach 3D", 1024, 768)
        
        try:
            # Setup exercises
            self.setup_exercises()
            
            # Calibrate
            if not self.calibrate():
                return
            
            # Main exercise loop
            self.current_exercise_index = 0
            
            while self.current_exercise_index < len(self.exercises):
                exercise = self.exercises[self.current_exercise_index]
                print(f"\nStarting: {exercise.name}")
                
                if not self.run_exercise():
                    break
                
                self.current_exercise_index += 1
                
                # Brief transition
                if self.current_exercise_index < len(self.exercises):
                    next_ex = self.exercises[self.current_exercise_index]
                    self.voice.speak(f"Next, {next_ex.name}", priority=1)
                    time.sleep(1.0)
            
            # AI Analysis and Results
            if self.current_exercise_index >= len(self.exercises):
                ai_analysis = await self.analyze_performance_ai()
                
                if self.show_results(ai_analysis):
                    # Restart
                    self.current_exercise_index = 0
                    self.kalman_filter.reset()
                    for ex in self.exercises:
                        ex.start_time = 0
                        ex.hold_time = 0
                        ex.best_angle = 0
                        ex.completed = False
                        ex.quality = 0
                    await self.run()  # Restart
                    
        except KeyboardInterrupt:
            print("\n\nProgram interrupted")
        except Exception as e:
            print(f"\nERROR: {e}")
        finally:
            if self.cap:
                self.cap.release()
            self.voice.stop()
            cv2.destroyAllWindows()
            print("\n✅ Program closed successfully")

# ================= RUN PROGRAM =================
if __name__ == "__main__":
    # Required packages:
    # pip install opencv-python mediapipe numpy pyttsx3 aiohttp
    
    import asyncio
    
    # Optional: Set your Gemini API key for AI analysis
    # os.environ["GEMINI_API_KEY"] = "your_api_key_here"
    
    coach = NeckExerciseCoachAdvanced()
    
    # Run async main function
    asyncio.run(coach.run())