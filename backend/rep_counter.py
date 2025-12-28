import cv2
import mediapipe as mp # type: ignore
import numpy as np
import threading
import time
import random
import pyttsx3 # type: ignore
from google import genai

# ================= CONFIG =================
ANGLE_STAND = 160      # knee angle when standing
ANGLE_SQUAT = 90       # knee angle when squatting
SMOOTH_ALPHA = 0.3     # smoothing factor
BACK_ANGLE_THRESHOLD = 150  # for checking if leaning too far forward
REST_TIMEOUT = 10      # seconds of inactivity to trigger analysis
CHECK_INTERVAL = 5     # reps between encouragement

# ================= TTS & AI SETUP =================
# Initialize offline text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 0.9)

# Fallback motivational quotes (used when AI is unavailable)
FALLBACK_QUOTES = [
    "Keep pushing, you're doing great!",
    "Focus on your form! Keep your chest up!",
    "Don't stop now! One more rep!",
    "Squeeze at the top! Feel that burn!",
    "Nice depth! Maintain that form!",
    "You've got this! Power through!",
    "Control the movement on the way down!",
    "Explode up! Use your glutes!",
    "Breathe out as you push up!",
    "Great work! Stay consistent!"
]

# Configure Gemini AI (comment out if no API key)
# genai.configure(api_key="YOUR_API_KEY_HERE")

# ================= MEDIAPIPE =================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= HELPERS =================
def calculate_angle(a, b, c):
    """Calculate angle at point b (degrees)"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle

def speak_thread(text):
    """Worker function to speak text without blocking video"""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")

def speak(text):
    """Call this to make the AI talk (non-blocking)"""
    t = threading.Thread(target=speak_thread, args=(text,))
    t.daemon = True
    t.start()

def get_smart_feedback(reps, form_errors):
    """Decides what to say: tries AI first, falls back to offline list"""
    try:
        # Try to use Gemini AI if configured
        if 'genai' in globals() and hasattr(genai, 'GenerativeModel'):
            model = genai.GenerativeModel('gemini-pro')
            prompt = f"I just completed {reps} squats. Form issues detected: {form_errors if form_errors else 'Good form overall'}. Give me one short, motivational sentence to improve my next set. Keep it under 10 words."
            response = model.generate_content(prompt)
            return response.text.strip()
        else:
            raise Exception("Gemini not configured")
    except Exception as e:
        # Fallback to offline quotes
        print(f"⚠️ AI Feedback Error: {e}. Using offline quotes.")
        base_quote = random.choice(FALLBACK_QUOTES)
        if reps >= 10:
            return f"Awesome! {reps} reps done! {base_quote}"
        elif form_errors:
            return f"{base_quote} Watch your form!"
        else:
            return base_quote

# ================= STATE =================
rep_count = 0
state = "UP"
angle_smooth = None
last_rep_time = time.time()
bad_form_counter = 0
current_set_errors = []
last_encouragement_at = 0
back_angle_history = []
analysis_triggered = False

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Squat Counter Started")
print("Press 'q' to quit")
print("Make sure you're sideways to the camera for best results")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Calculate time since last rep
    time_since_last_rep = time.time() - last_rep_time
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark

        # LEFT leg points
        hip = [lm[mp_pose.PoseLandmark.LEFT_HIP].x,
               lm[mp_pose.PoseLandmark.LEFT_HIP].y]
        knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x,
                lm[mp_pose.PoseLandmark.LEFT_KNEE].y]
        ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                 lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

        # Calculate knee angle
        angle_raw = calculate_angle(hip, knee, ankle)

        # Smoothing
        if angle_smooth is None:
            angle_smooth = angle_raw
        else:
            angle_smooth = ((1 - SMOOTH_ALPHA) * angle_smooth + 
                          SMOOTH_ALPHA * angle_raw)

        # Calculate back angle (for form check)
        # Using shoulder-hip-knee to check forward lean
        shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                   lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        back_angle = calculate_angle(shoulder, hip, knee)
        
        # Store recent back angles for smoothing
        back_angle_history.append(back_angle)
        if len(back_angle_history) > 5:
            back_angle_history.pop(0)
        back_angle_avg = np.mean(back_angle_history)

        # ========== FORM CHECK ==========
        form_error_detected = False
        
        if angle_smooth < ANGLE_SQUAT + 20:  # When in squatting range
            # Check for forward lean (back too horizontal)
            if back_angle_avg < BACK_ANGLE_THRESHOLD:
                form_error_detected = True
                if "Forward lean" not in current_set_errors:
                    current_set_errors.append("Forward lean")
                    bad_form_counter += 1
                    
                    # Voice feedback for bad form (not too frequent)
                    if bad_form_counter % 3 == 0:  # Every 3rd form error
                        speak("Keep your chest up! Back straight!")
        
        # ========== STATE MACHINE ==========
        if angle_smooth < ANGLE_SQUAT and state == "UP":
            state = "DOWN"
            # Check depth
            if angle_smooth > ANGLE_SQUAT + 10:
                if "Shallow squat" not in current_set_errors:
                    current_set_errors.append("Shallow squat")
                    speak("Go deeper for full range!")

        if angle_smooth > ANGLE_STAND and state == "DOWN":
            state = "UP"
            rep_count += 1
            last_rep_time = time.time()
            analysis_triggered = False
            
            # Reset bad form counter on successful rep
            if not form_error_detected:
                bad_form_counter = max(0, bad_form_counter - 1)
            
            # Periodic encouragement
            if rep_count % CHECK_INTERVAL == 0:
                encouragement = random.choice([
                    f"{rep_count} reps! Keep going!",
                    f"{rep_count} down! You're crushing it!",
                    f"Halfway to {rep_count * 2}! Stay strong!"
                ])
                speak(encouragement)
            
            print(f"REP: {rep_count} | Angle: {int(angle_smooth)}")

        # ========== REST DETECTION & AI ANALYSIS ==========
        if (time_since_last_rep > REST_TIMEOUT and 
            rep_count > 0 and 
            not analysis_triggered and
            state == "UP"):
            
            analysis_triggered = True
            speak("Set complete. Analyzing your performance...")
            
            # Get feedback (AI or fallback)
            feedback = get_smart_feedback(rep_count, current_set_errors)
            
            # Speak feedback
            time.sleep(1)  # Small pause
            speak(feedback)
            
            # Display on screen for 5 seconds
            analysis_start_time = time.time()
            while time.time() - analysis_start_time < 5:
                # Display analysis
                cv2.putText(frame, "ANALYSIS COMPLETE:", 
                           (30, 250), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, (0, 255, 255), 2)
                # Wrap text if long
                words = feedback.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line + word) < 30:
                        current_line += word + " "
                    else:
                        lines.append(current_line)
                        current_line = word + " "
                if current_line:
                    lines.append(current_line)
                
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, 
                               (30, 280 + i*30), cv2.FONT_HERSHEY_SIMPLEX,
                               0.6, (255, 255, 255), 2)
                
                cv2.imshow("Squat Counter (MediaPipe Pose)", frame)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
            
            # Reset for next set
            rep_count = 0
            current_set_errors = []
            last_rep_time = time.time()
            bad_form_counter = 0

        # ========== DRAW POSE ==========
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0, 255, 255), thickness=2),
            mp_draw.DrawingSpec(color=(255, 0, 255), thickness=2)
        )

        # ========== DISPLAY INFO ==========
        # Knee angle
        cv2.putText(frame, f"Knee Angle: {int(angle_smooth)}",
                   (30, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.9, (255, 255, 255), 2)
        
        # Back angle
        cv2.putText(frame, f"Back Angle: {int(back_angle_avg)}",
                   (30, 90), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (200, 200, 0), 2)
        
        # Form warnings
        if form_error_detected:
            cv2.putText(frame, "FORM: Leaning Forward!",
                       (30, 220), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 100, 255), 2)
        elif bad_form_counter > 0:
            cv2.putText(frame, f"Form errors: {bad_form_counter}",
                       (30, 220), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (255, 200, 0), 2)

    # ========== UI ==========
    # Rep count (always visible)
    cv2.putText(frame, f"REPS: {rep_count}",
               (30, 120), cv2.FONT_HERSHEY_SIMPLEX,
               1.3, (0, 255, 0), 3)
    
    # State
    cv2.putText(frame, f"STATE: {state}",
               (30, 170), cv2.FONT_HERSHEY_SIMPLEX,
               0.9, (0, 200, 255), 2)
    
    # Rest timer
    cv2.putText(frame, f"Rest: {int(time_since_last_rep)}s",
               (frame.shape[1] - 150, 40), cv2.FONT_HERSHEY_SIMPLEX,
               0.7, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit",
               (frame.shape[1] - 200, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)

    # Show frame
    cv2.imshow("Squat Counter (MediaPipe Pose)", frame)

    # Quit on 'q' press
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
cap.release()
cv2.destroyAllWindows()
print(f"\nSession ended. Final reps: {rep_count}")
if current_set_errors:
    print(f"Form issues to work on: {', '.join(current_set_errors)}")
print("Keep training!")
