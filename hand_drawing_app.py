"""
Hand Drawing App - Hand Detection Only
This application uses MediaPipe Hands to detect ONLY hands and fingers.
It does NOT detect faces or any other objects.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
try:
    from config import *
except ImportError:
    # Default values if config.py is not found
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    CAMERA_INDEX = 0
    BRUSH_SIZE = 10
    ERASER_SIZE = 100
    CANVAS_ALPHA = 0.7
    COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    HAND_DETECTION_CONFIDENCE = 0.8  # Increased confidence threshold
    HAND_TRACKING_CONFIDENCE = 0.7   # Increased tracking confidence
    MAX_NUM_HANDS = 1
    POSITION_HISTORY_LENGTH = 5
    FPS_UPDATE_INTERVAL = 1.0
    SHOW_FPS = True
    SHOW_MODE_INDICATOR = True
    SHOW_COLOR_INDICATOR = True
    SHOW_INSTRUCTIONS = True
    GESTURE_COOLDOWN = 0.3
    FINGER_EXTENSION_THRESHOLD = 0.02
    DEBUG_MODE = False
    SHOW_HAND_LANDMARKS = True
    SHOW_FINGER_STATES = False

class HandDrawingApp:
    def __init__(self):
        # Initialize MediaPipe - ONLY for hands
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Configure hands detection with stricter parameters
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
            model_complexity=1  # Use model complexity 1 for better accuracy
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Canvas and drawing variables
        self.canvas = None
        self.prev_point = None
        self.drawing_mode = False
        self.erasing_mode = False
        self.hover_mode = False
        self.bucket_armed = False
        self.last_bucket_time = 0.0
        
        # Colors and drawing settings
        self.colors = COLORS
        self.current_color_index = 0
        self.current_color = self.colors[self.current_color_index]
        self.brush_size = BRUSH_SIZE
        self.eraser_size = ERASER_SIZE
        
        # Position tracking
        self.prev_positions = deque(maxlen=POSITION_HISTORY_LENGTH)
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Hand validation
        self.hand_detection_frames = 0
        self.min_hand_frames = 3  # Require hand to be detected for multiple frames
        
    def validate_hand_detection(self, hand_landmarks, frame_shape):
        """
        Validate that the detected landmarks actually represent a hand.
        This function adds extra checks to filter out false positives like a face.
        """
        if hand_landmarks is None:
            return False
            
        # Get landmark coordinates as a list of tuples (x, y)
        landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
        
        h, w = frame_shape[:2]
        
        # Check if all landmarks are within the frame bounds
        if not all(0 <= x <= 1 and 0 <= y <= 1 for x, y in landmarks):
            return False
        
        # Calculate the size of the detected object based on the wrist and finger tips.
        # This helps to filter out very small or very large objects.
        wrist = landmarks[0]  # Wrist landmark
        middle_tip = landmarks[12]  # Middle finger tip
        
        # Calculate the distance between the wrist and the middle finger tip
        hand_size = np.sqrt((wrist[0] - middle_tip[0])**2 + (wrist[1] - middle_tip[1])**2)
        
        # A hand should have a reasonable size relative to the frame.
        # These thresholds (0.1 and 0.8) are based on typical webcam usage.
        if hand_size < 0.1 or hand_size > 0.8:
            return False
        
        # Check the position of the object. Faces are typically in the upper half
        # of the frame, while hands are more flexible. We can check if the hand
        # is at a very high Y-coordinate, which is a common indicator of a face.
        hand_center_y = (wrist[1] + middle_tip[1]) / 2
        # If the center of the hand is too high, it might be a face.
        if hand_center_y < 0.1:  # This threshold might need adjustment based on camera angle
            return False
            
        return True
    
    def get_finger_states(self, hand_landmarks):
        """Determine which fingers are extended based on hand landmarks"""
        # Get landmark coordinates
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        
        # Define finger tip and pip (middle joint) indices
        finger_tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        finger_pips = [3, 6, 10, 14, 18]  # corresponding middle joints
        
        finger_states = []
        
        for i, (tip, pip) in enumerate(zip(finger_tips, finger_pips)):
            if i == 0:  # Thumb (special case - check horizontal position)
                if landmarks[tip][0] < landmarks[pip][0] - FINGER_EXTENSION_THRESHOLD:  # Thumb extended left
                    finger_states.append(True)
                else:
                    finger_states.append(False)
            else:  # Other fingers (check vertical position)
                if landmarks[tip][1] < landmarks[pip][1] - FINGER_EXTENSION_THRESHOLD:  # Finger extended up
                    finger_states.append(True)
                else:
                    finger_states.append(False)
        
        return finger_states
    
    def determine_gesture(self, finger_states):
        """Determine the current gesture based on finger states"""
        thumb, index, middle, ring, pinky = finger_states
        
        # All fingers extended = Erase mode
        if all(finger_states):
            return "erase"
        # Fist (no fingers) = Bucket arm
        elif not any(finger_states):
            return "bucket_arm"
        # Only index finger = Drawing mode
        elif index and not middle and not ring and not pinky:
            return "draw"
        # Index and middle fingers = Hover mode
        elif index and middle and not ring and not pinky:
            return "hover"
        # Only thumb = Color change
        elif thumb and not index and not middle and not ring and not pinky:
            return "color_change"
        else:
            return "none"
    
    def get_finger_position(self, hand_landmarks, frame_shape):
        """Get the position of the index finger tip in frame coordinates"""
        index_tip = hand_landmarks.landmark[8]  # Index finger tip
        h, w = frame_shape[:2]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        return (x, y)
    
    def change_color(self):
        """Cycle to the next color in the color list"""
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        self.current_color = self.colors[self.current_color_index]
        print(f"Color changed to: {self.current_color}")
    
    def draw_on_canvas(self, position):
        """Draw on the canvas at the given position"""
        if self.prev_point is not None:
            cv2.line(self.canvas, self.prev_point, position, self.current_color, self.brush_size)
        self.prev_point = position
        self.prev_positions.append(position)
    
    def erase_from_canvas(self, position):
        """Erase from the canvas at the given position"""
        cv2.circle(self.canvas, position, self.eraser_size, (0, 0, 0), -1)
    
    def bucket_fill_canvas(self, seed_point):
        """Flood-fill the closed region on the canvas at the given seed point with the current color."""
        if seed_point is None:
            return
        h, w = self.canvas.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        # Tolerance for color similarity when filling
        lo_diff = (10, 10, 10, 10)
        up_diff = (10, 10, 10, 10)
        # Clamp seed inside bounds
        x = max(0, min(w - 1, seed_point[0]))
        y = max(0, min(h - 1, seed_point[1]))
        # Perform flood fill on the canvas layer only
        try:
            cv2.floodFill(self.canvas, mask, (x, y), self.current_color, loDiff=lo_diff, upDiff=up_diff, flags=4)
        except Exception as _:
            # In rare cases floodFill can fail if mask/seed invalid; ignore gracefully
            pass
    
    def clear_canvas(self):
        """Clear the entire canvas"""
        self.canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        self.prev_point = None
        self.prev_positions.clear()
    
    def save_drawing(self):
        """Save the current drawing to a file"""
        timestamp = time.strftime(SAVE_TIMESTAMP_FORMAT)
        filename = f"{SAVE_PREFIX}{timestamp}.{SAVE_FORMAT.lower()}"
        cv2.imwrite(filename, self.canvas)
        print(f"Drawing saved as: {filename}")
    
    def draw_ui_overlay(self, frame):
        """Draw UI elements on the frame"""
        y_offset = 10
        
        # Draw current color indicator
        if SHOW_COLOR_INDICATOR:
            color_display = np.zeros((60, 200, 3), dtype=np.uint8)
            color_display[:] = self.current_color
            cv2.putText(color_display, f"Color {self.current_color_index + 1}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            frame[y_offset:y_offset+60, 10:210] = color_display
            y_offset += 70
        
        # Draw mode indicator
        if SHOW_MODE_INDICATOR:
            mode_text = f"Mode: {self.drawing_mode}"
            if self.erasing_mode:
                mode_text = "Mode: Erase"
            elif self.hover_mode:
                mode_text = "Mode: Hover"
            
            cv2.putText(frame, mode_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            y_offset += 30
            
            # Show hand detection status
            if self.hand_detection_frames >= self.min_hand_frames:
                status_text = "Hand Detected ✓"
                status_color = (0, 255, 0)  # Green
            else:
                status_text = "No Hand Detected"
                status_color = (0, 0, 255)  # Red
            
            cv2.putText(frame, status_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, status_color, 2)
            y_offset += 30
        
        # Draw FPS
        if SHOW_FPS:
            self.fps_counter += 1
            if time.time() - self.fps_start_time >= FPS_UPDATE_INTERVAL:
                fps = self.fps_counter / (time.time() - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = time.time()
            else:
                fps = 0.0  # Default value if FPS not yet calculated
            
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            y_offset += 30
        
        # Draw instructions
        if SHOW_INSTRUCTIONS:
            instructions = [
                "HAND DETECTION ONLY - No Face Detection",
                "Controls:",
                "Index finger: Draw",
                "Index + Middle: Hover",
                "All fingers: Erase",
                "Thumb only: Change color",
                "C: Clear canvas",
                "S: Save drawing",
                "Q: Quit"
            ]
            
            for i, instruction in enumerate(instructions):
                y_pos = y_offset + i * 20
                cv2.putText(frame, instruction, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (200, 200, 200), 1)
    
    def run(self):
        """Main application loop"""
        print("Hand Drawing App Started!")
        print("Press 'Q' to quit, 'C' to clear canvas, 'S' to save drawing")
        
        # Initialize canvas
        self.canvas = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame to detect hands
            results = self.hands.process(rgb_frame)
            
            # Reset modes
            self.drawing_mode = False
            self.erasing_mode = False
            self.hover_mode = False
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Validate that this is actually a hand, not a face or other object
                    if not self.validate_hand_detection(hand_landmarks, frame.shape):
                        continue  # Skip this detection if it doesn't look like a hand
                    
                    # Increment hand detection counter
                    self.hand_detection_frames += 1
                    
                    # Only process hand if it's been detected for multiple frames (stability check)
                    if self.hand_detection_frames < self.min_hand_frames:
                        continue
                    
                    # Draw hand landmarks
                    if SHOW_HAND_LANDMARKS:
                        self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Get finger states and determine gesture
                    finger_states = self.get_finger_states(hand_landmarks)
                    gesture = self.determine_gesture(finger_states)
                    
                    # Debug information
                    if DEBUG_MODE and SHOW_FINGER_STATES:
                        finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
                        debug_text = f"Gesture: {gesture}"
                        for i, (name, state) in enumerate(zip(finger_names, finger_states)):
                            debug_text += f" | {name}: {'✓' if state else '✗'}"
                        cv2.putText(frame, debug_text, (10, CAMERA_HEIGHT - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Get finger position
                    position = self.get_finger_position(hand_landmarks, frame.shape)
                    
                    # Handle different gestures
                    if gesture == "draw":
                        self.drawing_mode = True
                        self.draw_on_canvas(position)
                        # Draw cursor
                        cv2.circle(frame, position, 8, self.current_color, -1)
                        
                    elif gesture == "hover":
                        self.hover_mode = True
                        # Draw cursor
                        cv2.circle(frame, position, 8, (255, 255, 255), 2)
                        self.prev_point = None  # Don't connect lines in hover mode
                        
                    elif gesture == "erase":
                        self.erasing_mode = True
                        self.erase_from_canvas(position)
                        # Draw eraser indicator
                        cv2.circle(frame, position, self.eraser_size, (0, 0, 255), 2)
                        
                    elif gesture == "color_change":
                        self.change_color()
                        time.sleep(GESTURE_COOLDOWN)  # Prevent rapid color changes
                    
                    # Draw trail for smooth drawing
                    if len(self.prev_positions) > 1 and self.drawing_mode:
                        for i in range(1, len(self.prev_positions)):
                            cv2.circle(frame, self.prev_positions[i], 3, self.current_color, -1)
            else:
                # Reset hand detection counter when no hands are detected
                self.hand_detection_frames = 0
            
            # Overlay canvas on frame
            # Create a mask for non-black pixels in canvas
            canvas_mask = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
            _, canvas_mask = cv2.threshold(canvas_mask, 1, 255, cv2.THRESH_BINARY)
            canvas_mask = cv2.cvtColor(canvas_mask, cv2.COLOR_GRAY2BGR)
            
            # Apply canvas to frame
            frame = cv2.addWeighted(frame, 1.0, self.canvas, CANVAS_ALPHA, 0)
            
            # Draw UI overlay
            self.draw_ui_overlay(frame)
            
            # Display the frame
            cv2.imshow('Hand Drawing App', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                self.clear_canvas()
            elif key == ord('s') or key == ord('S'):
                self.save_drawing()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    try:
        app = HandDrawingApp()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have a webcam connected and the required packages installed.")
