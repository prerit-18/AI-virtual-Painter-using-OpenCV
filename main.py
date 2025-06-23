import cv2
import mediapipe as mp
import numpy as np
import os
import math
import time
from collections import deque
import speech_recognition as sr
import pyttsx3
import threading
import queue

# Voice Assistant Setup
class VoiceAssistant:
    def __init__(self):
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.8)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Command queue for processing
        self.command_queue = queue.Queue()
        self.is_listening = False
        self.voice_thread = None
        self.wake_word = "jarvis"
        self.name = "Jarvis"
        
        # Available commands
        self.commands = {
            'clear': 'clear_canvas',
            'erase': 'clear_canvas',
            'red': 'select_color_red',
            'blue': 'select_color_blue', 
            'green': 'select_color_green',
            'black': 'select_color_black',
            'eraser': 'select_eraser',
            'rectangle': 'select_rectangle',
            'circle': 'select_circle',
            'line': 'select_line',
            'draw': 'select_draw',
            'help': 'show_help',
            'quit': 'quit_app',
            'exit': 'quit_app'
        }
        
        # Adjust for ambient noise
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    
    def speak(self, text):
        """Convert text to speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def listen_for_commands(self):
        """Listen for voice commands in a separate thread"""
        self.is_listening = True
        while self.is_listening:
            try:
                with self.microphone as source:
                    print(f"🎤 {self.name} is listening...")
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                
                try:
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"🎤 Heard: {command}")
                    
                    # Check if wake word is present
                    if self.wake_word in command:
                        # Extract the actual command after the wake word
                        command_parts = command.split(self.wake_word, 1)
                        if len(command_parts) > 1:
                            actual_command = command_parts[1].strip()
                            if actual_command:  # Only process if there's a command after the wake word
                                print(f"🎤 {self.name} processing: {actual_command}")
                                self.command_queue.put(actual_command)
                        else:
                            # Just the wake word was said
                            self.speak("Yes, how can I help you?")
                    else:
                        # Wake word not detected, ignore the command
                        pass
                        
                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    print(f"Speech recognition error: {e}")
                    
            except Exception as e:
                if "timeout" not in str(e).lower():
                    print(f"Voice listening error: {e}")
    
    def start_listening(self):
        """Start the voice listening thread"""
        self.voice_thread = threading.Thread(target=self.listen_for_commands, daemon=True)
        self.voice_thread.start()
        self.speak(f"{self.name} activated. Say {self.name} followed by your command.")
    
    def stop_listening(self):
        """Stop the voice listening thread"""
        self.is_listening = False
        if self.voice_thread:
            self.voice_thread.join(timeout=1)
    
    def process_command(self, command):
        """Process voice commands and return action"""
        for cmd, action in self.commands.items():
            if cmd in command:
                return action
        return None

# Initialize voice assistant
voice_assistant = VoiceAssistant()

# Global variables for voice commands
global selected_button, drawColor, selected_tool, imgCanvas, xp, yp

def handle_voice_command(command):
    """Handle voice commands and update application state"""
    global selected_button, drawColor, selected_tool, imgCanvas, xp, yp
    
    action = voice_assistant.process_command(command)
    if action:
        if action == 'clear_canvas':
            imgCanvas = np.zeros((height, width, 3), np.uint8)
            xp, yp = [0, 0]
            voice_assistant.speak("Canvas cleared")
            
        elif action == 'select_color_red':
            selected_button = 0
            drawColor = button_colors[0][0]
            selected_tool = 'Draw'
            voice_assistant.speak("Red color selected")
            
        elif action == 'select_color_blue':
            selected_button = 1
            drawColor = button_colors[1][0]
            selected_tool = 'Draw'
            voice_assistant.speak("Blue color selected")
            
        elif action == 'select_color_green':
            selected_button = 2
            drawColor = button_colors[2][0]
            selected_tool = 'Draw'
            voice_assistant.speak("Green color selected")
            
        elif action == 'select_color_black':
            selected_button = 3
            drawColor = button_colors[3][0]
            selected_tool = 'Draw'
            voice_assistant.speak("Black color selected")
            
        elif action == 'select_eraser':
            selected_button = 3
            drawColor = button_colors[3][0]
            selected_tool = 'Eraser'
            voice_assistant.speak("Eraser selected")
            
        elif action == 'select_rectangle':
            selected_button = 4
            selected_tool = 'Rectangle'
            voice_assistant.speak("Rectangle tool selected")
            
        elif action == 'select_circle':
            selected_button = 5
            selected_tool = 'Circle'
            voice_assistant.speak("Circle tool selected")
            
        elif action == 'select_line':
            selected_button = 6
            selected_tool = 'Line'
            voice_assistant.speak("Line tool selected")
            
        elif action == 'select_draw':
            selected_button = 0
            selected_tool = 'Draw'
            voice_assistant.speak("Draw mode selected")
            
        elif action == 'show_help':
            help_text = "Available voice commands: clear, red, blue, green, black, eraser, rectangle, circle, line, draw, help, quit"
            voice_assistant.speak(help_text)
            
        elif action == 'quit_app':
            voice_assistant.speak("Goodbye!")
            return 'quit'
    
    return None

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input (macOS compatible):
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 5)
width = 1280
height = 720
cap.set(3, width)
cap.set(4, height)

# Image that will contain the drawing and then passed to the camera image
imgCanvas = np.zeros((height, width, 3), np.uint8)

# Define color/eraser buttons (positions and colors)
button_width = 120
button_height = 90
button_y = 18
button_gap = 40
button_shapes = [
    ((128, 128, 128), 'Rectangle'),
    ((128, 128, 128), 'Circle'),
    ((128, 128, 128), 'Line')
]
button_colors = [
    ((0, 0, 255), 'Red'),
    ((255, 0, 0), 'Blue'),
    ((0, 255, 0), 'Green'),
    ((0, 0, 0), 'Eraser')
] + button_shapes
button_positions = []
for i in range(len(button_colors)):
    x1 = button_gap + i * (button_width + button_gap)
    x2 = x1 + button_width
    button_positions.append((x1, x2))

drawColor = button_colors[0][0]
selected_button = 0
thickness = 10 # Thickness of the painting
tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
xp, yp = [0, 0] # Coordinates that will keep track of the last position of the index finger

# For smoothing fingertip position
smooth_points = deque(maxlen=5)

# Debounce for color selection
selection_cooldown = 0.5  # seconds

# Add state variables
selected_tool = 'Draw'  # 'Draw', 'Eraser', 'Rectangle', 'Circle', 'Line'
shape_start = None
shape_preview = None

with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    global last_selection_time, was_hand_open
    last_selection_time = 0
    was_hand_open = False
    
    # Start voice assistant
    voice_assistant.start_listening()
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        # Process voice commands
        try:
            while not voice_assistant.command_queue.empty():
                command = voice_assistant.command_queue.get_nowait()
                result = handle_voice_command(command)
                if result == 'quit':
                    break
        except queue.Empty:
            pass

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Always draw the menu bar (header_area) at the top, even if no hand is detected
        header_area = np.ones((125, width, 3), np.uint8) * 240
        cv2.rectangle(header_area, (0, 0), (width, 125), (220, 220, 220), cv2.FILLED)
        
        # Draw voice assistant status
        voice_status = f"🎤 {voice_assistant.name} Active" if voice_assistant.is_listening else f"🎤 {voice_assistant.name} Inactive"
        cv2.putText(header_area, voice_status, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        for i, ((color, label), (x1b, x2b)) in enumerate(zip(button_colors, button_positions)):
            y1b, y2b = button_y, button_y + button_height
            # Button background with rounded corners
            button_bg = (255, 255, 255) if i != selected_button else (255, 255, 200)
            cv2.rectangle(header_area, (x1b, y1b), (x2b, y2b), button_bg, cv2.FILLED)
            # Button border
            border_color = (0, 255, 255) if i == selected_button else (180, 180, 180)
            cv2.rectangle(header_area, (x1b, y1b), (x2b, y2b), border_color, 3)
            # Color/eraser fill
            if label == 'Eraser':
                cv2.rectangle(header_area, (x1b+20, y1b+20), (x2b-20, y2b-20), (50, 50, 50), cv2.FILLED)
            else:
                cv2.rectangle(header_area, (x1b+20, y1b+20), (x2b-20, y2b-20), color, cv2.FILLED)
            # Label
            text_color = (0,0,0) if i != selected_button else (0,0,128)
            cv2.putText(header_area, label, (x1b+10, y2b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        image[0:125, 0:width] = header_area

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Getting all hand points coordinates
                points = []
                for lm in hand_landmarks.landmark:
                    points.append([int(lm.x * width), int(lm.y * height)])

                # Only go through the code when a hand is detected
                if len(points) != 0:
                    x1, y1 = points[8]  # Index finger
                    x2, y2 = points[12] # Middle finger
                    x3, y3 = points[4]  # Thumb
                    x4, y4 = points[20] # Pinky

                    # Smoothing fingertip position
                    smooth_points.append((x1, y1))
                    avg_x = int(sum([p[0] for p in smooth_points]) / len(smooth_points))
                    avg_y = int(sum([p[1] for p in smooth_points]) / len(smooth_points))
                    x1, y1 = avg_x, avg_y

                    ## Checking which fingers are up
                    fingers = []
                    # Checking the thumb
                    if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                    # The rest of the fingers
                    for id in range(1, 5):
                        if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    # Debug: Print finger states
                    print('Fingers:', fingers)

                    # Add a block to clear the canvas when all fingers are up (open palm)
                    if fingers == [1, 1, 1, 1, 1]:
                        imgCanvas = np.zeros((height, width, 3), np.uint8)
                        xp, yp = [x1, y1]

                    # 1. Inside the hand detection loop, after 'print('Fingers:', fingers)':
                    open_palm = (fingers == [1, 1, 1, 1, 1])
                    closed_fist = (fingers == [0, 0, 0, 0, 0])
                    if open_palm:
                        was_hand_open = True
                        hand_open_time = time.time()
                    elif closed_fist and was_hand_open:
                        # Allow a 1-second grace period after open palm
                        if (time.time() - hand_open_time) <= 1.0:
                            imgCanvas = np.zeros((height, width, 3), np.uint8)
                            xp, yp = [0, 0]
                            was_hand_open = False
                            print('Canvas cleared by gesture!')
                        else:
                            was_hand_open = False
                    elif not open_palm:
                        was_hand_open = False

                    ## Selection Mode - Two fingers are up
                    nonSel = [0, 3, 4] # indexes of the fingers that need to be down in the Selection Mode
                    if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                        xp, yp = [x1, y1]
                        # Debounced color/eraser selection
                        if(y1 < 125):
                            now = time.time()
                            for i, (bx1, bx2) in enumerate(button_positions):
                                if bx1 < x1 < bx2:
                                    if i != selected_button and (now - last_selection_time) > selection_cooldown:
                                        drawColor = button_colors[i][0]
                                        selected_button = i
                                        last_selection_time = now
                                        # Set tool
                                        label = button_colors[i][1]
                                        if label in ['Rectangle', 'Circle', 'Line']:
                                            selected_tool = label
                                        elif label == 'Eraser':
                                            selected_tool = 'Eraser'
                                        else:
                                            selected_tool = 'Draw'
                        # Draw selection rectangle
                        cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, 2)

                    ## Shape drawing logic (after gesture logic, before Draw Mode)
                    if selected_tool in ['Rectangle', 'Circle', 'Line']:
                        # Use index finger and thumb for shape drawing
                        # Check if index finger and thumb are close together (pinch gesture)
                        index_thumb_distance = np.linalg.norm(np.array(points[8]) - np.array(points[4]))
                        pinch_threshold = 50  # Distance threshold for pinch detection
                        
                        if index_thumb_distance < pinch_threshold and shape_start is None:
                            # Start drawing shape when fingers are pinched
                            shape_start = (x1, y1)
                        elif index_thumb_distance < pinch_threshold and shape_start is not None:
                            # Preview shape while fingers are pinched
                            shape_preview = (x1, y1)
                        elif index_thumb_distance >= pinch_threshold and shape_start is not None and shape_preview is not None:
                            # Draw shape when fingers are released
                            if selected_tool == 'Rectangle':
                                cv2.rectangle(imgCanvas, shape_start, shape_preview, drawColor, thickness)
                            elif selected_tool == 'Circle':
                                center = shape_start
                                radius = int(np.linalg.norm(np.array(shape_preview) - np.array(shape_start)))
                                cv2.circle(imgCanvas, center, radius, drawColor, thickness)
                            elif selected_tool == 'Line':
                                cv2.line(imgCanvas, shape_start, shape_preview, drawColor, thickness)
                            shape_start = None
                            shape_preview = None
                        
                        # Draw preview on image while fingers are pinched
                        if shape_start is not None and shape_preview is not None:
                            if selected_tool == 'Rectangle':
                                cv2.rectangle(image, shape_start, shape_preview, drawColor, thickness)
                            elif selected_tool == 'Circle':
                                center = shape_start
                                radius = int(np.linalg.norm(np.array(shape_preview) - np.array(shape_start)))
                                cv2.circle(image, center, radius, drawColor, thickness)
                            elif selected_tool == 'Line':
                                cv2.line(image, shape_start, shape_preview, drawColor, thickness)

                    ## Stand by Mode - Checking when the index and the pinky fingers are open and dont draw
                    nonStand = [0, 2, 3] # indexes of the fingers that need to be down in the Stand Mode
                    if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                        # The line between the index and the pinky indicates the Stand by Mode
                        cv2.line(image, (xp, yp), (x4, y4), drawColor, 5)
                        xp, yp = [x1, y1]

                    ## Draw Mode - One finger is up
                    nonDraw = [0, 2, 3, 4]
                    if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                        # The circle in the index finger indicates the Draw Mode
                        cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED)
                        if xp==0 and yp==0:
                            xp, yp = [x1, y1]
                        # Draw a line between the current position and the last position of the index finger
                        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                        # Update the last position
                        xp, yp = [x1, y1]

        # The image processing to produce the image of the camera with the draw made in imgCanvas
        imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(image, imgInv)
        img = cv2.bitwise_or(img, imgCanvas)

        cv2.imshow('MediaPipe Hands', img)
        key = cv2.waitKey(3) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            imgCanvas = np.zeros((height, width, 3), np.uint8)
            xp, yp = [0, 0]

# Cleanup
voice_assistant.stop_listening()
cap.release()
cv2.destroyAllWindows()