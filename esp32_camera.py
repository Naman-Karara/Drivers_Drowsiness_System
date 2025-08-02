import requests
import cv2
import time  # Added this import
import numpy as np
from io import BytesIO

class ESP32Camera:
    def __init__(self, ip_address):
        self.stream_url = f"http://{ip_address}/capture"
        self.alarm_pin_url = f"http://{ip_address}/control"
        self.last_frame_time = 0
        self.frame_interval = 0.05
        
    def capture_frame(self):
        try:
            current_time = time.time()
            if current_time - self.last_frame_time < self.frame_interval:
                return False, None
                
            response = requests.get(self.stream_url, timeout=2)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.last_frame_time = current_time
                return True, frame
            return False, None
        except Exception as e:
            print(f"Camera capture error: {str(e)}")
            return False, None
    
    def trigger_alarm(self, state=True):
        try:
            pin_state = 1 if state else 0
            requests.get(f"{self.alarm_pin_url}?pin=4&state={pin_state}", timeout=1)
            return True
        except Exception as e:
            print(f"Alarm trigger error: {str(e)}")
            return False

    def release(self):
        pass