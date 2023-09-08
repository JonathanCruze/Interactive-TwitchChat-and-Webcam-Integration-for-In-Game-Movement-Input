import cv2
import numpy as np
import random
from ultralytics import YOLO
import pydirectinput
import time
import threading

MODEL_PATH: str = '../yolov5nu.pt'

# Number of rows and columns in the grid
GRID_ROWS: int = 2
GRID_COLS: int = 5

# Display names and corresponding keyboard keys
# [(Displayed Text', 'Keyboard Key')]
DISPLAY_NAMES_AND_KEYS: list[(str,str)] = [
    ("Arriba", "up"),
    ("Abajo", "down"),
    ("Tecla A", "a"),
    ("Tecla B", "b"),
    ("Start", "x"),
    ("Derecha", "right"),
    ("Izquierda", "left"),
    ("Tecla R", "r"),
    ("Tecla L", "l"),
    ("Select", "z"),
]

GREEN: tuple = (0, 255, 0)  # Detected target text color
BLUE: tuple = (255, 0, 0)  # Person counter color
BLACK: tuple = (0, 0, 0)  # Default text color
GRAY: tuple = (192, 192, 192)  # Background text color


########################################################################################################
## Script Functions 
########################################################################################################

def get_centroid(box: np.ndarray) -> tuple:
    """
    Calculate the centroid (center point) of a bounding box.

    Parameters:
        box (np.ndarray): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        x_center, y_center (tuple): x and y coordinates of the centroid.
    """
    x1, y1, x2, y2 = box[:4]
    x_center: int = int((x1 + x2) / 2)
    y_center: int = int((y1 + y2) / 2)
    return x_center, y_center


def get_quadrant_indices(x: int, y: int, width: int, height: int) -> tuple:
    """
    Get the row and column indices of the quadrant in the drown grid.
    
    Parameters:
        x (int): x-coordinate of the point.
        y (int): y-coordinate of the point.
        width (int): Width of the captured frame.
        height (int): Height of the captured frame.

    Returns:
        row, column (tuple): Row and column indices of the quadrant in the grid.
    """
    quadrant_width: int = width // GRID_COLS
    quadrant_height: int = height // GRID_ROWS

    row: int = y // quadrant_height
    col: int = x // quadrant_width
    return row, col


def draw_grid_lines(frame: np.ndarray) -> None:
    """
    Draw grid lines on the frame.

    Parameters:
        frame (np.ndarray): The input frame to draw the grid lines on.
    """
    frame_height, frame_width = frame.shape[:2]
    cell_width: int = frame_width // GRID_COLS
    cell_height: int = frame_height // GRID_ROWS

    for row in range(1, GRID_ROWS):
        cv2.line(frame, (0, row * cell_height), (frame_width, row * cell_height), BLACK, 1)

    for col in range(1, GRID_COLS):
        cv2.line(frame, (col * cell_width, 0), (col * cell_width, frame_height), BLACK, 1)

def draw_on_grid(frame: np.ndarray, show_labels=True, person_counter: np.ndarray = None) -> None:
    """
    Draw quadrant labels, custom text, and person counter on the frame.

    Parameters:
        frame (np.ndarray): The input frame to draw on.
        show_labels (bool): Whether to display quadrant labels. Default is True.
        person_counter (np.ndarray): A 2D array representing the position of persons detected on a quadrant.
    """
    frame_height, frame_width = frame.shape[:2]
    cell_width: int = frame_width // GRID_COLS
    cell_height: int = frame_height // GRID_ROWS
    text_scale = 0.5

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * cell_width
            y = row * cell_height

            quadrant_label = f"({row},{col})"
            quadrant_text = tuple(name for name, _ in DISPLAY_NAMES_AND_KEYS)[row * GRID_COLS + col]
            persons_detected = person_counter[row, col] if person_counter is not None else 0

            text_x = x + (cell_width - cv2.getTextSize(quadrant_label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, 1)[0][0]) // 2
            text_y = y + cell_height // 2

            text_color = GREEN if (row, col) == get_most_people_quadrant(person_counter) else BLACK

            if show_labels:
                cv2.putText(frame, quadrant_label, (text_x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, GRAY, 1)
            cv2.putText(frame, quadrant_text, (text_x, text_y), cv2.FONT_HERSHEY_TRIPLEX, text_scale, text_color, 1)
            cv2.putText(frame, str(persons_detected), (x + cell_width - 30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, text_scale, BLUE, 2)
            

def get_most_people_quadrant(person_counter: np.ndarray) -> tuple:
    """
    Get the indices of the quadrant with the most people.
    If a tie occurs: Randomly choose one of the quadrants.
    If no people are detected: Return a tuple (99, 99)

    Parameters:
        person_counter (np.ndarray): A 2D array representing the number of persons detected in each quadrant.

    Returns:
        tuple: Row and column indices of the quadrant with the most people, or None if no people are detected.
    """
    total_people = np.sum(person_counter)
    
    if total_people == 0:
        print("No people being detected")
        return 99, 99
    else:
        return np.unravel_index(np.argmax(person_counter, axis=None), person_counter.shape)


def _send_key(key):
    """
    Simulate a key press using pydirectinput.
    """
    pydirectinput.keyDown(key)
    time.sleep(0.1)  # Adjust the delay as needed
    pydirectinput.keyUp(key)


def send_key_based_on_quadrant(grid_section: tuple) -> None:
    """x
    Send keyboard inputs based on the grid section using pydirectinput.

    Parameters:
        grid_section (tuple): Row and column indices of the quadrant with the most people.
        key_mapping (dictionary): Grid Section tuples as keys and its corresponding keyboard keys as values
    """
    key_mapper: dict[tuple[int, int], str] = {(i // GRID_COLS, i % GRID_COLS): key for i, (_, key) in enumerate(DISPLAY_NAMES_AND_KEYS)}

    if grid_section == (99, 99):
        available_keys = list(key_mapper.values())
        key = random.choice(available_keys)
    else:
        key = key_mapper.get(grid_section)

    if key is not None:
        _send_key(key)  # Use the custom send_key function
        time.sleep(0.06)  # 0.02 seconds delay

def validate_model_configuration():
    """
    Simple Exception Handling block 
    """
    # Check if the number of keys matches the grid dimensions
    if len(DISPLAY_NAMES_AND_KEYS) != GRID_ROWS * GRID_COLS:
        raise ValueError("Number of keys in 'DISPLAY_NAMES_AND_KEYS' doesn't match 'GRID_ROWS' and 'GRID_COLS' dimensions.")

    VALID_KEYBOARD_KEYS = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m","n","o", "p",
                           "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "up", "down", "left", "right",
                           "space", "enter", "backspace", "tab", "shift", "ctrl", "alt", "esc", "win", "menu","capslock", "delete",
                           "numlock", "printscreen", "f1", "f2", "f3", "f4", "f5","f6", "f7", "f8", "f9", "f10", "f11", "f12"}
    # Check if the keyboard keys are valid
    invalid_keys = [key for _, key in DISPLAY_NAMES_AND_KEYS if key not in VALID_KEYBOARD_KEYS]
    if invalid_keys:
        raise ValueError(f"Invalid keys found in 'DISPLAY_NAMES_AND_KEYS' variable: {', '.join(invalid_keys)}\List of valid keys: {VALID_KEYBOARD_KEYS}")

class Camera(object):
    """
    Base Camera object
    """

    def __init__(self):
        self._cam = None
        self._frame = None
        self._frame_width = None
        self._frame_height = None
        self._ret = False

        self.auto_undistortion = False
        self._camera_matrix = None
        self._distortion_coefficients = None

        self._is_running = False

    def _init_camera(self):
        """
        This is the first for creating our camera
        We should override this!
        """

        pass

    def start_camera(self):
        """
        Start the running of the camera, without this we can't capture frames
        Camera runs on a separate thread so we can reach a higher FPS
        """

        self._init_camera()
        self._is_running = True
        threading.Thread(target=self._update_camera, args=()).start()

    def _read_from_camera(self):
        """
        This method is responsible for grabbing frames from the camera
        We should override this!
        """

        if self._cam is None:
            raise Exception("Camera is not started!")

    def _update_camera(self):
        """
        Grabs the frames from the camera
        """

        while True:
            if self._is_running:
                self._ret, self._frame = self._read_from_camera()
            else:
                break

    def get_frame_width_and_height(self):
        """
        Returns the width and height of the grabbed images
        :return (int int): width and height
        """

        return self._frame_width, self._frame_height

    def read(self):
        """
        With this you can grab the last frame from the camera
        :return (boolean, np.arr ay): return value and frame
        """
        return self._ret, self._frame

    def release_camera(self):
        """
        Stop the camera
        """

        self._is_running = False

    def is_running(self):
        return self._is_running

    def set_calibration_matrices(self, camera_matrix, distortion_coefficients):
        self._camera_matrix = camera_matrix
        self._distortion_coefficients = distortion_coefficients

    def activate_auto_undistortion(self):
        self.auto_undistortion = True

    def deactivate_auto_undistortion(self):
        self.auto_undistortion = False

    def _undistort_image(self, image):
        if self._camera_matrix is None or self._distortion_coefficients is None:
            import warnings
            warnings.warn("Undistortion has no effect because <camera_matrix>/<distortion_coefficients> is None!")
            return image

        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self._camera_matrix,
                                                               self._distortion_coefficients, (w, h),
                                                               1,
                                                               (w, h))
        undistorted = cv2.undistort(image, self._camera_matrix, self._distortion_coefficients, None,
                                    new_camera_matrix)
        return undistorted


class WebCamera(Camera):
    """
    Simple Webcamera
    """

    def __init__(self, video_src=0):
        """
        :param video_src (int): camera source code (it should be 0 or 1, or the filename)
        """

        super().__init__()
        self._video_src = video_src

    def _init_camera(self):
        super()._init_camera()
        self._cam = cv2.VideoCapture(self._video_src)
        self._ret, self._frame = self._cam.read()
        if not self._ret:
            raise Exception("No camera feed")
        self._frame_height, self._frame_width, c = self._frame.shape
        return self._ret

    def _read_from_camera(self):
        super()._read_from_camera()
        self._ret, self._frame = self._cam.read()
        if self._ret:
            if self.auto_undistortion:
                self._frame = self._undistort_image(self._frame)
            return True, self._frame
        else:
            return False, None

    def release_camera(self):
        super().release_camera()
        self._cam.release()

lock = threading.Lock()
class DetectionRenderingThread(threading.Thread):
    def __init__(self, model, cap):
        super().__init__()
        self.model = model
        self.cap = cap
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:  # Regularly check the flag to determine if the thread should keep running
            ret, frame = self.cap.read()

            if not ret:
                raise RuntimeError("Error: Failed to grab a frame.")

            results = self.model(frame, classes=[0], conf=0.51, iou=0.55, imgsz=320)

            person_counter = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
            for result in results:
                for box in result.boxes.data:
                    x_center, y_center = get_centroid(box)
                    row, col = get_quadrant_indices(x_center, y_center, frame.shape[1], frame.shape[0])

                    cv2.circle(frame, (x_center, y_center), 5, BLUE, -1)

                    person_counter[row, col] += 1

            with lock:
                draw_on_grid(frame, show_labels=True, person_counter=person_counter)
                send_key_based_on_quadrant(get_most_people_quadrant(person_counter))

                draw_grid_lines(annotated_frame := results[0].plot(conf=False))
                cv2.imshow(window_name := "Artek Institute Plays Pokemon", annotated_frame)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            if cv2.waitKey(1) & 0xFF == ord('|'):
                break

        self.cap.release_camera()

########################################################################################################
## MAIN Function 
########################################################################################################
def main_obj():
    try:
        model = YOLO(MODEL_PATH)
        model.names[0] = 'Estudiante'

        cap = WebCamera(0)
        cap.start_camera()

        detection_thread = DetectionRenderingThread(model, cap)
        detection_thread.start()

        while True:
            if cv2.waitKey(1) & 0xFF == ord('|'):
                detection_thread.stop()  # Signal the detection thread to stop
                detection_thread.join()  # Wait for the thread to finish before exiting
                break

    except Exception as e:
        print(str(e))
        
    finally:
        cap.release_camera()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    validate_model_configuration()
    main_obj()