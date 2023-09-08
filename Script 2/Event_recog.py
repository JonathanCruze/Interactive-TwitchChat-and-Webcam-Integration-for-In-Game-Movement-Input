import cv2
import pytesseract
import pygetwindow as gw
import numpy as np
import mss
import csv
import time
from difflib import SequenceMatcher
from Twt import Twtbot

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

game_window_name = "VisualBoyAdvance-M"
game_window = gw.getWindowsWithTitle(game_window_name)[0]

# Open the CSV file in write mode
csv_file = open('Detect_object\Script 2\captured_text.csv', 'w', newline='', encoding='utf-8')
csv_writer = csv.writer(csv_file)

# Write the CSV header
csv_writer.writerow(['Frame', 'Extracted Text', 'Best Match'])

# Keywords to recognize and train
keywords = ['was defeated', 'enemy', 'fainted', 'gotcha', 'Gotcha']

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def update_text_and_frame():

    global frame_counter
    while True:
        left, top, width, height = game_window.left, game_window.top, game_window.width, game_window.height
        screenshot = np.array(sct.grab({"left": left, "top": top, "width": width, "height": height}))

        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        extracted_text = pytesseract.image_to_string(gray_frame)

        # Calculate the similarity of the keywords with the extracted text
        best_match = max(keywords, key=lambda keyword: similar(extracted_text.lower(), keyword.lower()))

        # Add the extracted text and the best result to the CSV file
        csv_writer.writerow([frame_counter, extracted_text, best_match])

        # Take a screenshot when a certain keyword appears on the screen
        if any(keyword in extracted_text for keyword in keywords):
            #cv2.imwrite(f'screenshot_{frame_counter}.png', frame)
            screenshot_path = 'Detect_object\Script 2\screenshot.png'
            cv2.imwrite(screenshot_path, frame)
            twt_bot = Twtbot()
            twt_bot.run(screenshot_path)  # Run Twtbot when keyword is detected

        frame_counter += 1
        time.sleep(1)  # Optional pause to control capture frequency

# Create the MSS object for screen capture
with mss.mss() as sct:
    frame_counter = 0
    update_text_and_frame()