# Interactive-TwitchChat-and-Webcam-Integration-for-In-Game-Movement-Input-in-Pokemon-Videogames



## Overview
This project consists of two Python scripts designed for different tasks: one for interacting with a Twitch stream using YOLO for people detection to play Pokémon, and the other for recognizing events using Tesseract OCR and uploading screenshots of tweets to Twitter using Selenium.

### Script 1: Twitch Pokémon Interaction
- **Description:** This script utilizes the YOLO (You Only Look Once) model to detect students using webcam to use it as inputs in the pokémon game, and it extracts the chat from the Twitch stream using the TwitchIO library and uses this information to interact with viewers and use the twitch chat commands as inputs.
- **Dependencies:**
  - YOLO model v5
  - TwitchIO library (for extracting the Twitch chat)
- **Usage:**
  - Ensure you have the YOLO model weights and configuration file in the appropriate directory.
  - Set up your Twitch credentials and channel information.
  - Run the script, which will connect to your Twitch channel, detect people with a webcam, extract the twitch chat commands  and interact with viewers.

### Script 2: Twitter Event Recognition
- **Description:** This script uses Tesseract OCR to recognize events from screenshots like a 'was defeated', 'enemy', 'fainted', 'gotcha', and then uploads these screenshots to Twitter using a Twitter bot implemented with Selenium.
- **Dependencies:**
  - Tesseract OCR
  - Selenium (with WebDriver for your preferred browser)
- **Usage:**
  - Ensure you have Tesseract OCR installed and set up properly.
  - Configure the Selenium WebDriver for your desired browser.
  - Run the script, which will monitor for new screenshots, perform OCR to extract events (tweets), and upload the screenshot and event to Twitter.
 ---
## Troubleshooting and Support
If you encounter any issues or have questions related to this project, please feel free to reach out to the project maintainers.

## Contributions
Contributions to this project are welcome! If you have any improvements or new features to add, please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License.
