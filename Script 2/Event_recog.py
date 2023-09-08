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
keywords = ['Enemy', 'Gotcha']

pokemon_names_keywords = [
    'Chikorita', 'Bayleef', 'Meganium', 'Cyndaquil', 'Quilava', 'Typhlosion',
    'Totodile', 'Croconaw', 'Feraligatr', 'Sentret', 'Furret', 'Hoothoot',
    'Noctowl', 'Ledyba', 'Ledian', 'Spinarak', 'Ariados', 'Crobat',
    'Chinchou', 'Lanturn', 'Pichu', 'Cleffa', 'Igglybuff', 'Togepi',
    'Togetic', 'Natu', 'Xatu', 'Mareep', 'Flaaffy', 'Ampharos',
    'Bellossom', 'Marill', 'Azumarill', 'Sudowoodo', 'Politoed', 'Hoppip',
    'Skiploom', 'Jumpluff', 'Aipom', 'Sunkern', 'Sunflora', 'Yanma',
    'Wooper', 'Quagsire', 'Espeon', 'Umbreon', 'Murkrow', 'Slowking',
    'Misdreavous', 'Unown', 'Wobbuffet', 'Girafarig', 'Pineco', 'Forretress',
    'Dunsparce', 'Gligar', 'Steelix', 'Snubbull', 'Granbull', 'Qwilfish',
    'Scizor', 'Shuckle', 'Heracross', 'Sneasel', 'Teddiursa', 'Ursaring',
    'Slugma', 'Magcargo', 'Swinub', 'Piloswine', 'Corsola', 'Remoraid',
    'Octillery', 'Delibird', 'Mantine', 'Skarmory', 'Houndour', 'Houndoom',
    'Kingdra', 'Phanpy', 'Donphan', 'Porygon2', 'Stantler', 'Smeargle',
    'Tyrogue', 'Hitmontop', 'Smoochum', 'Elekid', 'Magby', 'Miltank',
    'Blissey', 'Raikou', 'Entei', 'Suicune', 'Larvitar', 'Pupitar',
    'Tyranitar', 'Lugia', 'Ho-oh', 'Celebi', 'Treecko', 'Grovyle',
    'Sceptile', 'Torchic', 'Combusken', 'Blaziken', 'Mudkip', 'Marshtomp',
    'Swampert', 'Poochyena', 'Mightyena', 'Zigzagoon', 'Linoone', 'Wurmple',
    'Silcoon', 'Beautifly', 'Cascoon', 'Dustox', 'Lotad', 'Lombre',
    'Ludicolo', 'Seedot', 'Nuzleaf', 'Shiftry', 'Taillow', 'Swellow',
    'Wingull', 'Pelipper', 'Ralts', 'Kirlia', 'Gardevoir', 'Surskit',
    'Masquerain', 'Shroomish', 'Breloom', 'Slakoth', 'Vigoroth', 'Slaking',
    'Nincada', 'Ninjask', 'Shedinja', 'Whismur', 'Loudred', 'Exploud',
    'Makuhita', 'Hariyama', 'Azurill', 'Nosepass', 'Skitty', 'Delcatty',
    'Sableye', 'Mawile', 'Aron', 'Lairon', 'Aggron', 'Meditite',
    'Medicham', 'Electrike', 'Manectric', 'Plusle', 'Minun', 'Volbeat',
    'Illumise', 'Roselia', 'Gulpin', 'Swalot', 'Carvanha', 'Sharpedo',
    'Wailmer', 'Wailord', 'Numel', 'Camerupt', 'Torkoal', 'Spoink',
    'Grumpig', 'Spinda', 'Trapinch', 'Vibrava', 'Flygon', 'Cacnea',
    'Cacturne', 'Swablu', 'Altaria', 'Zangoose', 'Seviper', 'Lunatone',
    'Solrock', 'Barboach', 'Whiscash', 'Corphish', 'Crawdaunt', 'Baltoy',
    'Claydol', 'Lileep', 'Cradily', 'Anorith', 'Armaldo', 'Feebas',
    'Milotic', 'Castform', 'Kecleon', 'Shuppet', 'Banette', 'Duskull',
    'Dusclops', 'Tropius', 'Chimecho', 'Absol', 'Wynaut', 'Snorunt',
    'Glalie', 'Spheal', 'Sealeo', 'Walrein', 'Clamperl', 'Huntail',
    'Gorebyss', 'Relicanth', 'Luvdisc', 'Bagon', 'Shelgon', 'Salamence',
    'Beldum', 'Metang', 'Metagross', 'Regirock', 'Regice', 'Registeel',
    'Latias', 'Latios', 'Kyogre', 'Groudon', 'Rayquaza', 'Jirachi',
    'Deoxys', 'Pidgey', 'Rattata'
]

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def update_text_and_frame():
    
    global frame_counter
    pokemon_found = None
    
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
        if any(keyword.lower() in extracted_text.lower() for keyword in keywords):
            # Find the Pokémon name in the extracted text
            for pokemon in pokemon_names_keywords:
                if pokemon.lower() in extracted_text.lower():
                    pokemon_found = pokemon
                    break

            if pokemon_found:
                screenshot_path = 'Detect_object\Script 2\screenshot.png'
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(screenshot_path, rgb_frame)
                twt_bot = Twtbot()
                twt_bot.run(screenshot_path, pokemon_found)  # Run Twtbot when keyword is detected

        frame_counter += 1
        time.sleep(1) 

# Create the MSS object for screen capture
with mss.mss() as sct:
    frame_counter = 0
    update_text_and_frame()