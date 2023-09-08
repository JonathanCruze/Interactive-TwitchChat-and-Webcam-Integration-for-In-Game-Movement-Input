import threading
from Tch import Tchbot # from the Twitch file import the class Twitchbot
import Object_det # Import the Object_detection file

def main():
    
    tch_token = 'twitch_token'
    channel = 'twitch_channel'
    
    tch_bot = Tchbot(tch_token, channel)
    thread1 = threading.Thread(target=tch_bot.run)
    thread2 = threading.Thread(target=Object_det.main_obj)
    
    # Iniciar los hilosrr
    thread1.start()
    thread2.start()
    
    # Esperar a que ambos hilos terminen
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()