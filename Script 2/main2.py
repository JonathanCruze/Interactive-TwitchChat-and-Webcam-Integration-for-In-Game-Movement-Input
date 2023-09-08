from Event_recog import TextDetection as txt_det

def main_2():
    text_detection = txt_det(txt_det.game_window_name, txt_det.csv_filename, txt_det.keywords)
    text_detection.ufpdate_text_and_frame()

if __name__ == "__main__":
    main_2()