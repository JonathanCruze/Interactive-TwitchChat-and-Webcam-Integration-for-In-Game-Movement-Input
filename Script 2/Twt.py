import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class Twtbot:
    def __init__(self):
        self.driver = webdriver.Chrome()

    def login(self, username, password):
        # Open Twitter
        self.driver.get("https://twitter.com/search")
        time.sleep(10)

        username_input = self.driver.find_element(By.XPATH, '//input[@name="text"]')  # Updated XPath for the username input field
        username_input.send_keys(username)
        username_input.send_keys(Keys.RETURN)

        time.sleep(2)

        password_input = self.driver.find_element(By.XPATH, '//input[@name="password"]')  # Updated XPath for the password input field
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)

        time.sleep(5)

    def post(self, tweet_content, media_path):

        post_link = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="SideNav_NewTweet_Button"]')
        post_link.click()

        tweet_box = WebDriverWait(self.driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, '[data-testid="tweetTextarea_0"]'))
        )

        time.sleep(5)

        tweet_box.send_keys(tweet_content)

        time.sleep(2)

        ############# UPLOAD IMAGE ########

        media_button = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="fileInput"]')
        media_button.send_keys(os.path.abspath(media_path))

        time.sleep(5)   

        tweet_button = self.driver.find_element(By.CSS_SELECTOR, '[data-testid="tweetButton"]')
        tweet_button.click()

        time.sleep(5)

    def run(self, media_path):

        username = 'twitter_username'  
        password = 'twitter_password'
        pokemon = 'Chorizard'
        media_path = "Detect_object\Script 2\screenshot.png"
        tweet_content = f"Hemos encontrado a: {pokemon}" 

        self.login(username, password) 
        self.post(tweet_content, media_path)
        self.driver.quit()