from selenium import webdriver
import os


browser = webdriver.Chrome('./chromedriver.exe') 

browser.get("https://s3-us-west-1.amazonaws.com/ra-training/test1.html")
html = browser.page_source
browser.quit()

with open ('html.html', 'w+') as f:
    f.write(str(html))
