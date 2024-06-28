from selenium import webdriver
from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
import time

script = \
"""
let l = [[],[]];
let romajiDiv = document.querySelectorAll('div.romaji')[0];
romajiDiv.childNodes.forEach((node) => {
    if (node.nodeType === Node.TEXT_NODE && node.textContent.trim() != '') {
        l[0].push(node.textContent.trim());
        l[1].push(node.textContent.trim());
    } else if (node.nodeType === Node.ELEMENT_NODE) {
        if (node.nodeName === 'SPAN') {
            let rb = node.querySelectorAll('.rb')[0];
            let rt = node.querySelectorAll('.rt')[0];
            l[0].push(rb.textContent.trim());
            l[1].push(rt.textContent.trim());
        } else if (node.nodeName === 'BR') {
            l[0].push('/');
            l[1].push('/');
        }
    }
});
return l;
"""

def scrap(song_title, song_composer):
    
    # 웹 드라이버 설정
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(f"https://utaten.com/search?title={song_title}&artist_name={song_composer}&sort=popular_sort_asc")

    lyrics_link = driver.find_element(By.XPATH
            , '//*[@id="contents"]/main/section[3]/div/table/tbody/tr[2]/td[1]/p[1]/a')
    lyrics_link.click()
    # WebDriverWait(driver, 10).until( \
    #     EC.presence_of_element_located((By.XPATH, '//*[@id="contents"]')))
    time.sleep(2)
    
    yt_element = driver.find_element(By.XPATH 
            ,'//*[@id="contents"]/main/article/div[3]/div[1]/div/div/div[1]/div/div[4]/div/a/div/img')
    yt_link_pt = yt_element.get_attribute('src').removesuffix('/mqdefault.jpg')[-11:]
    yt_link = f'https://www.youtube.com/watch?v={yt_link_pt}'

    rom, jap = driver.execute_script(script)
    for i in range(len(rom)):
        rom[i] = rom[i].replace('sy', 'sh')

    # print(driver.current_url)
    driver.quit()
    
    return [rom, jap], yt_link