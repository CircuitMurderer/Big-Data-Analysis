import os
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains


class CrawlConfig:
    def __init__(self):
        self.search = '手机'
        self.price = (0, 2000)
        self.pri_range = [(0, 2000), (2000, 5000), (5000, 8000), (8000, 15000)]
        self.p_sort = '&psort=3'  # 按销量排序

        self.url0 = 'https://search.jd.com/search?keyword={}&wq={}{}&ev=exprice_{}-{}%5E'
        self.url = self.url0.format(self.search, self.search, self.p_sort, self.price[0], self.price[1])

        self.drv_path = 'msedgedriver.exe'
        self.comm_root = './comments/'

        self.xpath = {
            'goods_list': '//div[@id="J_goodsList"]//div[@class="p-name p-name-type-2"]/a[@target="_blank"]',
            'goods_comm': '//li[@clstag="shangpin|keycount|product|shangpinpingjia_1"]',
            'next_page': '//a[@clstag="shangpin|keycount|product|pinglunfanye-nextpage"]',
            'time_order': '//li[@clstag="shangpin|keycount|product|shijianpaixu"]',
            'order_float': '//div[@class="current"]/span[@class="J-current-sortType"]'
        }

        self.max_prod = 10
        self.comm_num = 500

    def update_url(self):
        self.url = self.url0.format(self.search, self.search, self.p_sort, self.price[0], self.price[1])


def crawl_1_goods(url, cnf: CrawlConfig):
    drv = webdriver.Edge(cnf.drv_path)
    drv.get(url)
    drv.maximize_window()
    drv.implicitly_wait(5)

    while not drv.find_elements_by_xpath(cnf.xpath['next_page']):
        drv.refresh()
        drv.find_element_by_xpath(cnf.xpath['goods_comm']).click()
        drv.execute_script('window.scrollBy(0, 500);')
        # time.sleep(2)

    flt = drv.find_element_by_xpath(cnf.xpath['order_float'])
    drv.execute_script('window.scrollBy(0, 500);')
    ActionChains(drv).move_to_element(flt).perform()
    # driver.execute_script("arguments[0].mouseover();", flt)
    drv.find_element_by_xpath(cnf.xpath['time_order']).click()  # 按时间排序
    # driver.execute_script('arguments[0].click();', tm)

    soup = BeautifulSoup(drv.page_source, 'html.parser')
    all_comm_rt = soup.find('div', attrs={'class': 'percent-con'}).text
    goods_name = soup.find('div', attrs={'class': 'sku-name'}).text

    count = 0
    comment = []
    while True:
        soup = BeautifulSoup(drv.page_source, 'html.parser')
        for it in soup.find_all('p', attrs={'class': 'comment-con'}):
            txt = it.text.replace('\n', ' ')
            comment.append(txt)
            count += 1
            if count == cnf.comm_num:
                break
        if count == cnf.comm_num:
            break
        btn = drv.find_elements_by_xpath(cnf.xpath['next_page'])
        if btn:
            drv.execute_script('arguments[0].click();', btn[0])
        else:
            break

    drv.close()
    df = pd.DataFrame(comment, columns=['text_a'])

    save_dir = cnf.comm_root + str(cnf.price[0]) + '~' + str(cnf.price[1])
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    save_path = '/'.join([save_dir, goods_name.strip(' \n').replace(' ', '_').replace(':', '_')])
    save_path = save_path + ';' + all_comm_rt.strip(' "')
    df.to_csv(save_path + '.csv', index=False)


if __name__ == '__main__':
    conf = CrawlConfig()

    for pr_range in conf.pri_range[:]:
        conf.price = pr_range
        conf.update_url()
        print(conf.price)

        driver = webdriver.Edge(conf.drv_path)
        driver.get(conf.url)
        driver.implicitly_wait(5)

        products = driver.find_elements_by_xpath(conf.xpath['goods_list'])
        hrefs = [i.get_attribute('href') for i in products]
        driver.close()

        for href in hrefs[:conf.max_prod]:
            crawl_1_goods(href, conf)
