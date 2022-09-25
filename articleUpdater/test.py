from django.core.files import File
from django.core.files.temp import NamedTemporaryFile
import urllib
from urllib.parse import urlparse
import urllib.request
from whatstrending.models import Article
from bs4 import BeautifulSoup as soup
import ssl
import requests

def update_articles(title, descr, image):
    article = Article()
    article.title = title
    article.descr = descr[:30] + "..."
    #image
    name = urlparse(image).path.split('/')[-1]
    ssl._create_default_https_context = ssl._create_unverified_context
    img_temp = NamedTemporaryFile(delete=True)
    img_temp.write(urllib.request.urlopen(image).read())
    img_temp.flush()

    article.image.save(name, File(img_temp), save=True)
    article.save()

def run():
    nbc_url="https://www.nbcnews.com/"
    r = requests.get(nbc_url)
    b = soup(r.content, features="html.parser")


    links = []
    for news in b.findAll('h2', {'class': 'tease-card__headline'}):
        links.append(news.a['href'])

    for news in b.findAll('h2', {'class': 'styles_headline__ice3t'}):
        links.append(news.a['href'])

    descr = []
    title = []
    news_articles = {}
    for link in links[:1]:
        try:
            page = requests.get(link)
            bsobj = soup(page.content, features="html.parser")
            text = ""

            for news in bsobj.findAll('div', {'class': 'article-hero-headline'}):
                text += news.text.strip()
            # print(text)
            title.append(text)

            article = ""
            for news in bsobj.findAll('div', {'class': 'article-body__content'}):
                article += news.text.strip()
            descr.append(article)

            image = bsobj.find('picture', {'class': 'article-hero__main-image'}).find('img')['src']
            news_articles[text] = {"description": article, "image": image}
        except:
            continue

    for i in news_articles:
        update_articles(i, news_articles[i]["description"], news_articles[i]["image"])
