# -*- coding: utf-8 -*-

BOT_NAME = 'spider'

SPIDER_MODULES = ['spiders']
NEWSPIDER_MODULE = 'spiders'

REDIRECT_ENABLED = True

ROBOTSTXT_OBEY = False

with open('cookie.txt', 'rt', encoding='utf-8') as f:
    cookie = f.read().strip()

DEFAULT_REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/113.0.0.0 Safari/537.36',
    'Cookie': cookie
}

CONCURRENT_REQUESTS = 16

DOWNLOAD_DELAY = 1

DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': None,
    'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': None,
    'middlewares.IPProxyMiddleware': 100,
    'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 101,
}

ITEM_PIPELINES = {
    'pipelines.JsonWriterPipeline': 300,
}

# Whether to enable logging
LOG_ENABLED = True

# Log level CRITICAL, ERROR, WARNING, INFO, DEBUG
LOG_LEVEL = 'ERROR'
