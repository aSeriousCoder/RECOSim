#!/usr/bin/env python
# encoding: utf-8
"""
Author: rightyonghu
Created Time: 2022/10/22
"""
import datetime
import json
import re
from scrapy import Spider, Request
from .common import parse_tweet_info, parse_long_tweet


class TweetSpiderByKeyword(Spider):
    """
    关键词搜索采集
    """
    name = "tweet_spider_by_keyword_None"
    base_url = "https://s.weibo.com/"
    keyword = None

    def start_requests(self):
        """
        爬虫入口
        """
        keyword = TweetSpiderByKeyword.keyword
        # 话题时间：2023/05/23 12:00:00
        # 前后浮动三天（合计一周）
        start_time = datetime.datetime(year=2023, month=5, day=20, hour=0)
        end_time = datetime.datetime(year=2023, month=5, day=27, hour=0)
        # 是否按照小时进行切分，数据量更大; 对于非热门关键词**不需要**按照小时切分
        is_split_by_hour = True
        if not is_split_by_hour:
            _start_time = start_time.strftime("%Y-%m-%d-%H")
            _end_time = end_time.strftime("%Y-%m-%d-%H")
            url = f"https://s.weibo.com/weibo?q={keyword}&timescope=custom%3A{_start_time}%3A{_end_time}&page=1"
            yield Request(url, callback=self.parse, meta={'keyword': keyword})
        else:
            time_cur = start_time
            while time_cur < end_time:
                _start_time = time_cur.strftime("%Y-%m-%d-%H")
                _end_time = (time_cur + datetime.timedelta(hours=1)).strftime("%Y-%m-%d-%H")
                url = f"https://s.weibo.com/weibo?q={keyword}&timescope=custom%3A{_start_time}%3A{_end_time}&page=1"
                yield Request(url, callback=self.parse, meta={'keyword': keyword})
                time_cur = time_cur + datetime.timedelta(hours=1)

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        html = response.text
        if '<p>抱歉，未找到相关结果。</p>' in html:
            self.logger.info(f'no search result. url: {response.url}')
            return
        tweet_ids = re.findall(r'weibo\.com/\d+/(.+?)\?refer_flag=1001030103_" ', html)
        for tweet_id in tweet_ids:
            url = f"https://weibo.com/ajax/statuses/show?id={tweet_id}"
            yield Request(url, callback=self.parse_tweet, meta=response.meta, priority=10)
        next_page = re.search('<a href="(.*?)" class="next">下一页</a>', html)
        if next_page:
            url = "https://s.weibo.com" + next_page.group(1)
            yield Request(url, callback=self.parse, meta=response.meta)

    @staticmethod
    def parse_tweet(response):
        """
        解析推文
        """
        data = json.loads(response.text)
        item = parse_tweet_info(data)
        item['keyword'] = response.meta['keyword']
        if item['isLongText']:
            url = "https://weibo.com/ajax/statuses/longtext?id=" + item['mblogid']
            yield Request(url, callback=parse_long_tweet, meta={'item': item}, priority=20)
        else:
            yield item
