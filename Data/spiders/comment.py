#!/usr/bin/env python
# encoding: utf-8
"""
Author: nghuyong
Mail: nghuyong@163.com
Created Time: 2020/4/14
"""
import json
from scrapy import Spider
from scrapy.http import Request
from .common import parse_user_info, parse_time, url_to_mid


class CommentSpider(Spider):
    """
    微博评论数据采集
    """
    name = "comment_spider"
    post_ids = None

    def start_requests(self):
        """
        爬虫入口
        """
        # 这里tweet_ids可替换成实际待采集的数据
        # 一次20条
        tweet_ids = CommentSpider.post_ids
        for tweet_id in tweet_ids:
            url = f"https://weibo.com/ajax/statuses/buildComments?" \
                  f"is_reload=1&id={tweet_id}&is_show_bulletin=2&is_mix=0&count=20"
            yield Request(url, callback=self.parse, meta={'source_url': url, 'mid': tweet_id})

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        data = json.loads(response.text)
        if response.status != 200 or data['ok'] != 1:
            self.logger.error('NETWORK FAILURE - ', response.status)
        for comment_info in data['data']:
            item = self.parse_comment(comment_info, response.meta['mid'])
            yield item
        if data.get('max_id', 0) != 0 and data.get('max_id', 0) < 1000:  # 最多爬取1000条评论
            url = response.meta['source_url'] + '&max_id=' + str(data['max_id'])
            yield Request(url, callback=self.parse, meta=response.meta)

    @staticmethod
    def parse_comment(data, mid):
        """
        解析comment
        """
        item = dict()
        item['created_at'] = parse_time(data['created_at'])
        item['_id'] = str(data['id'])
        item['like_counts'] = data['like_counts']
        if 'source' in data:
            item['ip_location'] = data['source']
        else:
            item['ip_location'] = ''
        item['content'] = data['text_raw']
        item['comment_user'] = parse_user_info(data['user'])
        item['commented_post'] = mid
        return item
