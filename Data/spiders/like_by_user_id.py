
import json
from scrapy import Spider
from scrapy.http import Request
from .common import parse_tweet_info, parse_long_tweet


class LikeSpiderByUserID(Spider):
    """
    用户推文数据采集
    """
    name = "like_spider_by_user_id"
    base_url = "https://weibo.cn"
    user_ids = None

    def start_requests(self):
        """
        爬虫入口
        """
        # 话题时间：2023/05/23 12:00:00
        # 前后浮动三天（合计一周）
        # _start_time = datetime.datetime(year=2023, month=5, day=20, hour=0).strftime("%Y-%m-%d-%H")
        # _end_time = datetime.datetime(year=2023, month=5, day=27, hour=0).strftime("%Y-%m-%d-%H")
        # URL：https://weibo.com/ajax/statuses/likelist?uid=5625557517&page=1
        for user_id in LikeSpiderByUserID.user_ids:
            url = f"https://weibo.com/ajax/statuses/likelist?uid={user_id}&page=1"
            yield Request(url, callback=self.parse, meta={'user_id': user_id, 'page_num': 1})

    def parse(self, response, **kwargs):
        """
        网页解析
        """
        data = json.loads(response.text)
        tweets = data['data']['list']
        continue_flag = True  # 爬取的微博是按照时间顺序排列的，如果遇到在start_time之前的微博，则停止爬取
        for tweet in tweets:
            item = parse_tweet_info(tweet)
            item['source_user_id'] = response.meta['user_id']
            if item['created_at'] < '2023-05-20 00:00:00':
                continue_flag = False
                break
            elif item['isLongText']:
                url = "https://weibo.com/ajax/statuses/longtext?id=" + item['mblogid']
                yield Request(url, callback=parse_long_tweet, meta={'item': item})
            else:
                yield item
        if tweets and continue_flag:
            user_id, page_num = response.meta['user_id'], response.meta['page_num']
            page_num += 1
            url = f"https://weibo.com/ajax/statuses/likelist?uid={user_id}&page={page_num}"
            yield Request(url, callback=self.parse, meta={'user_id': user_id, 'page_num': page_num})
