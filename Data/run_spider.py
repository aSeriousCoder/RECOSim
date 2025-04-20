import os
import sys
import json
from tqdm import tqdm
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from spiders.tweet_by_user_id import TweetSpiderByUserID
from spiders.tweet_by_keyword import TweetSpiderByKeyword
from spiders.comment import CommentSpider
from spiders.follower import FollowerSpider
from spiders.user import UserSpider
from spiders.like_by_user_id import LikeSpiderByUserID


# 1. Config the topics to crawl
TOPICS = [
    "Topic 1",
    "Topic 2",
    "Topic 3",
]
OUTPUT_DIR = "Data/output"  # Output directory, which saves the raw crawled data
RESULT_DIR = "Data/result"  # Result directory, which saves the processed data
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def get_user_id_list_from_topic_participate(crawling_topic):
    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    TweetSpiderByKeyword.keyword = crawling_topic
    TweetSpiderByKeyword.name = "tweet_spider_by_keyword_{}".format(crawling_topic)
    process.crawl(TweetSpiderByKeyword)
    process.start()  # the script will block here until the crawling is finished


def merge_user_id_list_from_topic_participate():  # Seed Users from Topic
    all_user_id = set()
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("tweet_spider_by_keyword_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in fr:
                    record = json.loads(line)
                    user_id = record["user"]["_id"]
                    all_user_id.add(user_id)
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_id_list_from_topic_participate.txt"), "w", encoding="utf-8") as fw:
        for user_id in all_user_id:
            fw.write("{}\n".format(user_id))


def read_user_id_list_from_topic_participate():
    user_id_list = []
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_id_list_from_topic_participate.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            user_id_list.append(line.strip())
    return user_id_list


def get_user_following_relations(user_ids):
    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    FollowerSpider.user_ids = user_ids
    process.crawl(FollowerSpider)
    process.start()  # the script will block here until the crawling is finished


def merge_user_following_relations(user_ids):
    # The number is too large, let's just use seed users as the user list
    user_following_relations: list[list[str]] = []
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("follower_spider_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    if record["follower_info"]["_id"] in user_ids:
                        user_id = record["fan_id"]
                        follower_id = record["follower_info"]["_id"]
                        user_following_relations.append([user_id, follower_id])  # user_id follows follower_id
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_following_relations.txt"), "w", encoding="utf-8") as fw:
        for (user_id, follower_id) in user_following_relations:
            fw.write("{} {}\n".format(user_id, follower_id))


def read_user_following_relations():
    user_following_relations: list[list[str]] = []
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_following_relations.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            user_following_relations.append(line.split())
    return user_following_relations


def get_user_info(user_ids):
    crawled_user_list = get_info_crawled_user_list()
    user_ids = [user_id for user_id in user_ids if user_id not in crawled_user_list]
    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    UserSpider.user_ids = user_ids
    process.crawl(UserSpider)
    process.start()  # the script will block here until the crawling is finished


def get_info_crawled_user_list():
    crawled_user_list = []
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("user_spider_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    crawled_user_list.append(record["_id"])
    return crawled_user_list


def merge_user_info():
    user_info = {}
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("user_spider_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    if record["_id"] not in user_info:
                        user_info[record["_id"]] = record
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_info.txt"), "w", encoding="utf-8") as fw:
        fw.write(json.dumps(user_info))


def read_user_info():
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_info.txt"), "r", encoding="utf-8") as fr:
        return json.loads(fr.read())


def get_user_posts(user_ids):
    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    TweetSpiderByUserID.user_ids = user_ids
    process.crawl(TweetSpiderByUserID)
    process.start()  # the script will block here until the crawling is finished


def get_post_crawled_user_list():
    crawled_user_list = set()
    num_like = []
    num_comment = []
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("tweet_spider_by_user_id_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    crawled_user_list.add(record['user']["_id"])
                    num_like.append(record["attitudes_count"])
                    num_comment.append(record["comments_count"])
    return crawled_user_list, num_like, num_comment


def merge_user_posts(participating_user_id_list):
    post_id_list, post_info, user_posting_relations, user_reposting_relations = set(), {}, set(), set()
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("tweet_spider_by_user_id_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    if 'retweeted_status' in record:  # Repost
                        # Check - 1. User in list  2. Time  3. Not processed
                        if record['retweeted_status']['user']["_id"] not in participating_user_id_list:  # Repost not from users in the list, can be ignored
                            continue
                        if record['retweeted_status']['created_at'] < '2023-05-20 00:00:00': 
                            continue
                        if record["_id"] not in post_id_list:  # Not processed - Process in advance
                            post_id_list.add(record['retweeted_status']["_id"])
                            post_info[record['retweeted_status']["_id"]] = record['retweeted_status']
                            user_posting_relations.add((record['retweeted_status']['user']["_id"], record['retweeted_status']["_id"]))
                        # Record repost relationship
                        post_id_list.add(record["_id"])
                        post_info[record["_id"]] = record
                        user_reposting_relations.add((record['user']["_id"], record["_id"], record['retweeted_status']["_id"]))  # user_id, reposted_tweet_id, original_tweet_id
                    else:  # Original / Liked tweet information
                        # Check - 1. User in list  2. Time  3. Not processed
                        if record['user']["_id"] not in participating_user_id_list:  # Liked tweet not from users in the list, can be ignored
                            continue
                        if record['created_at'] < '2023-05-20 00:00:00': 
                            continue
                        if record["_id"] in post_id_list:  # Duplicate tweet, can be ignored; Possible because the user liked a tweet within the range
                            continue
                        post_id_list.add(record["_id"])
                        post_info[record["_id"]] = record
                        user_posting_relations.add((record['user']["_id"], record["_id"]))
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "post_id_list.txt"), "w", encoding="utf-8") as fw:
        for post_id in post_id_list:
            fw.write("{}\n".format(post_id))
    with open(os.path.join(result_data_dir, "post_info.txt"), "w", encoding="utf-8") as fw:
        fw.write(json.dumps(post_info))
    with open(os.path.join(result_data_dir, "user_posting_relations.txt"), "w", encoding="utf-8") as fw:
        for (user_id, post_id) in user_posting_relations:
            fw.write("{} {}\n".format(user_id, post_id))
    with open(os.path.join(result_data_dir, "user_reposting_relations.txt"), "w", encoding="utf-8") as fw:
        for (user_id, post_id, reposted_id) in user_reposting_relations:
            fw.write("{} {} {}\n".format(user_id, post_id, reposted_id))


def read_user_posts():
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    post_id_list, user_posting_relations, user_reposting_relations = [], [], []
    with open(os.path.join(result_data_dir, "post_id_list.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            post_id_list.append(line.strip())
    with open(os.path.join(result_data_dir, "post_info.txt"), "r", encoding="utf-8") as fr:
        post_info = json.loads(fr.read())
    with open(os.path.join(result_data_dir, "user_posting_relations.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            user_posting_relations.append(line.split())
    with open(os.path.join(result_data_dir, "user_reposting_relations.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            user_reposting_relations.append(line.split())
    return post_id_list, post_info, user_posting_relations, user_reposting_relations


def get_user_likes(user_ids):
    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    LikeSpiderByUserID.user_ids = user_ids
    process.crawl(LikeSpiderByUserID)
    process.start()  # the script will block here until the crawling is finished


def merge_user_likes(participating_user_id_list, post_mid_list):
    user_list = set(participating_user_id_list)
    item_list = set(post_mid_list.keys())
    user_liking_relations = []
    like_info = {}
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("like_spider_by_user_id_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    if record['source_user_id'] in user_list and record['mblogid'] in item_list:
                        user_liking_relations.append([record['source_user_id'], post_mid_list[record['mblogid']]])
                        like_info['{}-{}'.format(record['source_user_id'], record['mblogid'])] = record
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "user_liking_relations.txt"), "w", encoding="utf-8") as fw:
        for (user_id, post_id) in user_liking_relations:
            fw.write("{} {}\n".format(user_id, post_id))
    with open(os.path.join(result_data_dir, "like_info.txt"), "w", encoding="utf-8") as fw:
        fw.write(json.dumps(like_info))


def read_user_likes():
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    user_liking_relations = []
    with open(os.path.join(result_data_dir, "user_liking_relations.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            user_liking_relations.append(line.split())
    with open(os.path.join(result_data_dir, "like_info.txt"), "r", encoding="utf-8") as fr:
        like_info = json.loads(fr.read())
    return like_info, user_liking_relations


def get_post_comments(post_ids):
    os.environ["SCRAPY_SETTINGS_MODULE"] = "settings"
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    CommentSpider.post_ids = post_ids
    process.crawl(CommentSpider)
    process.start()  # the script will block here until the crawling is finished


def merge_post_comments(participating_user_id_list):
    user_list = set(participating_user_id_list)
    comment_id_list, comment_info, user_comment_relations = [], {}, []
    raw_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), OUTPUT_DIR)
    for filename in os.listdir(raw_data_dir):
        f = os.path.join(raw_data_dir, filename)
        if os.path.isfile(f) and filename.startswith("comment_spider_"):
            print('Processing file: {}'.format(filename))
            with open(f, "r", encoding="utf-8") as fr:
                for line in tqdm(fr):
                    record = json.loads(line)
                    if record['comment_user']['_id'] in user_list:  # post一定在list中，只需要检查user是否在list中
                        comment_id_list.append(record['_id'])
                        comment_info[record['_id']] = record
                        user_comment_relations.append([record['comment_user']['_id'], record['_id'], record['commented_post']])  # user_id, comment_id, commented_post_id
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    with open(os.path.join(result_data_dir, "comment_id_list.txt"), "w", encoding="utf-8") as fw:
        for comment_id in comment_id_list:
            fw.write("{}\n".format(comment_id))
    with open(os.path.join(result_data_dir, "comment_info.txt"), "w", encoding="utf-8") as fw:
        fw.write(json.dumps(comment_info))
    with open(os.path.join(result_data_dir, "user_comment_relations.txt"), "w", encoding="utf-8") as fw:
        for (user_id, comment_id, commented_post_id) in user_comment_relations:
            fw.write("{} {} {}\n".format(user_id, comment_id, commented_post_id))


def read_post_comments():
    result_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULT_DIR)
    comment_id_list, user_comment_relations = [], []
    with open(os.path.join(result_data_dir, "comment_id_list.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            comment_id_list.append(line.strip())
    with open(os.path.join(result_data_dir, "comment_info.txt"), "r", encoding="utf-8") as fr:
        comment_info = json.loads(fr.read())
    with open(os.path.join(result_data_dir, "user_comment_relations.txt"), "r", encoding="utf-8") as fr:
        for line in fr:
            user_comment_relations.append(line.split())
    return comment_id_list, comment_info, user_comment_relations


def main():
    # 2. For each topic to be crawled, crawl all the tweets under the topic and store the users who posted the tweets ➡️ User list
    for topic in TOPICS:
        get_user_id_list_from_topic_participate(topic)
    merge_user_id_list_from_topic_participate()
    participating_user_id_list = read_user_id_list_from_topic_participate()

    # 3. For each user in the user list, crawl their personal information and the users they follow ➡️ User social network (follow)
    # Users they follow
    get_user_following_relations(participating_user_id_list[:])
    merge_user_following_relations(participating_user_id_list)
    user_following_relations = read_user_following_relations()
    # Personal information
    get_user_info(participating_user_id_list[:])
    merge_user_info()
    user_info = read_user_info()

    # 4. For each user in the user list, crawl their tweets 
    get_user_posts(participating_user_id_list[:])
    merge_user_posts(participating_user_id_list)
    post_id_list, post_info, user_posting_relations, user_reposting_relations = read_user_posts()

    # 5. For each tweet in the tweet list, crawl their comments & retweets
    post_with_comment = {p: post_info[p]['comments_count'] for p in post_info if post_info[p]['comments_count'] > 0}
    get_post_comments(list(post_with_comment.keys())[:])
    merge_post_comments(participating_user_id_list)
    comment_id_list, comment_info, user_comment_relations = read_post_comments()

    # 6. For each tweet in the tweet list, crawl their likes
    get_user_likes(participating_user_id_list[:])
    merge_user_likes(participating_user_id_list, {post_info[p]['mblogid']: p for p in post_info})
    like_info, user_liking_relations = read_user_likes()

    print('Done!')
