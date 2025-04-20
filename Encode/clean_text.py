import os
import sys
import json
import re
import numpy as np
import pandas as pd
import scipy.special
from itertools import combinations
from collections import Counter
from tqdm import tqdm
from harvesttext import HarvestText

from Data.run_spider import read_user_posts, read_post_comments

sys.path.append('.')

ht = HarvestText()

def main():

    print('Loading raw post data ...')
    post_ids, post_info, _, _ = read_user_posts()
    post_text = []  # [post#1, post#2, ...]
    repost_text = []  # [[post#1, repost#1], [post#2, repost#2], ...]
    for post_id in tqdm(post_ids):
        if 'retweeted_status' not in post_info[post_id].keys():
            post_text.append(post_info[post_id]['content'])
        else:
            repost_text.append([f"{post_info[post_id]['retweeted_status']['content']}",f"{post_info[post_id]['content']}"])
    
    print('Loading raw comment data ...')
    comment_ids, comment_info,_ = read_post_comments()
    comment_text = []  # [[post#1, comment#1], [post#2, comment#2], ...]
    for comment_id in comment_ids:
        comment_content = comment_info[comment_id]['content']
        post_content = post_info[comment_info[comment_id]['commented_post']]['content']
        comment_text.append([f"{post_content}", f"{comment_content}"])
    
    cleaned_post_text = []
    cleaned_repost_text = []
    cleaned_comment_text = []

    print('Cleaning post text ...')
    for post in tqdm(post_text):
        cleaned_post = clean(post, max_len=100, is_control_length=True, is_format=False)
        cleaned_post_text.append(cleaned_post)
    
    print('Cleaning repost text ...')
    for repost in tqdm(repost_text):
        cleaned_post = clean(repost[0], max_len=100, is_control_length=True, is_format=False)
        cleaned_repost = clean(repost[1], max_len=100, is_control_length=True, is_format=False)
        cleaned_repost_text.append([cleaned_post, cleaned_repost])
    
    print('Cleaning comment text ...')
    for comment in tqdm(comment_text):
        cleaned_post = clean(comment[0], max_len=100, is_control_length=True, is_format=False)
        cleaned_comment = clean(comment[1], max_len=100, is_control_length=True, is_format=False)
        cleaned_comment_text.append([cleaned_post, cleaned_comment])

    post_df = pd.DataFrame({'post': cleaned_post_text})
    repost_df = pd.DataFrame({'post': [item[0] for item in cleaned_repost_text], 'repost': [item[1] for item in cleaned_repost_text]})
    comment_df = pd.DataFrame({'post': [item[0] for item in cleaned_comment_text], 'comment': [item[1] for item in cleaned_comment_text]})

    post_df.to_csv('Data/result/post.csv', index=False, sep='\t')
    repost_df.to_csv('Data/result/repost.csv', index=False, sep='\t')
    comment_df.to_csv('Data/result/comment.csv', index=False, sep='\t')

def clean(weibo_text, max_len=100, is_format=True, is_control_length=False):
    '''
    :param text: Weibo text
    :param max_len: Set the maximum length requirement for text processing, based on textrank implementation
    :param is_format: Whether to format text (clear all tabs and spaces, and add spaces between each character)
    :param is_control_length: Whether to control the length of the text, when it is False, max_len is invalid
    :return: Processed Weibo text
    '''
    text = weibo_text
    # 清除微博@符
    text = ht.clean_text(text)
    # 删除【】括号
    text = re.sub(r'[【】]', '', text)
    # 删除##括号
    text = re.sub(r'#', '', text)
    # 删除()括号中内容
    text = re.sub(r'\(.*?\)', '', text)
    # 删除（）括号中内容
    text = re.sub(r'（.*?）', '', text)
    # 删除xxx的微博视频
    text = re.sub(r'网友：', '', text)
    # 删除‘网页链接’
    text = re.sub(r'网页链接', '', text)
    # 删除xxx的微博视频
    text = re.sub(r'的微博视频', '', text)

    # Control the length of the text
    if is_control_length:
        text = control_text_length(text, max_len=max_len)  

    # Whether to format the text
    if is_format:
        # Delete space
        text = re.sub(r'\s', '', text)
        # Delete tab
        text = re.sub(r'\t', '', text)
        # Add spaces between tokens
        text = " ".join(re.findall(".{1}",text))

    # Count the length
    length = len(text)
    
    return text

def control_text_length(text, max_len=100):
    if len(text) <= max_len:
        return text

    # Start summarizing, if there is only one sentence, cut it directly
    docs = ht.cut_sentences(text)
    if len(docs) == 1:
        return text[:max_len]
    
    # len_list = [len(item) for item in docs]

    res = get_summary(docs, maxlen=max_len, sorted_by_order=True)
    return ''.join(res)

def sent_sim_cos(words1, words2):
    eps = 1e-5
    bow1 = Counter(words1)
    norm1 = sum(x ** 2 for x in bow1.values()) ** 0.5 + eps
    bow2 = Counter(words2)
    norm2 = sum(x ** 2 for x in bow2.values()) ** 0.5 + eps
    cos_sim = sum(bow1[w] * bow2[w] for w in set(bow1) & set(bow2)) / (norm1 * norm2)
    return cos_sim

def sent_sim_textrank(words1, words2):
    if len(words1) <= 1 or len(words2) <= 1:
        return 0.0
    return (len(set(words1) & set(words2))) / (np.log2(len(words1)) + np.log2(len(words2)))

def get_summary(sents, topK=5, with_importance=False, maxlen=None, avoid_repeat=False, sim_func='default', sorted_by_order=False):
    '''Use the Textrank algorithm to get the key sentences in the text

    :param sents: str sentence list
    :param topK: Select how many sentences, if maxlen is set, the length will be prioritized
    :param stopwords: Stop words used in the algorithm
    :param with_importance: Whether to include the sentence importance obtained by the algorithm
    :param standard_name: If there is an entity_mention_list, normalize the entity name in the algorithm, which generally helps to improve the algorithm effect
    :param maxlen: Set the maximum length of the summary to be obtained, if the length limit has been reached but the topK sentence has not been reached, it will stop
    :param avoid_repeat: Use the MMR principle to punish the sentence that is repeated with the already extracted summary, avoiding repetition
    :param sim_func: The similarity function used in textrank, the default is the function based on word overlap (original paper), and can also be any function that accepts two string list parameters
    :param sorted_by_order: Whether to output the extracted summary sentences in the order of the original text
    :return: Sentence list
    '''
    assert topK > 0
    import networkx as nx
    maxlen = float('inf') if maxlen is None else maxlen
    sim_func = sent_sim_textrank if sim_func == 'default' else sim_func
    # Use standard_name, the similarity can be calculated based on the result of entity linking and be more accurate
    # sent_tokens = [self.seg(sent.strip(), standard_name=standard_name, stopwords=stopwords) for sent in sents]
    sent_tokens = [sent for sent in sents if len(sent) > 0]
    G = nx.Graph()
    for u, v in combinations(range(len(sent_tokens)), 2):
        G.add_edge(u, v, weight=sim_func(sent_tokens[u], sent_tokens[v]))

    try:
        pr = nx.pagerank_scipy(G)  # sometimes fail to converge
    except:
        pr = nx.pagerank_numpy(G)
    pr_sorted = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    if not avoid_repeat:
        ret = []
        curr_len = 0
        if sorted_by_order:
            origin_sent_order = sorted(pr_sorted[:topK], key=lambda tup: tup[0])
        else:
            origin_sent_order = pr_sorted[:topK]
        
        first_index, first_imp  = origin_sent_order[0]
        curr_len += len(sents[first_index])
        if curr_len > maxlen:
            ret.append((sents[first_index][:maxlen], first_imp) if with_importance else sents[first_index][:maxlen])
            return ret

        for i, imp in origin_sent_order[1:]:
            curr_len += len(sents[i])
            if curr_len > maxlen: break
            ret.append((sents[i], imp) if with_importance else sents[i])
        
        return ret
    else:
        assert topK <= len(sent_tokens)
        ret = []
        origin_sent_order = []
        curr_len = 0
        curr_sumy_words = []
        candidate_ids = list(range(len(sent_tokens)))
        i, imp = pr_sorted[0]
        curr_len += len(sents[i])
        if curr_len > maxlen:
            ret.append((sents[i][:maxlen], imp) if with_importance else sents[i][:maxlen])
            return ret
        ret.append((sents[i], imp) if with_importance else sents[i])
        if sorted_by_order: origin_sent_order.append((sents[i], imp))
        curr_sumy_words.extend(sent_tokens[i])
        candidate_ids.remove(i)
        for iter in range(topK-1):
            importance = [pr[i] for i in candidate_ids]
            norm_importance = scipy.special.softmax(importance)
            redundancy = np.array([sent_sim_cos(curr_sumy_words, sent_tokens[i]) for i in candidate_ids])
            scores = 0.6*norm_importance - 0.4*redundancy
            id_in_cands = np.argmax(scores)
            i, imp = candidate_ids[id_in_cands], importance[id_in_cands]
            curr_len += len(sents[i])
            if curr_len > maxlen:
                if sorted_by_order:
                    res = []
                    origin_sent_order_topK = sorted(origin_sent_order[:topK], key=lambda tup: tup[0])
                    for i, imp in origin_sent_order_topK:
                        res.append((sents[i], imp) if with_importance else sents[i])
                    return res
                else:
                    return ret               
            ret.append((sents[i], imp) if with_importance else sents[i])
            if sorted_by_order: origin_sent_order.append((sents[i], imp))
            curr_sumy_words.extend(sent_tokens[i])
            del candidate_ids[id_in_cands]

        if sorted_by_order:
            res = []
            origin_sent_order_topK = sorted(origin_sent_order[:topK], key=lambda tup: tup[0])
            for i, imp in origin_sent_order_topK:
                res.append((sents[i], imp) if with_importance else sents[i])
            return res
        else:
            return ret


if __name__ == '__main__':
    main()

