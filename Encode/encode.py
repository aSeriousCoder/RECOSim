from Data.run_spider import read_user_posts, read_post_comments
from Encode.clean_text import clean
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os
import numpy as np


def main():
    print('Loading model ...')
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    batch_size = 128
    post_ids, post_info, _, _ = read_user_posts()
    comment_ids, comment_info, _ = read_post_comments()

    if os.path.exists("Data/result/cleaned_post_text.txt"):
        print('Post - Loading cleaned data ...')
        with open("Data/result/cleaned_post_text.txt", 'r') as f:
            post_text = f.readlines()
        post_text = [text.strip() for text in post_text]
    else:
        print('Post - Loading raw data ...')
        post_text = []
        for post_id in tqdm(post_ids):
            if 'retweeted_status' not in post_info[post_id].keys():
                post_text.append(clean(post_info[post_id]['content']).replace(' ', ''))
            else:
                post_text.append(clean(post_info[post_id]['content']).replace(' ', '') + "[转发]" + clean(post_info[post_id]['retweeted_status']['content']).replace(' ', ''))
        with open("Data/result/cleaned_post_text.txt", 'w') as f:
            for post in post_text:
                f.write(post + '\n')

    print('Post - Encoding ...')
    post_batch_num = len(post_text) // batch_size + 1
    post_embeddings = []
    for i in tqdm(range(post_batch_num)):
        post_embeddings.append(model.encode(post_text[i * batch_size: (i + 1) * batch_size]))

    print('Post - Saving ...')
    post_embedding_array = np.concatenate(post_embeddings, axis=0)
    np.save("Data/result/post_embeddings.npy", post_embedding_array)

    print('Post - Checking ...')
    post_embedding_array = np.load("Data/result/post_embeddings.npy")
    print('Post - Shape of post embeddings:', post_embedding_array.shape)

    if os.path.exists("Data/result/cleaned_comment_text.txt"):
        print('Comment - Loading cleaned data ...')
        with open("Data/result/cleaned_comment_text.txt", 'r') as f:
            comment_text = f.readlines()
        comment_text = [text.strip() for text in comment_text]
    else:
        print('Comment - Loading raw data ...')
        comment_text = []
        for comment_id in comment_ids:
            comment_content = comment_info[comment_id]['content']
            post_content = post_info[comment_info[comment_id]['commented_post']]['content']
            comment_text.append(clean(comment_content).replace(' ', '') + "[评论]" + clean(post_content).replace(' ', ''))
        with open("Data/result/cleaned_comment_text.txt", 'w') as f:
            for comment in comment_text:
                f.write(comment + '\n')

    print('Comment - Encoding ...')
    comment_batch_num = len(comment_text) // batch_size + 1
    comment_embeddings = []
    for i in tqdm(range(comment_batch_num)):
        comment_embeddings.append(model.encode(comment_text[i * batch_size: (i + 1) * batch_size]))

    print('Comment - Saving ...')
    comment_embedding_array = np.concatenate(comment_embeddings, axis=0)
    np.save("Data/result/comment_embeddings.npy", comment_embedding_array)

    print('Comment - Checking ...')
    comment_embedding_array = np.load("Data/result/comment_embeddings.npy")
    print('Shape of comment embeddings:', comment_embedding_array.shape)

    print('Done!')
