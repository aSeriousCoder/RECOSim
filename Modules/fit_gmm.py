import dgl
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
from Simulation.config import get_config
from Simulation.util.log import init_logger
from Simulation.data.graph import SimuLineGraph


def main():
    post_embedding_path = 'Data/result/post_embeddings.npy'
    comment_embedding_path = 'Data/result/comment_embeddings.npy'
    post_embedding = np.load(post_embedding_path)
    comment_embedding = np.load(comment_embedding_path)
    all_content_embeddings = np.concatenate((post_embedding, comment_embedding), axis=0)
    print(all_content_embeddings.shape)

    # === Configurations ===
    config = get_config()
    # === Build Logger ===
    logger = init_logger(config)
    logger.info('Configurations & Logger Build!')
    # === Build Graph ===
    graph = SimuLineGraph(config, logger, force_reload=False)
    logger.info('Graph Build!')

    G = graph.graph

    # Follow
    sub_g_follow = dgl.to_homogeneous(
        G.edge_type_subgraph(["follow"])
    )  # It also contains all nodes of incident type
    nx_g_follow = dgl.to_networkx(sub_g_follow)

    # Interact
    create_post_links = G.edges(etype="create_post")
    create_post_df = pd.DataFrame(
        {
            "user": create_post_links[0].numpy(),
            "post": create_post_links[1].numpy(),
        }
    )
    like_links = torch.stack(G.edges(etype="like"))
    like_links = like_links[:, G.in_degrees(like_links[1], etype='like') <= 100]
    like_df = pd.DataFrame(
        {
            "user": like_links[0].numpy(),
            "post": like_links[1].numpy(),
        }
    )
    repost_on_links = torch.stack(G.edges(etype="repost_on"))
    repost_on_links = repost_on_links[:, G.in_degrees(repost_on_links[1], etype='repost_on') <= 100]
    repost_on_df = pd.DataFrame(
        {
            "user": repost_on_links[0].numpy(),
            "post": repost_on_links[1].numpy(),
        }
    )
    comment_on_links = torch.stack(G.edges(etype="comment_on"))
    comment_on_links = comment_on_links[:, G.in_degrees(comment_on_links[1], etype='comment_on') <= 100]
    comment_on_df = pd.DataFrame(
        {
            "user": comment_on_links[0].numpy(),
            "post": comment_on_links[1].numpy(),
        }
    )
    print("Build interaction_augment_links")
    all_interaction_df = pd.concat(
        [like_df, repost_on_df, comment_on_df], ignore_index=True
        # [create_post_df, like_df, repost_on_df, comment_on_df], ignore_index=True
    )
    interaction_augment_links = all_interaction_df.join(
        all_interaction_df.set_index("post"),
        on="post",
        how="inner",
        lsuffix="_src",
        rsuffix="_dst",
    )
    print("Clean augment_links")
    augment_links = interaction_augment_links.values[:, [0,2]]
    augment_links = augment_links[augment_links[:, 0] != augment_links[:, 1]]

    follow_links = torch.stack(G.edges(etype='follow')).T
    rel_links = torch.concat([follow_links, follow_links[:, [1,0]], torch.from_numpy(augment_links)]).numpy()
    rel_links = np.unique(rel_links, axis=0)
    all_user_ids = np.unique(rel_links)

    num_post = G.num_nodes('post')
    num_repost = G.num_nodes('repost')
    num_comment = G.num_nodes('comment')
    content_rel_graph = dgl.graph([], num_nodes=num_post+num_repost+num_comment, idtype=torch.int32)
    content_rel_graph.ndata['embedding'] = torch.concat([G.nodes['post'].data['embedding'], G.nodes['repost'].data['embedding'], G.nodes['comment'].data['embedding']])

    def fetch(uid, G, content_rel_graph):
        user_related_contents = torch.concat([
            G.out_edges(uid, etype='create_post')[1],
            G.out_edges(uid, etype='comment_on')[1],
            G.out_edges(uid, etype='repost_on')[1],
            G.out_edges(uid, etype='like')[1],
            G.out_edges(uid, etype='create_repost')[1] + num_post,
            G.out_edges(uid, etype='create_comment')[1] + num_post + num_repost,
        ])
        user_related_contents_embedding = content_rel_graph.ndata['embedding'][user_related_contents]
        return user_related_contents, user_related_contents_embedding

    def pair(u1, u2, user_map):
        u1_nodes, u1_nodes_emb = user_map[u1]
        u2_nodes, u2_nodes_emb = user_map[u2]
        dot_similarity = torch.matmul(u1_nodes_emb, u2_nodes_emb.T)
        coords = torch.nonzero(dot_similarity > dot_similarity.mean() * 5)
        return torch.stack([u1_nodes[coords[:,0]], u2_nodes[coords[:,1]]]).T

    user_map = {uid: fetch(uid, G, content_rel_graph) for uid in tqdm(all_user_ids)}
    content_links = torch.concat([pair(u1, u2, user_map) for u1, u2 in tqdm(rel_links)])
    content_rel_graph.add_edges(content_links[:,0], content_links[:,1])
    content_rel_graph.add_edges(content_links[:,1], content_links[:,0])
    content_rel_subgraph_not_orph = dgl.node_subgraph(content_rel_graph, content_rel_graph.out_degrees() > 0)
    print(content_rel_subgraph_not_orph)

    content_rel_subgraph_nx = dgl.to_networkx(dgl.to_homogeneous(content_rel_subgraph_not_orph))
    communities = nx.community.louvain_communities(content_rel_subgraph_nx)
    communities = [com for com in communities if len(com) >= 100]
    print(len(communities))

    gmm_models = []
    all_embeddings_used = []

    for com in tqdm(communities):
        com_embeddings = content_rel_subgraph_not_orph.ndata['embedding'][list(com)]
        gmm = GaussianMixture(n_components=1, covariance_type='full', random_state=0)
        gmm.fit(com_embeddings)
        gmm_models.append(gmm)
        all_embeddings_used.append(com_embeddings)

    with open('Modules/ckpts/gmm_models.pkl', 'wb') as f:
        pickle.dump(gmm_models, f)

    num_community = len(gmm_models)
    gmm = GaussianMixture(n_components=num_community, covariance_type='full', random_state=0)
    gmm.fit(torch.concat(all_embeddings_used))
    gmm_labels = gmm.predict(torch.concat(all_embeddings_used))

    community_tag_ours = []
    for i, emb in enumerate(all_embeddings_used):
        community_tag_ours.extend([i] * len(emb))

    labels_pred_a = community_tag_ours
    labels_pred_b = gmm_labels

    ari = adjusted_rand_score(labels_pred_a, labels_pred_b)
    print(f"Adjusted Rand Index: {ari}")

    nmi = normalized_mutual_info_score(labels_pred_a, labels_pred_b)
    print(f"Normalized Mutual Information: {nmi}")

    homogeneity = homogeneity_score(labels_pred_a, labels_pred_b)
    completeness = completeness_score(labels_pred_a, labels_pred_b)
    v_measure = 2 * (homogeneity * completeness) / (homogeneity + completeness)

    print(f"Homogeneity: {homogeneity}")
    print(f"Completeness: {completeness}")
    print(f"V-Measure: {v_measure}")
