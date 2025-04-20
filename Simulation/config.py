from argparse import ArgumentParser
import os


def get_config():
    parser = ArgumentParser()
    # === SimuLine Graph Configs ===
    parser.add_argument("--raw_graph_save_dir", type=str, default="Simulation/data/dataset")
    parser.add_argument("--history_length", type=int, default=5)  # the retrieval amount
    parser.add_argument("--beginning_tag", type=int, default=1007)
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--top_k_pop", type=int, default=10)
    parser.add_argument("--padding", type=int, default=1000)
    parser.add_argument("--embedding_dim", type=int, default=512)
    parser.add_argument("--simu_batch_size", type=int, default=1024)
    parser.add_argument("--max_post_num_per_round", type=int, default=100)
    parser.add_argument("--build_dataset", type=bool, default=False)
    # === RecSys-Agent Training Configs ===
    parser.add_argument("--bole_post_data_dir", type=str, default="Simulation/tmp/SSN_P")
    parser.add_argument("--bole_friend_data_dir", type=str, default="Simulation/tmp/SSN_F")
    parser.add_argument("--feature_model_training_configs", type=str, default="Simulation/agent/recsys/bole_configs/bole_feature_train.yaml")
    parser.add_argument("--post_ranking_model_training_configs", type=str, default="Simulation/agent/recsys/bole_configs/bole_post_ranking_train.yaml")
    parser.add_argument("--friend_ranking_model_training_configs", type=str, default="Simulation/agent/recsys/bole_configs/bole_friend_ranking_train.yaml")
    parser.add_argument("--feature_model_list", type=str, default="LightGCN,BPR")
    parser.add_argument("--ranking_model", type=str, default="DeepFM")
    # === RecSys-Agent Service Configs ===
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--faiss_n_unit", type=int, default=100)
    parser.add_argument("--faiss_n_probe", type=int, default=10)
    parser.add_argument("--post_rec_list_length", type=int, default=100)
    parser.add_argument("--friend_rec_list_length", type=int, default=10)
    parser.add_argument("--ranking_batch_size", type=int, default=4096)
    parser.add_argument("--gmm_topic_num", type=int, default=257)
    # === Saving Configs ===
    parser.add_argument("--num_round", type=int, default=7)
    parser.add_argument("--show_progress", type=bool, default=True)
    parser.add_argument("--continue_simulation", type=bool, default=False)
    parser.add_argument("--continue_tag", type=int, default=1007)
    parser.add_argument("--version", type=str, default="MixRec_Revision_r1")
    args = parser.parse_args([])  # WARNING: REMOVE THE [] WHEN RUNNING IN TERMINAL
    args.simulation_result_dir = 'Simulation/result/{}'.format(args.version)
    if not os.path.exists(args.simulation_result_dir):
        os.makedirs(args.simulation_result_dir)
    args.simulation_logfile = 'Simulation/result/{}/sysout.log'.format(args.version)
    return args

