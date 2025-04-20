import torch
from Simulation.config import get_config
from Simulation.util.log import init_logger
from Simulation.service.user_action_service import UserActionService
from Simulation.service.metrics_service import MetricsService
from Simulation.data.graph import SimuLineGraph
from Simulation.agent.recsys.deep_recsys import DeepRecSys
from Simulation.agent.recsys.random_recsys import RandomRecSys
from Simulation.agent.recsys.pop_recsys import PopRecSys
from Simulation.agent.recsys.social_recsys import SocialRecSys
from Simulation.agent.recsys.mix_recsys import MixRecSys
from Simulation.agent.user.awesome_user import AwesomeUser as User


def main():    
    # === Configurations ===
    config = get_config()

    if 'Random' in config.version:
        recsys_class = RandomRecSys
    elif 'Pop' in config.version:
        recsys_class = PopRecSys
    elif 'Social' in config.version:
        recsys_class = SocialRecSys
    elif 'Deep' in config.version:
        recsys_class = DeepRecSys
    elif 'Mix' in config.version:
        recsys_class = MixRecSys
    else:
        raise Exception('RecSys Type Not Implement !')

    # === Build Logger ===
    logger = init_logger(config)
    logger.info('Configurations & Logger Build!')
    logger.info('Start Simulation for {}'.format(config.version))

    # === Build Services ===
    user_action_service = UserActionService(config, logger)  # generate and score
    metrics_service = MetricsService(config, logger)  # save UGT state, action density, community belonging
    if config.continue_simulation:
        metrics_service.load()
    logger.info('Services Build!')

    # === Build Graph ===
    graph = SimuLineGraph(config, logger, force_reload=False)
    logger.info('Graph Build!')

    # === Build RecSys ===
    recsys = recsys_class(config, logger, graph)
    logger.info('RecSys Build!')

    # === Build User ===
    user = User(config, logger, graph, user_action_service, metrics_service)
    logger.info('User Build!')
    
    graph.save_simulation(tag=graph._cur_tag-1)

    # === Run Pipeline ===
    for i in range(config.num_round):
        logger.info(f'SimuLating Round {i+1} (Tag {graph._cur_tag}) Start!')
        # === UGT Update ===
        user.update_state()
        # === Recommend ===
        recsys.prepare()
        post_recommendation = recsys.recommend_post().long()
        friends_recommendation = recsys.recommend_friends().long()
        # === User Action ===
        user_browse_actions = user.browse(post_recommendation)
        user_follow_actions = user.follow(friends_recommendation, user_browse_actions)
        # === Write Back to Graph ===
        # graph.load_simulation(tag=graph._cur_tag-1)
        graph.update_graph(post_recommendation, friends_recommendation, user_browse_actions, user_follow_actions)
        # === Save Results ===
        metrics_service.save()
        graph.save_simulation(tag=graph._cur_tag-1)
        logger.info(f'Round {i+1} End!')
        logger.info('=============================')

    # === Save Results ===
    metrics_service.save()
    graph.save_simulation()

