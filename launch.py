from Data.run_spider import main as crawling_data
from Encode.encode import main as encode_data
from Modules.build_dataset import main as build_module_datasets
from Modules.fit_decoder import main as fit_decoder
from Modules.fit_gmm import main as fit_distribution
from Modules.fit_scoring_model import main as fit_scoring_model
from Modules.fit_activity_model import main as fit_activity_model
from Simulation.run_simulation import main as run_simulation
from Decode.train_decode_model import main as decode_data
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # === Data Collection and Encoding ===
    crawling_data()  # collecting data from Sina-Weibo
    encode_data()
    # === Training Modules ===
    fit_distribution()  # >>> gmm_models
    build_module_datasets()  # >>> Modules/data/*.pt
    fit_scoring_model()  # >>> scoring_model & quantile_values
    fit_activity_model()  # >>> activity_model & activity_mi
    # === Simulation ===
    run_simulation()  # >>> simulation results
    # === Decoding ===
    decode_data()  # >>> projector & decoder
