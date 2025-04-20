# RECOSIM: A Universal, Accurate, and Scalable Simulation Framework for Online Community Recommendations

As recommender systems become increasingly important components in online communities, studying their impact on these communities becomes ever more crucial. Facing the high costs and ethical risks of real-world social experiments, researchers construct recommendation simulators to study the interactions between recommender systems and users. However, existing simulators face challenges in providing universal, accurate, and scalable interaction modeling for various types of online communities involving millions of contents and users with diverse action types. To address these challenges, we propose RECOSIM, a simulation framework capable of offering efficient recommendation interaction simulations across a wide range of scenarios. RECOSIM decomposes the user agent into five fundamental modules: Encode Model, Decode Model, Activity Model, Scoring Model, and Generation Model, allowing for accurate and extensible modeling of user behavior and interaction dynamics. The recommender system agent adheres to established industry architectures, implementing three stages and four fundamental strategies, thereby improving generalizability across various platforms and the computational efficiency of simulation. Utilizing two real-world datasets (Weibo and Zhihu), we validate the accuracy and stability of each component and the overall framework of RECOSIM, demonstrating the reliability of RECOSIM as a simulation environment. Subsequently, we delve into analyzing the impact of the four fundamental recommendation strategies on online communities, providing design inspirations for enhancing user engagement and community growth. 

## Features

- Data Collection: Automated data crawling from Sina Weibo
- Data Encoding: Transforming raw data into standardized vector representations
- Modular Design: Includes Activity model, Scoring model, and Generation Model
- Simulation System: Simulates user-content interaction processes
- Decoding Analysis: Decode vector representations to text for post-simulation analysis

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd RECOSim-Reorganize
```

2. Create and activate a virtual environment (recommended):
```bash
conda create -n recosim python==3.9
conda activate recosim
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main program:
```bash
python launch.py
```

## Project Structure

```
.
├── Data/             # Data collection and storage
├── Encode/           # Data encoding module
├── Modules/          # Core functional modules
├── Simulation/       # Simulation system
├── Decode/           # Data decoding module
├── requirements.txt  # Project dependencies
└── launch.py         # Main program entry
```

## Tech Stack

- Python 3.x
- PyTorch
- DGL (Deep Graph Library)
- FAISS
- RecBole
- Sentence Transformers
- Other machine learning and data processing libraries

## Notes

- Ensure PyTorch with CUDA support is installed
- Data collection requires appropriate network environment
- GPU is recommended for model training and simulation
