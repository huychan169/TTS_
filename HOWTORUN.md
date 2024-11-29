# Structure

tts_project/
│
├── venv/
├── data/
│ ├── **init**.py
│ ├── preprocessing.py
│ └── dataset.py
├── models/
│ ├── **init**.py
│ ├── encoder.py
│ ├── decoder.py
│ └── vocoder.py
├── train.py
├── inference.py
└── config.yaml

# Create project directory

mkdir tts_project
cd tts_project

# Create virtual environment

python3 -m venv venv
source venv/bin/activate

# Install dependencies

pip install -r requirements.txt

# Training

python train.py

# Inference

python inference.py
