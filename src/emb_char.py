import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

class EmbeddingCharacter:
    def __init__(self):