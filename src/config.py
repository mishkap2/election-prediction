import os
import torch

NEWS_API_KEY: str | None = os.getenv('NEWS_API_KEY', None)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
MAX_ARTICLES = 990
