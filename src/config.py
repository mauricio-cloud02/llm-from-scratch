# ---- Data settings ----
CONTEXT_LENGTH = 128   # number of tokens per training sample
STRIDE = 64           # sliding window step size

# ---- Training settings ----
BATCH_SIZE = 8
LEARNING_RATE = 3e-4

# ---- Model settings (initial placeholders) ----
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4

# Which means in other files you do: from src.config import CONTEXT_LENGTH, STRIDE, BATCH_SIZE, etc.
# And then use context_length = CONTEXT_LENGTH, stride = STRIDE, etc. in your code.