from pathlib import Path
import torch

ROOT        = Path(__file__).parents[1]
DATA_DIR    = ROOT / "data"
CKPT_DIR    = ROOT / "checkpoints"
OUTPUTS_DIR = ROOT / "outputs"

CKPT_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

device = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()           else
    torch.device("cpu")
)
