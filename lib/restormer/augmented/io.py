
from .restormer import Restormer
from ..utils.model_utils import load_checkpoint
from pathlib import Path


def load_model(*args,**kwargs):

    model = Restormer()

    fdir = Path(__file__).resolve().parents[0] / "../../../" # parent of "./lib"
    fdir = fdir.resolve()
    weights = fdir / "./weights/motion_deblurring.pth"
    load_checkpoint(model, weights)

    return model

