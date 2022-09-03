
# -- api --
from . import original
from . import augmented
from . import warmup_scheduler
from . import configs


# -- for loading model --
from .utils.misc import optional
from .utils.model_utils import extract_search # set input params

def load_model(*args,**kwargs):
    attn_mode = optional(kwargs,"attn_mode","product_dnls")
    if attn_mode == "original":
        return original.load_model(*args,**kwargs)
    else:
        return augmented.load_model(*args,**kwargs)
