from .encoder import *
from .decoder import *
from .decoder_TRM import *
from .decoder_modified import *
from .capmodel import CapModel
from .attention_TRM import *
def setup(opt, vocab):
    try:
        mod = __import__('.'.join(['models', opt.model]), fromlist=['Model'])
        model = getattr(mod, 'CapModel')(opt, vocab)
    except:
        raise Exception("Model not supported: {}".format(opt.model))

    return model
