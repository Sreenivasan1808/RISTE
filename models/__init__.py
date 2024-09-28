# from resnet import *
# from I2C import *
# from transformers import *

# import resnet
# import I2C
# import transformers

# Define the __all__ variable
__all__ = ["resnet", "transformers", "I2C"]

# Import the submodules
from . import resnet
from . import transformers
from . import I2C