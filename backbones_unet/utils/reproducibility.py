def set_seed(seed=45):
    """
    Sets the seed of the entire notebook 
    so results are the same every time we run.
    This is for REPRODUCIBILITY.
    seed: int, default=45
        In the context of machine and deep learning, a seed is a 
        randomly generated number that is used to initialize the 
        random number generator. This random number generator is 
        used by many algorithms in machine and deep learning, 
        including the initialization of model weights and the 
        shuffling of data during training. Using a seed ensures that 
        the random number generator produces the same sequence of 
        random numbers every time it is used, provided the seed value 
        is the same. This allows for greater reproducibility of 
        experiments, making it easier to compare results across 
        different runs of the same experiment. Set the seed value for these.
    """
    import numpy as np
    import random
    import torch
    import os

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print('> SEEDING DONE')