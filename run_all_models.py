import numpy as np
from parameters import *
import model
import sys

gpu_id = sys.argv[1]

save_fn = 'test' + '_' + str(0) + '.pkl'
updates = {'save_fn': save_fn}
update_parameters(updates)
model.main(gpu_id = gpu_id)
