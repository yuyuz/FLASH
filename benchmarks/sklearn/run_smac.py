import os
import subprocess
import time
import numpy as np
from importlib import import_module
from HPOlib.format_converter.tpe_to_smac import convert_tpe_to_smac_from_object


# generate .pcs from space.py
module = import_module('space')
search_space = module.space
smac_space = convert_tpe_to_smac_from_object(search_space)
smac_space_file = 'smac_2_06_01-dev/params.pcs'
fh = open(smac_space_file, 'w')
fh.write(smac_space)
fh.close()
print ('Sapce file for SMAC generated: %s' % smac_space_file)

rand_stamp = np.random.randint(10000, 99999)
call_smac = 'HPOlib-run -o ../../optimizers/smac/smac_2_06_01-dev -s %d' % rand_stamp
print 'Command:', call_smac
subprocess.call(call_smac, shell=True)
