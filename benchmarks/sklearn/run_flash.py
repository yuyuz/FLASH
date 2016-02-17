import os
import subprocess
import time
import glob
import cPickle
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
call_lr = 'HPOlib-run -o ../../optimizers/lr/lr -s %d -a SMAC' % rand_stamp
print 'Command:', call_lr
subprocess.call(call_lr, shell=True)

# get the experiment directory
cwd = os.getcwd()
dirs = [v for v in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, v)) and str(rand_stamp) in v]
print 'dirs:', dirs
assert len(dirs) == 1
exp_dir = dirs[0]
os.chdir(exp_dir)

# run SMAC
call_smac = 'HPOlib-run -o ../../../optimizers/smac/smac_2_06_01-dev -s %d' % rand_stamp
print 'Command:', call_smac
subprocess.call(call_smac, shell=True)

# merge pickle files of LR and SMAC
fh = open('lr.pkl', 'r')
log_lr = cPickle.load(fh)
fh.close()

smac_pkl = glob.glob('smac*/smac*.pkl')
print 'smack_pkl:', smac_pkl
assert len(smac_pkl) == 1
fh = open(smac_pkl[0])
log_smac = cPickle.load(fh)
fh.close()

log_merged = dict()
for field in log_lr:
    if type(log_lr[field]) is list:
        log_merged[field] = log_lr[field] + log_smac[field]
    elif field == 'total_wallclock_time':
        log_merged[field] = log_lr[field] + log_smac[field]
    else:
        log_merged[field] = log_lr[field]

output = 'flash.pkl'
fh = open(output, 'w')
cPickle.dump(log_merged, fh)
fh.close()

print 'Pickle files merged:', output
