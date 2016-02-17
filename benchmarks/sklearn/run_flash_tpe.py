import os
import subprocess
import time
import numpy as np


rand_stamp = np.random.randint(10000, 99999)
call_lr_tpe = 'HPOlib-run -o ../../optimizers/lr/lr -s %d -a TPE' % rand_stamp
print 'Command:', call_lr_tpe
subprocess.call(call_lr_tpe, shell=True)
