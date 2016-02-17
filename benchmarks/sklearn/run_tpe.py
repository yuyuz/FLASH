import os
import subprocess
import time
import numpy as np


rand_stamp = np.random.randint(10000, 99999)
call_tpe = 'HPOlib-run -o ../../optimizers/tpe/hyperopt_august2013_mod -s %d' % rand_stamp
print 'Command:', call_tpe
subprocess.call(call_tpe, shell=True)
