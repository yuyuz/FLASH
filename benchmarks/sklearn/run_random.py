import os
import subprocess
import time
import numpy as np


rand_stamp = np.random.randint(10000, 99999)
call_random = 'HPOlib-run -o ../../optimizers/tpe/random_hyperopt_august2013_mod -s %d' % rand_stamp
print 'Command:', call_random
subprocess.call(call_random, shell=True)
