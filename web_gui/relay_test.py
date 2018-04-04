## Importing the relay module does nothing.
import relay
from numpy import *

## Send data over.
import time
for t in linspace( 0, 1, 100 ):
    relay.send_data( [(t,t,t), (t,t,t)] )
    time.sleep(0.1)
