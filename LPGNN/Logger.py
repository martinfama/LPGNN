# Module to save logs, associated with results of a certain test (or several)

import numpy as np

## Should receive a "run" type object (not yet defined) which contains a list
## of information about the network tested on, the models used and their parameters,
## and the results of the tests. This should all be saved verbosely in a file, although
## a header containing a standard to recreate the test would be nice.