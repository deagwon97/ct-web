import os
import sys

for library in sys.argv[1:]:
    os.system('python3 -m pip install {}'.format(library))