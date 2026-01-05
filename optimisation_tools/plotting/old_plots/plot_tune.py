import shutil
import os
import glob
import sys
import json
import math
import xboa.common
import numpy

from utils import utilities


def main(fglob):
    for file_name in glob.glob(fglob):
        plot_dir = os.path.split(file_name)[0]+"/plot_tune/"
        if os.path.exists(plot_dir):
            shutil.rmtree(plot_dir)
        os.makedirs(plot_dir)
        try:
            data = load_file(file_name)
        except IndexError:
            continue
        plot_data_1d(data, 'fractional cell tune', plot_dir, None, 1, False) # cell tune or ring tune
        plot_data_2d(data, 'fractional cell tune', plot_dir) # cell tune or ring tune, plot x tune vs y tune

if __name__ == "__main__":
    main(sys.argv[1])
    input("Press <CR> to end")
