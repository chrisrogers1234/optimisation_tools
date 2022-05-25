import glob
import json
import os

import utilities

def print_substitutions(file_name):
    fin = open(file_name)
    print(file_name)
    for line in fin.readlines():
        subs = json.loads(line)["substitutions"]
        print("   ", end=' ')
        for key in sorted(subs.keys()):
            print(utilities.sub_to_name(key), subs[key], end=' ')
        print()


def main():
    base_dir = "output/"
    for file_name in glob.glob(base_dir+"*/find_tune"):
        plot_dir = os.path.split(file_name)[0]
        try:
            print_substitutions(file_name)
        except IndexError:
            continue

if __name__ == "__main__":
    main()
