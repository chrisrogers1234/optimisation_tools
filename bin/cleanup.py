import time
import glob
import os

def cleanup(glob_file, dry_run):
    file_list = sorted(glob.glob(glob_file))
    for file in file_list:
        print(file)
        if not dry_run:
            os.remove(file)

def main():
    cleanup("output/musr_cooling*/*/tmp/*/output*txt", True)

if __name__ == "__main__":
    main()