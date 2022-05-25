import sys
import json

def file_mode(file_name, key_list):
    fin = open(file_name)
    my_str = fin.read()
    try:
        my_json = json.loads(my_str)
    except Exception:
        return False
    for key in key_list:
        try:
            my_json = my_json[key]
        except TypeError:
            my_json = my_json[int(key)]
    print(json.dumps(my_json, indent=4))
    return True

def line_mode(file_name, key_list):
    fin = open(file_name)

    for my_str in fin.readlines():
        try:
            my_json = json.loads(my_str)
        except Exception:
            return False
        for key in key_list:
            try:
                my_json = my_json[key]
            except TypeError:
                my_json = my_json[int(key)]
        print(json.dumps(my_json, indent=4))
    return True

def main():
    file_name = sys.argv[1]
    key_list = sys.argv[2:]
    print("Loading", file_name)
    if file_mode(file_name, key_list):
        return
    print("Failed in file_mode - trying line_mode")
    if line_mode(file_name, key_list):
        return
    print("Failed in line_mode - giving up")


if __name__ == "__main__":
    main()


