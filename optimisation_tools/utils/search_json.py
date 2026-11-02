import fnmatch

def search_json(pattern : str, data) -> tuple:
    """
    Search through a json data structure for keys or paths matching a given pattern.
    @param pattern: A search pattern with wildcards (*, ?) to match keys or paths. Dictionary keys are separated by
            full stops, while array indices are enclosed in regular parentheses. Escape characters have not been
            implemented so that a(0) will return a hit with either {"a":[value]} or {"a(0)":value}
    @param data: Some json data structure.
    Returns a tuple of (matches, paths) where matches is a list of values corresponding to the matched keys/paths, and
    paths is a list of the corresponding paths.
    """
    return SearchJson().search_json(pattern, data)

class SearchJson:
    def __init__(self):
        self.data = {}

    def search_json(self, data, pattern):
        self.data = data
        matches, paths = self.search_json_recurse(pattern, "", self.data)
        return matches, paths

    def search_json_recurse(self, pattern, path, current_data):
        """
        Recursively searches current_data for keys or paths
        matching a given pattern and returns a list of their values.

        Parameters:
            pattern (str): A search pattern with wildcards (*, ?) to match keys or paths.
            path (str): Internal argument used to track the current traversal path.

        Returns:
            list: A list of values corresponding to the matched keys/paths.
        """
        matches = []
        paths = []
        if isinstance(current_data, dict):
            for key, value in current_data.items():
                current_path = f"{path}.{key}" if path else key
                # Match the key or full path
                if fnmatch.fnmatch(current_path, pattern):
                    matches.append(value)
                    paths.append(current_path)
                else:
                    # Recursively search in nested dictionaries or lists
                    this_matches, this_paths = self.search_json_recurse(pattern, current_path, value)
                    matches += this_matches
                    paths += this_paths
        elif isinstance(current_data, list):
            for index, value in enumerate(current_data):
                current_path = f"{path}({index})"
                # If wildcards are expected for array indices, adjust here
                if fnmatch.fnmatch(current_path, pattern):
                    matches.append(value)
                    paths.append(current_path)
                else:
                    # Recursively search in nested dictionaries or lists
                    this_matches, this_paths = self.search_json_recurse(pattern, current_path, value)
                    matches += this_matches
                    paths += this_paths
        return matches, paths

class Test:
    def __init__(self):
        self.plotter = PlotJson()
        self.plotter.data = {
            "a": 1,
            "b": 2,
            "c": {
                "d": 3,
                "e": 4
            },
            "f": [5, 6, 7]
        }

    def test_search_json(self):
        pattern = "f(*)"
        expected_matches = [3]
        actual_matches = self.plotter.search_json(pattern)
        print(actual_matches)

def main():
    Test().test_search_json()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
