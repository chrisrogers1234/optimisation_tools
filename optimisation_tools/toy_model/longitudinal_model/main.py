import json
import matplotlib
import matplotlib.pyplot
import optimisation_tools.toy_model.longitudinal_model.run_control as run_control


def main():
    run_numbers = [74, 113] # + [i for i in range(98, 108)] + [i for i in range(109, 114)]
    with open("2024-12-21_settings.json") as fin:
        config_json = json.loads(fin.read())
        for config in reversed(config_json):
            if config["data"]["file_number"] not in run_numbers:
                continue
            runner = run_control.RunControl()
            runner.setup(config)
            runner.run()

if __name__ == "__main__":
    main()
    #matplotlib.pyplot.show(block=False)
    #input("Press <CR> to finish")