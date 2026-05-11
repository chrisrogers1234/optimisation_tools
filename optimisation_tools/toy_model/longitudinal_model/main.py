import json
import matplotlib
import matplotlib.pyplot
import optimisation_tools.toy_model.longitudinal_model.run_control as run_control


def main():
    run_numbers = [0] # + [i for i in range(98, 108)] + [i for i in range(109, 114)]
    with open("2026-03-06_v3_settings.json") as fin:
        config_json = json.loads(fin.read())
        for config in config_json[:-1]:
            if config["data"]["file_number"] not in run_numbers:
                continue
            runner = run_control.RunControl()
            runner.setup(config)
            runner.run()

if __name__ == "__main__":
    main()
    #matplotlib.pyplot.show(block=False)
    #input("Press <CR> to finish")