import os
import subprocess
import json
import glob
import shutil
import tempfile

class Gifseriser:
    def __init__(self):
        self.delay = 100
        self.loop_count = 0


    def get_files(self, file_glob, strip_stuff):
        file_list = glob.glob(file_glob)
        item_list = []
        for file_name in file_list:
            item = {"file_name":file_name}
            my_kv_list = file_name.split(";_")
            my_kv_list = [item.split("=") for item in my_kv_list]
            for kv_pair in my_kv_list:
                for stripper in strip_stuff:
                    kv_pair[0] = kv_pair[0].replace(stripper, "")
                    kv_pair[1] = kv_pair[1].replace(stripper, "")
            item["kv_list"] = dict(my_kv_list)
            item_list.append(item)
        return item_list

    def setup_gif_dir(self, item_list):
        src_dir = os.path.split(item_list[0]["file_name"])[0]
        gif_dir = f"{src_dir}/gifsicle/"
        if os.path.exists(gif_dir):
            return gif_dir
        os.makedirs(gif_dir)
        return gif_dir


    def generate_gif(self, item_list, output_file_name):
        gif_file_list = []
        gif_dir = self.setup_gif_dir(item_list)
        print()
        for item in item_list:
            file_name = item["file_name"]
            gif_file_name = file_name.replace(".png", ".gif")
            gif_file_name = os.path.split(gif_file_name)[1]
            gif_file_name = f"{gif_dir}/{gif_file_name}"
            if not os.path.exists(gif_file_name):
                subprocess.check_output(["convert",  file_name, gif_file_name])
                print("\rConverting", file_name, gif_file_name, end=" ")
            gif_file_list.append(gif_file_name)
        print()
        command = ["gifsicle", f"--delay={self.delay}", "--output="+output_file_name, f"--loopcount={self.loop_count}"]+gif_file_list
        print(f"running command\n  "+" ".join(command))
        proc = subprocess.run(command)

def generate_webp(input_glob, output_file_name, frame_duration):
    png_file_list = []
    file_list = sorted(glob.glob(input_glob))
    if len(file_list) > 9999:
        raise RuntimeError("Max 9999 frames supported")
    tmp_dir = tempfile.TemporaryDirectory()
    tmp_name = os.path.join(tmp_dir.name, "tmp.webp")
    for i, a_file in enumerate(file_list):
        a_name = os.path.join(tmp_dir.name, f"frame_{i:04d}.png")
        shutil.copy(a_file, a_name)
        print(f"copy {a_file} {a_name}")
    tmp_files = os.path.join(tmp_dir.name, "frame_%4d.png")
    command = ["ffmpeg", "-i", tmp_files, tmp_name]
    print("Command line", " ".join(command))
    proc = subprocess.run(command)
    # now slow it down
    command = ["webpmux", "-duration", str(frame_duration), tmp_name, "-o", output_file_name]
    proc = subprocess.run(command)
    os.unlink(tmp_name)

def main_not():
    gifseriser = Gifseriser()
    gifseriser.delay = 20
    run_dir_glob = "/home/cr67/work/2025-01-01_low-energy-cooling/rogers-low-energy-cooling/output/induction_v47/name=run*/"
    for run_dir in glob.glob(run_dir_glob):
        for prefix in [f"time_z-ke", "time_x-px"]: #"plane_{prefix}"
            glob_file = f"{run_dir}/{prefix}_*.png"
            #glob_file = f"{run_dir}/*.png"
            item_list = glob.glob(glob_file)#gifseriser.get_files(run_dir+prefix+"_*.png", [run_dir, ".png"])
            item_list = [{"file_name":item} for item in item_list]
            item_list = sorted(item_list, key = lambda item: item["file_name"]) # float(item["kv_list"]["z"]))
            if len(item_list) == 0:
                print(f"Failed to find anything in glob {glob_file}")
            print(json.dumps(item_list, indent=2))
            out_file = run_dir+"animation_"+prefix+".gif"
            gifseriser.generate_gif(item_list, out_file)
            print("Output in", out_file)

def main():
    gifseriser = Gifseriser()
    gifseriser.delay = 10
    #dir_glob = "/home/cr67/work/2020-07-06_mc/final_cooling/low-energy-cooling/output/induction_v14/1/0/"
    #dir_glob = "/home/cr67/work/2020-07-06_mc/final_cooling/low-energy-cooling-g4bl/g4bl_low_energy_cooling/plots/"
    dir_glob = "/home/cr67/work/2017-07-07_isis2/horizontal_isis3/output/foil_heating_test_fets_ring_v2/"
    for a_dir in glob.glob(dir_glob):
        run_dir = a_dir
        #glob_file = f"{run_dir}/plane_*.png"
        glob_file = f"{run_dir}/foil_temp_*.png"
        #glob_file = f"{run_dir}/z_vs_kinetic_energy_*.png"
        generate_webp(glob_file, f"{run_dir}/animation.webp", 100)
    return

    for a_dir in glob.glob("output/2024-12-21_v4/job_0111"):
        run_dir = a_dir
        glob_file = f"{run_dir}/longitudinal_*.png"
        generate_webp(glob_file, f"{run_dir}/longitudinal_animation.webp", 500)


if __name__ == "__main__":
    main_not()
