import os
import subprocess

class G4BLElement:
    def __init__(self):
        self.type = ""
        self.name = ""
        self.element_data = {}
        self.place_data = {}

    def get_g4bl_string(self):
        g4bl_string = f"{self.type} {self.name} "
        for key, value in reversed(self.element_data.items()):
            g4bl_string += f"{key}={value} "
        g4bl_string = g4bl_string[:-1]+"\n"
        if self.place_data and len(self.place_data):
            g4bl_string += f"place {self.name} "
            for key, value in reversed(self.place_data.items()):
                g4bl_string += f"{key}={value} "
            g4bl_string = g4bl_string[:-1]+"\n"
        g4bl_string += "\n"
        return g4bl_string

class G4BLLattice:
    def __init__(self):
        self.elements = []
        self.filename = "lattice.g4bl"

    def build_lattice_file(self):
        fout = open(self.filename, "w")
        for element in self.elements:
            g4bl_string = element.get_g4bl_string()
            fout.write(g4bl_string)
        fout.close()

    def execute(self, g4bl_path):
        command = [g4bl_path, self.filename]
        print("Running", command, "in dir", os.getcwd())
        with open("log", "w") as logfile:
            proc = subprocess.run(
                    command,
                    stdout=logfile, stderr=subprocess.STDOUT)
        print("   ... completed with return code", proc.returncode)
        if proc.returncode:
            raise RuntimeError("G4BL did not execute successfully")
