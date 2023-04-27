import shutil
import os
import subprocess

import xboa.hit

import optimisation_tools.opal_tracking._opal_tracking
OpalTracking = optimisation_tools.opal_tracking._opal_tracking.OpalTracking
StoreDataInMemory = optimisation_tools.opal_tracking._opal_tracking.StoreDataInMemory


class PyOpalTracking(OpalTracking):
	def __init__(self, script_filename, beam_filename, reference_hit, output_filename, log_filename = None, n_cores = 1, mpi = "mpirun"):
		opal_path = subprocess.check_output(["which", "python"]).rstrip()
		subprocess.check_output([opal_path, "--version"])
		super().__init__(script_filename, beam_filename, reference_hit, output_filename, opal_path, log_filename, n_cores, mpi)

def main():
	a_dir = "output/test_pyopaltracking"
	here = os.getcwd()
	try:
		shutil.rmtree(a_dir)
	except OSError:
		pass
	os.makedirs(a_dir)
	os.chdir(a_dir)
	lattice_name = os.path.join(here, "lattice/fets_ffa.py")
	print(lattice_name)
	tracking = PyOpalTracking(lattice_name, "disttest.dat", xboa.hit.Hit(), "ring_probe_*.h5", "log")
	tracking.set_file_format("hdf5")
	tracking.pass_through_analysis = StoreDataInMemory({})
	tracking.track_many([])
	os.chdir(here)


if __name__ == "__main__":
	main()