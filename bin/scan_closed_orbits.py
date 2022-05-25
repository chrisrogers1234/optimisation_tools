
class ScanClosedOrbits(object):
    def __init__(self, sub_index, subs, seed, max_iterations):
        for key in sorted(subs.keys()):
            print(utilities.sub_to_name(key), subs[key], end=' ')
        print()
        out_dir = OUT_DIR
        run_dir = RUN_DIR
        tmp_dir = "./"
        try:
            os.makedirs(run_dir)
        except OSError: # maybe the dir already exists
            pass
        os.chdir(run_dir)
        print("Running in", os.getcwd())
        common.substitute(CONFIG.tracking["lattice_file"], tmp_dir+'SectorFFAGMagnet.tmp', subs)
        energy = subs["__energy__"]
        ref_hit = reference(energy)
        opal_exe = os.path.expandvars("${OPAL_EXE_PATH}/opal")
        tracking = OpalTracking(tmp_dir+'/SectorFFAGMagnet.tmp', tmp_dir+'/disttest.dat', ref_hit, 'PROBE*.loss', opal_exe, tmp_dir+"/log")
        seed_hit = ref_hit.deepcopy()

    def reference(energy):
        """
        Generate a reference particle
        """
        hit_dict = {}
        hit_dict["pid"] = CONFIG.find_closed_orbits["pdg_pid"]
        hit_dict["mass"] = common.pdg_pid_to_mass[abs(hit_dict["pid"])]
        hit_dict["charge"] = 1
        hit_dict["x"] = 0.
        hit_dict["px"] = 0.
        hit_dict["kinetic_energy"] = energy
        return Hit.new_from_dict(hit_dict, "pz")

    def plot_iteration(sub_index, i, iteration, energy):
        """
        Plot the closed orbit ellipse and the ellipse fit
        """
        canvas, hist, graph, fit = iteration.plot_ellipse("x", "px", "mm", "MeV/c")
        hist.SetTitle('KE='+str(energy)+' iter='+str(i))
        canvas.Update()
        name = os.path.join(OUT_DIR, "closed_orbit-sub-index_"+str(sub_index)+"-i_"+str(i))
        for format in "eps", "root", "png":
            canvas.Print(name+"."+format)

    def find_closed_orbit():
        """
        Find the closed orbit; algorithm is to track turn by turn; fit an ellipse to
        the tracking; find the centre of the ellipse; repeat until no improvement or
        10 iterations.
        - energy: (float) kinetic energy at which the co is calculated
        - step: (float) step size in tracking
        - poly_order: (int) order of the polynomial fit to the field map (not used)
        - smooth_oder: (int) order of smoothing the polynomial fit to the field map
                      (not used)
        - seed: (list of 2 floats) [x, px] value to be used as the seed for the next 
                iteration; px value is ignored, sorry about that.
        """
        seed_hit["x"] = seed[0]
        seed_hit["px"] = seed[1]
        os.chdir(out_dir)
        if i > 0:
            plot_iteration(sub_index, i, iteration, energy)
        return tracking.last[0]
