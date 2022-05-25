import numpy
import ROOT
import xboa.common as common
import da_finder

P_MASS = common.pdg_pid_to_mass[2212]
MIN_N_CELLS = 100
CELL_STEP = 10

def cleanup(da_element):
    cell_dirty_ending = -5
    da_element['ekin'] = get_ekin(da_element)
    for key in 'da_1', 'da_2', 'da_3', 'ref':
        new_list = []
        if key not in da_element:
            continue
        for seed_x, points_list in da_element[key]:
            if len(points_list) < MIN_N_CELLS:
                continue
            points_list = points_list[:cell_dirty_ending] # remove last few points - as the event is wandering out of aperture?
            points_list = [point for i, point in enumerate(points_list) if (i-1) % CELL_STEP == 0] # turns only; nb: after lots of debugging, I came to the realisation that we rely on cell number is odd here (not even)
            new_list.append([seed_x, points_list])
            #print key, seed_x, points_list[0]
        da_element[key] = new_list
    return da_element

def get_acceptance(da_element, eigenspace_key, da_eigenspace):
    global P_MASS
    tm = da_finder.get_tm(da_element, False)
    largest_aperture = None
    for seed, points_list in da_element[eigenspace_key]:
        largest_aperture = seed, points_list
    if largest_aperture == None:
        return [0.]
    n1 = da_eigenspace*2
    n2 = da_eigenspace*2+2
    v_t = tm.v_t[n1:n2, n1:n2]
    v_t_inv = numpy.linalg.inv(v_t)
    amplitude_list = []
    for point in largest_aperture[1]:
        point = numpy.array(point[n1:n2])
        point_trans = numpy.transpose(point)
        amplitude = numpy.dot(point_trans, numpy.dot(v_t_inv, point))/P_MASS
        amplitude_list.append(amplitude)
    return amplitude_list

def get_ekin(da_element):
    ptot = (da_element['pphi']**2+da_element['py']**2+da_element['pr']**2)**0.5
    ekin = (ptot**2+P_MASS**2)**0.5 - P_MASS
    return ekin

def ekin_to_p(ekin):
    return ((abs(ekin)+P_MASS)**2-P_MASS**2)**0.5

def plot_one(da_element, do_time = False):
    global P_MASS, MIN_N_CELLS, CELL_STEP
    tm = da_finder.get_tm(da_element)
    plot_dir = "baseline/"
    pphi = da_element['pphi']
    ptot = (da_element['pphi']**2+da_element['py']**2+da_element['pr']**2)**0.5
    ekin = da_element['ekin']
    da_spaces = ['ref', 'da_1', 'da_2', 'da_3']
    if do_time:
        eigenspaces = list(range(3))
        physical_spaces = list(range(4))
    else:
        eigenspaces = list(range(2))
        physical_spaces = list(range(2))
    for key in da_spaces:
        if key not in da_element:
            continue
        da_eigenspace = {'ref':'ref', 'da_1':0, 'da_2':1, 'da_3':2}[key]
        for plot_eigenspace in eigenspaces:
            print("Doing key", key, "plot", plot_eigenspace)
            point_x_list = []
            point_px_list = []
            for seed_x, points_list in da_element[key]:
                print("   ...", key, "seed", seed_x, "with", len(points_list), "turns")
                point_x_list += [point[2*plot_eigenspace] for point in points_list]
                point_px_list += [point[2*plot_eigenspace+1] for point in points_list]
                print(point_x_list[-10:])
            name = "eigenspace_p_"+str(round(ptot, 1))+"-da_"+str(da_eigenspace)+"-plot_"+str(plot_eigenspace)
            canvas = common.make_root_canvas(name)
            canvas.Draw()
            canvas.cd()
            hist, graph = common.make_root_graph("da_"+str(plot_eigenspace),
                                    point_x_list, "u_{"+str(2*plot_eigenspace)+"}",
                                    point_px_list, "u_{"+str(2*plot_eigenspace+1)+"}")
            title = "DA for eigenspace: "+str(da_eigenspace)+\
                    " P: "+str(round(ptot, 1))+" MeV/c and E_{k}: "+\
                    str(round(ekin, 1))+" MeV requiring "+str(MIN_N_CELLS/CELL_STEP)+" turns"
            print(title)
            hist.SetTitle(title)
            hist.Draw()
            graph.Draw("SAMEP")
            graph.SetMarkerStyle(7)
            canvas.Update()
            for format in "png", "root", "pdf":
                canvas.Print(plot_dir+name+"."+format)
        physical_axes = ['x [mm]', 'px [MeV/c]', 'y [mm]', 'py [MeV/c]', 't [ns]', 'KE [MeV]', 't [ns]', 'p_{tot} [MeV/c]']
        for plot_physical_space in physical_spaces:
            point_x_list = []
            point_px_list = []
            for seed_x, points_list in da_element[key]:
                points_list = points_list[::]
                points_list = [tm.coupled(point) for point in points_list]
                if plot_physical_space < 3:
                    point_x_list += [point[2*plot_physical_space] for point in points_list]
                    point_px_list += [point[2*plot_physical_space+1] for point in points_list]
                elif plot_physical_space == 3:
                    point_x_list += [point[4] for point in points_list]
                    point_px_list += [ekin_to_p(point[5]+ekin) for point in points_list]

            if len(point_x_list) < MIN_N_CELLS:
                continue
            name = "physical_p_"+str(round(ptot, 1))+"-da_"+str(da_eigenspace)+"-plot_"+str(plot_physical_space)
            canvas = common.make_root_canvas(name)
            canvas.Draw()
            canvas.cd()
            hist, graph = common.make_root_graph("da_"+str(plot_physical_space),
                                    point_x_list, physical_axes[2*plot_physical_space],
                                    point_px_list, physical_axes[2*plot_physical_space+1])
            title = "DA for eigenspace: "+str(da_eigenspace)+\
                    " P: "+str(round(ptot, 1))+" MeV/c and E_{k}: "+\
                    str(round(ekin, 1))+" MeV requiring "+str(MIN_N_CELLS/CELL_STEP)+" turns"
            print(title)
            hist.SetTitle(title)
            hist.Draw()
            graph.Draw("SAMEP")
            graph.SetMarkerStyle(7)
            canvas.Update()
            for format in "png", "root", "pdf":
                canvas.Print(plot_dir+name+"."+format)

ERRORS_GRAPH = []
def plot_acceptance(da_list, eigenspace_key):
    global ERRORS_GRAPH, MIN_N_CELLS
    plot_dir = "baseline/"
    acceptance_list_0 = []
    acceptance_list_1 = []
    pphi_list = []
    mean_acceptance_list_0 = []
    mean_acceptance_list_1 = []
    mean_pphi_list = []
    means_0 = ROOT.TGraphErrors(len(da_list))
    means_0.SetMarkerStyle(21)
    means_0.SetMarkerColor(8)
    means_0.SetLineColor(8)
    means_1 = ROOT.TGraphErrors(len(da_list))
    means_1.SetMarkerStyle(21)
    means_1.SetMarkerColor(4)
    means_1.SetLineColor(4)
    ERRORS_GRAPH.append(means_0)
    ERRORS_GRAPH.append(means_1)
    for i, element in enumerate(da_list):
        ekin = get_ekin(element)
        acceptance_list = get_acceptance(element, eigenspace_key, 0)
        if len(acceptance_list) < 5:
            continue
        acceptance_list_0 += acceptance_list
        means_0.SetPoint(i, ekin, numpy.mean(acceptance_list))
        means_0.SetPointError(i, 0., 0.) #numpy.std(acceptance_list))
        acceptance_list = get_acceptance(element, eigenspace_key, 1)
        acceptance_list_1 += acceptance_list
        means_1.SetPoint(i, ekin, numpy.mean(acceptance_list))
        means_1.SetPointError(i, 0., 0.) #numpy.std(acceptance_list))
        pphi_list += [ekin for item in acceptance_list]
    canvas = common.make_root_canvas("acceptance "+eigenspace_key)
    canvas.Draw()
    axes, graph = common.make_root_graph('axes', pphi_list, "E_{kin} [MeV]", acceptance_list_0, "A_{2D} [mm]", ymin=-0.01, ymax=20.)
    axes.Draw()
    da_eigenspace = {'da_1':0, 'da_2':1, 'da_3':2}[eigenspace_key]
    axes.SetTitle("Acceptance for eigenspace "+str(da_eigenspace)+" requiring "+str(MIN_N_CELLS/CELL_STEP)+" turns")
    for acceptance_list, color in [(acceptance_list_0, 8), (acceptance_list_1, 4)]:
        dummy, graph = common.make_root_graph('axes', pphi_list, "P_{phi} [MeV/c]", acceptance_list, "A_{2D} [mm]")
        graph.SetMarkerStyle(7)
        graph.SetMarkerColor(color)
        graph.Draw('pSAME')
        means_0.Draw('PSAME')
        means_1.Draw('PSAME')
    canvas.Update()
    name = "acceptance_eigenspace_"+str(da_eigenspace)
    for format in "png", "root", "pdf":
        canvas.Print(plot_dir+name+"."+format)
        
def main():
    da_list = da_finder.load_closed_orbits("baseline/closed_orbits_1_MeV_with_da_tmp")
    print(len(da_list), list(da_list[0].keys()))
    da_list = [cleanup(da_element) for da_element in da_list]
    for da_element in []: #da_list:
        try:
            plot_one(da_element, do_time = False)
        except IndexError:
            pass
    plot_acceptance(da_list, "da_1")
    plot_acceptance(da_list, "da_2")
    if "da_3" in list(da_list[0].keys()):
        plot_acceptance(da_list, "da_3")


if __name__ == "__main__":
    main()
    print("Finished - press <CR> to close")
    input()

