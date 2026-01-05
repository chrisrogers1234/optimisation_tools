import matplotlib.pyplot
import csv
import math

def enge(delta_s, c10, eta, lambda_end):
    try:
        value = 1/(1+math.exp(c10*math.cos(eta)*delta_s/lambda_end))
    except OverflowError:
        return 0.0
    return value

def function(s0):
    b_scale = 0.4067
    r0 = 4.0
    s = s0
    eta = math.radians(45)
    c10 = 3.91
    lambda_long = 0.140
    lambda_short = 0.070
    start_gap = 2*math.pi*r0/80*1.2
    f_length = 0.3142
    fd_gap = 0.1571
    d_length = 0.1571
    df_gap = 0.9425
    bf = -1.0*b_scale
    bd = 0.75677*b_scale
    initial_offset = start_gap+0.01 #+math.pi*r0*(1./2-1./8)

    field = 0
    delta_s = -s+initial_offset
    field +=  bf*enge(delta_s, c10, eta, lambda_long)
    delta_s = -s+initial_offset+f_length
    field += -bf*enge(delta_s, c10, eta, lambda_short)
    delta_s = -s+initial_offset+f_length+fd_gap
    field +=  bd*enge(delta_s, c10, eta, lambda_short)
    delta_s = -s+initial_offset+f_length+fd_gap+d_length
    field += -bd*enge(delta_s, c10, eta, lambda_short)
    return field*10

def do_plot_2():
    s_list = [-2*math.pi*4.0/16/1000*i for i in range(1001)]
    by_list = [function(s) for s in s_list]
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(s_list, by_list)

def get_bins(x_list):
    x_points = sorted(list(set(x_list)))
    x_bins = [(x-x_points[i+1])/2+x for i, x in enumerate(x_points[:-1])]
    x_bins.append(x_bins[-1]+(x_bins[-1]-x_bins[-2])/2.0)
    return x_bins

def do_plot_3(file_name, max_bz_plot=None):
    fin = open(file_name)
    data = [row for row in fin.readlines()][13:]
    data = [row.split() for row in data]
    data = [[float(element) for element in row] for row in data]
    x_list = [row[0] for row in data]
    y_list = [row[1] for row in data]  
    x_bins = get_bins(x_list)
    y_bins = get_bins(y_list)
    bz_list = [row[6] for row in data]
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    min_z, max_z = min(bz_list), max(bz_list)
    if max_bz_plot == None:
        max_bz_plot = max([abs(min_z), max_z])
    hist = axes.hist2d(x_list, y_list, bins=[x_bins, y_bins], weights=bz_list,
        cmin=min_z, cmax=max_z, cmap="PiYG", vmin=-max_bz_plot, vmax=max_bz_plot)
    figure.colorbar(hist[3], ax=axes)
    return figure


def do_plot(file_name, r0):
    fin = open(file_name)
    data = [row for row in fin.readlines()][13:]
    data = [row.split() for row in data]
    data = [[float(element) for element in row] for row in data]
    theta_list = [row[1] for row in data]
    s_list = [theta/180.0*math.pi*r0 for theta in theta_list]  
    by_list = [row[6] for row in data]
    figure = matplotlib.pyplot.figure()
    axes = figure.add_subplot(1, 1, 1)
    axes.plot(s_list, by_list)
    #by_ref = [function(s) for s in s_list]
    #axes.plot(s_list, by_ref)
    axes.set_xlabel("s [m]")
    axes.set_ylabel("by [kGauss]")
    axes = axes.twiny()
    axes.plot(theta_list, by_list)
    axes.set_xlabel("$\\theta$ [$^\\circ$]")
    axes.set_ylabel("by [kGauss]")
    return figure

def main():
    r0 = 4.0
    fig1 = do_plot("output/2022-07-01_baseline/baseline_reversed/tmp/find_closed_orbits/data/Fields1.dat", r0)
    fig1.savefig("fields1.png")
    fig2 = do_plot("output/2022-07-01_baseline/baseline_reversed/tmp/find_closed_orbits/data/Fields2.dat", r0)
    fig2.savefig("fields2.png")
    
    do_plot_2()
    #do_plot_3("data/FieldsXY.dat", 1.0)

def main():
    r0 = 10.0
    fig1 = do_plot("output/muon_ffynchrotron/tmp/find_bump/data/Fields1.dat", r0)
    fig1.savefig("fields1.png")
    fig2 = do_plot("output/muon_ffynchrotron/tmp/find_bump/data/Fields2.dat", r0)
    fig2.savefig("fields2.png")


if __name__ == "__main__":
    main()
    matplotlib.pyplot.show(block=False)
    input("Press <CR> to finish")
