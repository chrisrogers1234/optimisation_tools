import bisect
import math
import scipy.constants

class RFBucket(object):
    def __init__(self, real_estate_gradient, frequency, ref_time, ref_energy, ref_mass, t0, t1, e0, e1):
        self.real_estate_gradient = real_estate_gradient
        self.frequency = frequency
        self.ref_time = ref_time
        self.ref_energy = ref_energy
        self.ref_mass = ref_mass
        self.central_p = self.p(self.ref_energy) # NOT the momentum of the synchronous particle... RF can be tuned off momentum
        self.e0 = e0
        self.e1 = e1
        self.t0 = t0
        self.t1 = t1
        self.dz = 0.01
        self.omega = self.frequency*2*math.pi
        self.y_var = "p"
        self.acceptance_contour = None
        # we can do E = E0 + V_{re}[sin(phi) - sin(phi_s)] (accelerating)
        # or E = E0 + V_{ref}[sin(phi)] - dE0 (foil)
        # it doesnt matter which; both are implemented
        self.phi_s = 0.0
        self.pseudo_foil_de = 0.0

        self.contours = []

    def get_contour_by_area(self, area, area_tolerance):
        # 0 is the lower bound
        dt_0, dt_1 = self.fixed_points()
        area_0 = 0.0
        area_1 = self.get_contour(dt_1-1e-3, self.ref_energy)["area"]
        #print("Get contour by area... dt", dt_0, dt_1, "area", area_0, area_1, "target", area)
        if area > area_1:
            raise RuntimeError("RF bucket not big enough to find this contour")
        while True:
            #print("                       dt", dt_0, dt_1, "area", area_0, area_1, "target", area)
            new_dt = (dt_1+dt_0)/2
            new_contour = self.get_contour(new_dt, self.ref_energy)
            new_area = new_contour["area"]
            if abs(new_area-area) < area_tolerance:
                return new_contour
            if new_area > area:
                contour_1 = new_contour
                dt_1 = new_dt
                area_1 = new_area
            else:
                contour_0 = new_contour
                dt_0 = new_dt
                area_0 = new_area

    def add_acceptance_contour(self, area):
        if area < 1e-12:
            raise ValueError("Area "+str(area)+" was too close to 0 ")
        contour = self.get_contour_by_area(area, area*1e-3)
        contour["colour"] = "orange"
        self.acceptance_contour = contour
        self.contours.append(contour)
        self.contours = sorted(self.contours, key = lambda c: c["area"])

    def plot_contours(self, axes):
        for contour in self.contours:
            axes.plot(contour["t"], contour["e"],  color=contour["colour"])

    def add_contour(self, time):
        a_contour = self.get_contour(time, self.ref_energy)
        self.contours.append(a_contour)
        self.contours = sorted(self.contours, key = lambda c: c["area"])

    def p(self, total_energy):
        return (total_energy**2-self.ref_mass**2)**0.5

    def phi(self, time):
        phi = (time-self.ref_time)*self.frequency*2.0*math.pi+self.phi_s
        return phi
        while phi > math.pi*2.0:
            phi -= math.pi*2.0
        while phi < 0.0:
            phi += math.pi*2.0
        return phi

    def fixed_points(self):
        """Return stable and unstable fixed points [ns]"""
        # dE = self.real_estate_gradient*(math.sin(phi+self.phi_s))*self.dz-self.pseudo_foil_de
        # dE = 0 -> sin(f+f_s) = (pfde)/V_re/dz
        const = self.pseudo_foil_de/self.real_estate_gradient/self.dz
        phi_ref = math.asin(const)-self.phi_s
        fixed_point_0 = phi_ref/self.frequency/2.0/math.pi+self.ref_time
        fixed_point_1 = (math.pi-phi_ref)/self.frequency/2.0/math.pi+self.ref_time
        return fixed_point_0, fixed_point_1


    def geometric_phi(self, time, energy):
        return math.atan2(energy-self.ref_energy, -time+self.ref_time)

    def geometric_radius(self, time, energy):
        return ((energy-self.ref_energy)**2 +(-time+self.ref_time)**2)**0.5

    def e_var(self, energy):
        if self.y_var == "energy":
            return energy
        elif self.y_var == "kinetic_energy":
            return energy-self.ref_mass
        elif self.y_var == "p":
            return self.p(energy)
        elif self.y_var == "dp_over_p":
            return (self.p(energy)-self.central_p)/self.central_p
        else:
            raise RuntimeError("Did not recognise y_var '"+str(self.y_var)+"'")

    def periodicarise(self, time, harmonic_number):
        t0 = time*self.frequency/harmonic_number
        t0 = (t0-math.floor(t0))*harmonic_number/self.frequency
        return t0

    def get_contour(self, time, energy):
        t_list, e_list, phi_list, r_list = [time], [self.e_var(energy)], [self.geometric_phi(time, energy)], [self.geometric_radius(time, energy)]
        if time == self.ref_time and energy == self.ref_energy:
            return t_list, e_list
        c_light = scipy.constants.c*1e-6 # mm/ns
        dtrefdz = (self.ref_energy**2/(self.ref_energy**2-self.ref_mass**2))**0.5/c_light

        is_closed = False
        while time <= self.t1 and time >= self.t0 and energy <= self.e1 and energy >= self.e0:
            dtdz = (energy**2/(energy**2-self.ref_mass**2))**0.5/c_light
            time = time+(dtdz-dtrefdz)*self.dz
            phi = self.phi(time)
            energy = energy+self.real_estate_gradient*(math.sin(phi))*self.dz-self.pseudo_foil_de
            t_list.append(time)
            e_list.append(self.e_var(energy))
            phi_list.append(self.geometric_phi(time, energy))
            r_list.append(self.geometric_radius(time, energy))
            if abs(phi_list[-1] - phi_list[-2]) > math.pi:
                is_closed = True
                break
        area = self.contour_area(t_list, e_list)
        contour = {
                "t":[t for t in reversed(t_list)],
                "e":[e for e in reversed(e_list)],
                "phi":[phi for phi in reversed(phi_list)],
                "r":[r for r in reversed(r_list)],
                "area":area,
                "is_closed":is_closed,
                "colour":"lightgrey",
            }
        return contour

    def inside_acceptance(self, time, dp_over_p, verbose):
        if self.acceptance_contour == None:
            return False
        p = dp_over_p*self.central_p+self.central_p
        energy = (p**2+self.ref_mass**2)**0.5
        phi = self.geometric_phi(time, energy)
        r = self.geometric_radius(time, energy)
        i1 = bisect.bisect_right(self.acceptance_contour["phi"][1:], phi)+1
        #is_inside = r <= self.acceptance_contour["r"][index]
        i0 = i1-1
        # distance from a point (xp, yp) to a line  (ax+by+c=0) is given by
        # (a xp + b yp + c) / sqrt(a^2 + b^2)
        # use line from i0 to i1 having a = x1-x0, b = y0-y1, c = (y1-y0)x0 - (x1-x0)y0 = y1x0-x1y0
        x1, x0 = self.acceptance_contour["t"][i1], self.acceptance_contour["t"][i0]
        y1, y0 = self.acceptance_contour["e"][i1], self.acceptance_contour["e"][i0]
        # we are only interested in the sign - so ignore the square root term
        distance = (x1-x0)*dp_over_p + (y0-y1)*time + (y1*x0-x1*y0)
        is_inside = distance < 0
        if verbose:
            print("INSIDE ACCEPTANCE i:", time, dp_over_p, distance, is_inside, "x0,y0:", x0, y0, "x1,y1:", x1, y1)
        return is_inside

    def contour_area(self, t_list, e_list):
        total_area = 0.0
        my_t_list = t_list+[t_list[0]]
        my_e_list = e_list+[e_list[0]]
        for i, t1 in enumerate(my_t_list[1:]):
            t0 = my_t_list[i]
            e0 = my_e_list[i]
            e1 = my_e_list[i+1]
            chord_length = ((e1-e0)**2+(t1-t0)**2)**0.5 # length of line from t0 e0 to t1 e1
            radius_length = self.geometric_radius((t1+t0)/2, (e1+e0)/2) # distance from 0, 0 to the mid point of chord t0,e0 to t1,e1
            area = chord_length/2*radius_length # area is (1/2 * chord_length * radius_length) - area of triangle 0,0 to t1,e1 to t0,e0
            #print(i, area, "*", end=" ")
            total_area += area
        return total_area

    def pmin(self):
        pmin = min([min(contour["e"]) for contour in self.contours])
        return pmin

    def pmax(self):
        pmax = max([max(contour["e"]) for contour in self.contours])
        return pmax

