import numpy
from optimisation_tools.utils.decoupled_transfer_matrix import DecoupledTransferMatrix

class TwoDTransferMatrix(DecoupledTransferMatrix):
    """
    Specialisation of the generalised decoupled transfer matrix that assumes the
    input matrix is decoupled already
    """
    def _check_evectors(self):
        """
        Check the evectors and evalues are ordered okay
        """
        pass

    def _get_decoupling(self):
        """
        Calculate the transformation that decouples the transfer matrix and some
        other associated set-up.
        """
        self.t = self.m
        self.t_evector, self.t_evalue = self.m_evector, self.m_evalue
        self.v_t = numpy.zeros([self.dim*2, self.dim*2]) # real
        self.r = numpy.identity(2*self.dim)
        self.r_inv = numpy.identity(2*self.dim)
        for i in range(self.dim):
            j=2*i
            v_2d = self.get_v2d(self.m[j:2+j, j:2+j])
            self.v_t[j:2+j, j:2+j] = v_2d
        print("VT", self.v_t, numpy.linalg.det(self.v_t))
        try:
            self.chol = numpy.linalg.cholesky(self.v_t)
            self.chol_inv = numpy.linalg.inv(self.chol)
        except numpy.linalg.LinAlgError:
            # matrix is not positive definite. Never mind, let's plug on
            print("ERRORROROEEORO")
            self.chol = None
            self.chol_inv = None

    def get_v2d(self, m2d):
        cosmu = (m2d[0,0]+m2d[1,1])/2
        if abs(cosmu) > 1:
            raise RuntimeError("m2d was unstable with cosmu: "+str(cosmu))
        n2d = m2d - numpy.array([[cosmu, 0.0],[0.0,cosmu]])
        sinmu = numpy.linalg.det(n2d)**0.5
        n2d /= sinmu
        v2d = numpy.array([[n2d[0,1], -n2d[0,0]], [-n2d[0,0], -n2d[1, 0]]])
        return v2d
