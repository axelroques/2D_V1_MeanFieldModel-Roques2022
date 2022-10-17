
from scipy.integrate import odeint
from scipy.special import erfc
import numpy as np


class MeanField():
    """
    Mean-field formalism.
    """

    def __init__(self, exc_pop, inh_pop, network, ext_drive) -> None:

        # Neuron populations
        self.exc_pop = exc_pop
        self.inh_pop = inh_pop

        # Network
        self.network = network

        # External drive
        self.ext_drive = ext_drive

        # Find fixed point
        self.fe0, self.fi0 = self._find_fixed_point_first_order()

        # Find initial membrane potential
        self.muV0, _, _, _ = self.get_fluct_regime_vars(
            self.fe0+self.ext_drive, self.fi0,
        )

    def _find_fixed_point_first_order(self):
        """
        Used to initialize the model with the external drive until a stationary state
        """

        # Load transfer functions
        TF1, TF2 = self.loadTF()

        t = np.arange(2000)*1e-4  # Time vector

        # Solve equation (first order)
        def dX_dt_scalar(X, t=0):
            return self._build_up_differential_operator_first_order(TF1, TF2)(X, exc_aff=self.ext_drive)

        X0 = [1, 10]  # need inhibition stronger than excitation
        X = odeint(dX_dt_scalar, X0, t)

        return X[-1][0], X[-1][1]

    @staticmethod
    def _build_up_differential_operator_first_order(TF1, TF2, T=5e-3):
        """
        Solves mean field equation for the first order system
        Only used for the initialization of the model with the find_fixed_point_first_order function
        """

        def A0(V, exc_aff=0):
            return 1./T*(TF1(V[0]+exc_aff, V[1])-V[0])

        def A1(V, exc_aff=0):
            return 1./T*(TF2(V[0]+exc_aff, V[1])-V[1])

        def Diff_OP(V, exc_aff=0):
            return np.array([A0(V, exc_aff=exc_aff),
                             A1(V, exc_aff=exc_aff)])

        return Diff_OP

    def loadTF(self):
        """
        Returns transfer functions for both RS and FS cell types
        """

        # Excitatory population

        def TF_exc(fe, fi):
            return self._TF(fe, fi, type='RS')

        def TF_inh(fe, fi):
            return self._TF(fe, fi, type='FS')

        return TF_exc, TF_inh

    def _TF(self, fe, fi, type):
        """
        Transfer function computation.

        Uses the total (excitatory and inhibitory) input to estimate
        the population's firing rate.

        The type parameter dictates whether to use the 'P' parameters
        from the RS or FS cell types.
        """

        muV, sV, muGn, TvN = self.get_fluct_regime_vars(fe, fi)
        Vthre = self._threshold_func(muV, sV, muGn, TvN, type)
        Fout_th = self._erfc_func(
            muV, sV, TvN, Vthre,
            # Gl is equal for exc or inh pop
            self.inh_pop.parameters['Gl'],
            # Cm is equal for exc or inh pop
            self.inh_pop.parameters['Cm']
        )

        return Fout_th

    def get_fluct_regime_vars(self, Fe, Fi):
        """
        Computes values needed for the transfer function.
        """

        Ntot = self.network.parameters['Ntot']
        gei = self.network.parameters['gei']
        # pconn is equal for exc or inh pop
        pconn = self.exc_pop.parameters['p_conn']
        # Gl is equal for exc or inh pop
        Gl = self.inh_pop.parameters['Gl']
        # El is equal for exc or inh pop
        El = self.inh_pop.parameters['El']
        # Cm is equal for exc or inh pop
        Cm = self.inh_pop.parameters['Cm']
        Qe, Qi = self.exc_pop.synapses_parameters['Q'], self.inh_pop.synapses_parameters['Q']
        Te, Ti = self.exc_pop.synapses_parameters['T'], self.inh_pop.synapses_parameters['T']
        Ee, Ei = self.exc_pop.synapses_parameters['E'], self.inh_pop.synapses_parameters['E']

        # here TOTAL (sum over synapses) excitatory and inhibitory input
        fe = Fe*(1.-gei)*pconn*Ntot
        fi = Fi*gei*pconn*Ntot

        muGe, muGi = Qe*Te*fe, Qi*Ti*fi
        muG = Gl+muGe+muGi
        muV = (muGe*Ee+muGi*Ei+Gl*El)/muG
        muGn, Tm = muG/Gl, Cm/muG

        Ue, Ui = Qe/muG*(Ee-muV), Qi/muG*(Ei-muV)

        sV = np.sqrt(
            fe*(Ue*Te)**2/2./(Te+Tm) +
            fi*(Qi*Ui)**2/2./(Ti+Tm)
        )

        fe, fi = fe+1e-9, fi+1e-9  # just to insure a non zero division,
        Tv = (fe*(Ue*Te)**2 + fi*(Qi*Ui)**2) / \
            (fe*(Ue*Te)**2/(Te+Tm) + fi*(Qi*Ui)**2/(Ti+Tm))
        TvN = Tv*Gl/Cm

        return muV, sV+1e-12, muGn, TvN

    def _threshold_func(self, muV, sV, muGn, TvN, type):
        """
        Threshold function.

        Here are the previous mysterious comments from Yann Zerlaut, if they can be of any help:
            setting by default to True the square
            because when use by external modules, coeff[5:]=np.zeros(3)
            in the case of a linear threshold
        2 years later, I still haven't figured out the meaning of those words :)
        """

        # Get the correct P parameters depending on the cell type considered.
        if type == 'RS':
            P = self.exc_pop.parameters['P']
        elif type == 'FS':
            P = self.inh_pop.parameters['P']
        else:
            raise RuntimeError('How did you get here?')

        P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10 = P[0], P[1], P[2], \
            P[3], P[4], P[5], P[6], P[7], P[8], P[9], P[10]
        muV0, DmuV0 = -60e-3, 10e-3
        sV0, DsV0 = 4e-3, 6e-3
        TvN0, DTvN0 = 0.5, 1.

        return P0 +\
            P1*(muV-muV0)/DmuV0 +\
            P2*(sV-sV0)/DsV0 + \
            P3*(TvN-TvN0)/DTvN0 +\
            P4*np.log(muGn) +\
            P5*((muV-muV0)/DmuV0)**2 +\
            P6*((sV-sV0)/DsV0)**2 +\
            P7*((TvN-TvN0)/DTvN0)**2 +\
            P8*(muV-muV0)/DmuV0*(sV-sV0)/DsV0 +\
            P9*(muV-muV0)/DmuV0*(TvN-TvN0)/DTvN0 +\
            P10*(sV-sV0)/DsV0*(TvN-TvN0)/DTvN0

    @staticmethod
    def _erfc_func(muV, sV, TvN, Vthre, Gl, Cm):
        """
        Error function.
        """
        return .5/TvN*Gl/Cm*(erfc((Vthre-muV)/np.sqrt(2)/sV))

    def getFixedPoint(self, array_shape):
        """
        Return fixed point parameters.
        """
        return array_shape+self.fe0, array_shape+self.fi0, array_shape+self.muV0
