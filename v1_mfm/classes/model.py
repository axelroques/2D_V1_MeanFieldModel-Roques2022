
from ..utils.connectivity_matrices import generateConnectivityMatrices_TORUSRANDOM
from ..utils.connectivity_matrices import generateConnectivityMatrices_TORUS
from ..utils.connectivity_matrices import generateConnectivityMatrices_SHEET
from ..utils.plot import xz_combined, xz_plot, xz_movie
from ..utils.parsers import getXZVectors

from ..classes.neurons import NeuronPopulation
from ..classes.mean_field import MeanField
from ..classes.network import Network
from ..classes.stimuli import Stimuli

import numpy as np
import os


class Model():
    """
    Model class.
    """

    def __init__(
        self,
        network='SHEET',
        simulation_length=1.0,
        stimulation_type='CENTER',
        BIN=5e-3,
        custom_network_parameters={},
        custom_stimulation_parameters={},
        random_conn_parameters={}
    ) -> None:

        # Type check
        if network not in ['SHEET', 'TORUS', 'TORUSRANDOM']:
            raise RuntimeError(
                "Unknown model type. Expected 'SHEET', 'TORUS' or 'TORUSRANDOM'"
            )
        if (network == 'TORUS') and (random_conn_parameters == None):
            self.random_conn_parameters = {
                'nb_random_conn': 90,
                'weight_rand': 1
            }

        # Input parameters
        dt = 5e-4
        self.simulation_length = simulation_length
        self.t = np.linspace(
            0,
            simulation_length,
            int(simulation_length/dt),
            endpoint=False
        )
        self.network_type = network
        self.BIN = BIN
        self.random_conn_parameters = random_conn_parameters

        # Neuron population & parameters
        self.exc_pop = NeuronPopulation(type='RS')
        self.inh_pop = NeuronPopulation(type='FS')

        # Network & parameters
        self.network = Network(
            network, custom_parameters=custom_network_parameters)
        self.network_parameters = self.network.getParameters()

        # Stimulation parameters
        self.X, self.Z = getXZVectors(self.network_parameters)
        self.stimulation = Stimuli(
            self.t, self.X, self.Z,
            stimulation_type=stimulation_type,
            custom_parameters=custom_stimulation_parameters
        )
        self.stimulation_parameters = self.stimulation.getParameters()

        # Other parameters
        self.ext_drive = 2.

        # Connectivity matrices
        self._getConnectivityMatrices()

        # Mean field formalism
        self.mean_field = MeanField(
            self.exc_pop, self.inh_pop,
            self.network, self.ext_drive
        )

        # Load transfer functions
        self.TF_exc, self.TF_inh = self.mean_field.loadTF()

        # Get afferent stimulation
        self.Fe_aff = self.stimulation.getAffStim()

    def _getConnectivityMatrices(self):
        """
        Retrieve or generate connectivity matrices for the 
        model type specified. 
        """

        absolute_path = os.path.realpath(os.path.dirname(__file__))
        relative_file_path = f'../data/ConnMatrices_{self.network_type}.npy'
        path = os.path.join(absolute_path, relative_file_path)

        # Check if the matrices were previously generated
        try:
            self.M_conn_exc, self.M_conn_inh, \
                self.nb_exc_neighb, self.nb_inh_neighb = \
                np.load(path, allow_pickle=True)
            print('Successfully loaded connectivity matrices.')

        # Otherwise create and load matrices
        except FileNotFoundError:
            self._generateConnectivityMatrices(
                self.network_type,
                self.network_parameters,
                self.random_conn_parameters
            )
            self.M_conn_exc, self.M_conn_inh, \
                self.nb_exc_neighb, self.nb_inh_neighb = \
                np.load(path, allow_pickle=True)
            print('Successfully built connectivity matrices.')

        return

    @staticmethod
    def _generateConnectivityMatrices(type, network_parameters, random_conn_parameters):
        """
        Generate connectivity matrices for the 
        model type specified.
        """

        model_type = {
            'SHEET': generateConnectivityMatrices_SHEET,
            'TORUS': generateConnectivityMatrices_TORUS,
            'TORUSRANDOM': generateConnectivityMatrices_TORUSRANDOM
        }

        try:
            model_type[type](
                network_parameters=network_parameters,
                random_conn_parameters=random_conn_parameters
            )

        except KeyError:
            raise RuntimeError(
                "Unknown model type. Expected 'SHEET', 'TORUS' or 'TORUSRANDOM'"
            )

        return

    def run(self):
        """
        Run simulation.

        Given two afferent rate input excitatory and inhibitory respectively
        this function computes the prediction of a first order rate model
        (e.g. Wilson and Cowan in the 70s, or 1st order of El Boustani and
        Destexhe 2009) by implementing a simple Euler method.
        ----------------------------------------------------------------
        The core of the formalism is the transfer function, see Zerlaut et 
        al. 2015, Kuhn et al. 2004 or Amit & Brunel 1997
        -----------------------------------------------------------------
        nu_0 is the starting value value of the recurrent network activity
        it should be the fixed point of the network dynamics
        -----------------------------------------------------------------
        t is the discretization used to solve the euler method
        BIN is the initial sampling bin that should correspond to the
        markovian time scale where the formalism holds (~5ms)
        """

        # Initialization
        conduction_velocity = self.network.parameters['conduction_velocity']
        dt = (self.t[1] - self.t[0]) / 10
        Fe, Fi, muVn = self.mean_field.getFixedPoint(
            array_shape=np.zeros_like(self.Fe_aff))
        X, Z = self.X, self.Z

        # Time loop
        for i_t in range(len(self.t)-1):

            # Progress bar
            print('\rComputing... [{0:<50s}] {1:5.1f}%'
                  .format('#' * int((i_t+1)/(len(self.t)-1)*50),
                          (i_t+1)/(len(self.t)-1)*100), end="")

            # Loop over every mean field network
            for i_z in range(len(Z)):
                for i_x in range(len(X)):

                    fe = self.ext_drive
                    fi = 0
                    fe_pure_exc = self.Fe_aff[i_t, i_x, i_z]

                    # Excitatory neighbours
                    for i_exc in range(len(self.M_conn_exc[i_z][i_x])):
                        # Delay in propagation due to limited axon conduction
                        if i_t > int(abs(self.M_conn_exc[i_z][i_x][i_exc]['dist'])/conduction_velocity/dt):
                            it_delayed = i_t - \
                                int(abs(self.M_conn_exc[i_z][i_x][i_exc]
                                        ['dist'])/conduction_velocity/dt)
                        else:
                            it_delayed = 0
                        # Using the connectivity matrix to find the weight and the position of this neighbour
                        fe += self.M_conn_exc[i_z][i_x][i_exc]['weight'] * \
                            Fe[it_delayed, self.M_conn_exc[i_z][i_x][i_exc]
                                ['pos_x'], self.M_conn_exc[i_z][i_x][i_exc]['pos_z']]

                    # Inhibitory neighbours
                    for i_inh in range(len(self.M_conn_inh[i_z][i_x])):
                        # Delay in propagation due to limited axon conduction
                        if i_t > int(abs(self.M_conn_inh[i_z][i_x][i_inh]['dist'])/conduction_velocity/dt):
                            it_delayed = i_t - \
                                int(abs(self.M_conn_inh[i_z][i_x][i_inh]
                                        ['dist'])/conduction_velocity/dt)
                        else:
                            it_delayed = 0
                        # Using the connectivity matrix to find the weight and the position of this neighbour
                        fi += self.M_conn_inh[i_z][i_x][i_inh]['weight'] * \
                            Fi[it_delayed, self.M_conn_inh[i_z][i_x][i_inh]
                                ['pos_x'], self.M_conn_inh[i_z][i_x][i_inh]['pos_z']]

                    # Model output
                    muVn[i_t+1, i_x, i_z], _, _, _ = self.mean_field.get_fluct_regime_vars(
                        fe, fi
                    )
                    Fe[i_t+1, i_x, i_z] = Fe[i_t, i_x, i_z] + dt/self.BIN * \
                        (self.TF_exc(fe+fe_pure_exc, fi) - Fe[i_t, i_x, i_z])
                    Fi[i_t+1, i_x, i_z] = Fi[i_t, i_x, i_z] + \
                        dt/self.BIN*(self.TF_inh(fe, fi) - Fi[i_t, i_x, i_z])

        print('\nSimulation completed.')

        self.simulation_results = {
            'network_parameters': self.network_parameters,
            't': self.t,
            'X': self.X,
            'Z': self.Z,
            'Fe_aff': self.Fe_aff,
            'Fe': Fe,
            'Fi': Fi,
            'deltaVn': np.abs((muVn-self.mean_field.muV0)/self.mean_field.muV0)
        }

        return self.simulation_results

    def plot(self, x=None, z=None):
        """
        2D plots of Fe_aff, Fe, Fi and muVn against time, for a single population at (x,z).
        """

        # Defaults to the population at the center of the network
        if not x:
            x = int(self.network_parameters['X_discretization']//2)
        if not z:
            z = int(self.network_parameters['Z_discretization']//2)

        Fe = self.simulation_results['Fe']
        Fi = self.simulation_results['Fi']
        muVn = self.simulation_results['deltaVn']

        xz_plot(self.Fe_aff, Fe, Fi, muVn, len(self.t), x, z)
        xz_combined(self.Fe_aff, Fe, Fi, muVn, len(self.t), x, z)

        return

    def movie(self, fps=10, path='results/movies/', title='output'):
        """
        Movie of contour plots of Fe_aff, Fe, Fi and muVn in the (x,z) plane. 
        """

        Fe = self.simulation_results['Fe']
        Fi = self.simulation_results['Fi']
        muVn = self.simulation_results['deltaVn']

        xz_movie(
            self.Fe_aff, Fe, Fi, muVn,
            self.X, self.Z, len(self.t),
            fps=fps, path=path, title=title
        )

        return

    def save(self, path):
        """
        Save simulation results.
        """

        from datetime import datetime
        import pickle

        date = datetime.now()
        filename = f'{date.year}_{date.month}_{date.day}-{date.hour}h{date.minute}min.npy'

        # absolute_path = os.path.realpath(os.path.dirname(__file__))
        # relative_file_path = f'../results/{filename}'
        # path = os.path.join(absolute_path, relative_file_path)

        with open(f'{path}/{filename}', 'wb') as f:
            pickle.dump(self.simulation_results, f)

        print(f'Simulation saved in {path}{filename}.')

        return

    def load(self, file):
        """
        Load simulation results.
        """

        import pickle

        with open(file, 'rb') as f:
            self.simulation_results = pickle.load(f)

        return
