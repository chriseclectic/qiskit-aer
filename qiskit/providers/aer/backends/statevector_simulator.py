# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Qiskit Aer statevector simulator backend.
"""

import logging
from qiskit.util import local_hardware_info
from qiskit.providers.models import QasmBackendConfiguration
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.providers.aer.backends.backend_utils import (cpp_execute,
                                                         available_methods,
                                                         MAX_QUBITS_STATEVECTOR
                                                         )
from qiskit.providers.aer.aererror import AerError
from qiskit.providers.aer.version import __version__
# pylint: disable=import-error, no-name-in-module
from qiskit.providers.aer.backends.controller_wrappers import statevector_controller_execute

# Logger
logger = logging.getLogger(__name__)

BASIS_GATES = [
    'u1', 'u2', 'u3', 'cx', 'cz', 'id', 'x', 'y', 'z', 'h', 's', 'sdg', 't',
    'tdg', 'swap', 'ccx', 'unitary', 'initialize', 'cu1', 'cu2', 'cu3',
    'cswap', 'mcx', 'mcy', 'mcz', 'mcu1', 'mcu2', 'mcu3', 'mcswap',
    'multiplexer'
]

DEFAULT_CONFIGURATION = {
    'backend_name': 'statevector_simulator',
    'backend_version': __version__,
    'n_qubits': MAX_QUBITS_STATEVECTOR,
    'url': 'https://github.com/Qiskit/qiskit-aer',
    'simulator': True,
    'local': True,
    'conditional': True,
    'open_pulse': False,
    'memory': True,
    'max_shots': int(1e6),  # Note that this backend will only ever
    # perform a single shot. This value is just
    # so that the default shot value for execute
    # will not raise an error when trying to run
    # a simulation
    'description': 'A C++ statevector simulator for QASM Qobj files',
    'coupling_map': None,
    'basis_gates': BASIS_GATES,
    'gates': []
}


class StatevectorSimulator(AerBackend):
    """Ideal quantum circuit statevector simulator

    **Backend options**

    The following backend options may be used with in the
    ``backend_options`` kwarg for :meth:`StatevectorSimulator.run` or
    ``qiskit.execute``.

    * ``"zero_threshold"`` (double): Sets the threshold for truncating
      small values to zero in the result data (Default: 1e-10).

    * ``"validation_threshold"`` (double): Sets the threshold for checking
      if the initial statevector is valid (Default: 1e-8).

    * ``"max_parallel_threads"`` (int): Sets the maximum number of CPU
      cores used by OpenMP for parallelization. If set to 0 the
      maximum will be set to the number of CPU cores (Default: 0).

    * ``"max_parallel_experiments"`` (int): Sets the maximum number of
      qobj experiments that may be executed in parallel up to the
      max_parallel_threads value. If set to 1 parallel circuit
      execution will be disabled. If set to 0 the maximum will be
      automatically set to max_parallel_threads (Default: 1).

    * ``"max_memory_mb"`` (int): Sets the maximum size of memory
      to store a state vector. If a state vector needs more, an error
      is thrown. In general, a state vector of n-qubits uses 2^n complex
      values (16 Bytes). If set to 0, the maximum will be automatically
      set to half the system memory size (Default: 0).

    * ``"statevector_parallel_threshold"`` (int): Sets the threshold that
      "n_qubits" must be greater than to enable OpenMP
      parallelization for matrix multiplication during execution of
      an experiment. If parallel circuit or shot execution is enabled
      this will only use unallocated CPU cores up to
      max_parallel_threads. Note that setting this too low can reduce
      performance (Default: 14).
    """

    # Cache available methods
    _AVAILABLE_METHODS = None

    def __init__(self,
                 configuration=None,
                 properties=None,
                 provider=None,
                 **backend_options):

        self._controller = statevector_controller_execute()

        if StatevectorSimulator._AVAILABLE_METHODS is None:
            StatevectorSimulator._AVAILABLE_METHODS = available_methods(
                self._controller, [
                    'automatic', 'statevector', 'statevector_gpu',
                    'statevector_thrust'
                ])
        if configuration is None:
            configuration = QasmBackendConfiguration.from_dict(
                DEFAULT_CONFIGURATION)
        super().__init__(
            configuration,
            properties=properties,
            available_methods=StatevectorSimulator._AVAILABLE_METHODS,
            provider=provider,
            backend_options=backend_options)

    def _execute(self, qobj, run_config):
        """Execute a qobj on the backend.

        Args:
            qobj (QasmQobj): simulator input.
            run_config (dict): run config for overriding Qobj config.

        Returns:
            dict: return a dictionary of results.
        """
        controller_input = qobj.to_dict()
        for key, val in run_config.items():
            if hasattr(val, 'to_dict'):
                controller_input['config'][key] = val.to_dict()
            else:
                controller_input['config'][key] = val
        # Execute on controller
        return cpp_execute(self._controller, controller_input)

    def _validate(self, qobj, options):
        """Semantic validations of the qobj which cannot be done via schemas.
        Some of these may later move to backend schemas.

        1. Set shots=1.
        2. Check number of qubits will fit in local memory.
        """
        name = self.name()
        if options and 'noise_model' in options:
            raise AerError("{} does not support noise.".format(name))

        n_qubits = qobj.config.n_qubits
        max_qubits = self.configuration().n_qubits
        if n_qubits > max_qubits:
            raise AerError(
                'Number of qubits ({}) is greater than max ({}) for "{}" with {} GB system memory.'
                .format(n_qubits, max_qubits, name,
                        int(local_hardware_info()['memory'])))

        if qobj.config.shots != 1:
            logger.info('"%s" only supports 1 shot. Setting shots=1.', name)
            qobj.config.shots = 1

        for experiment in qobj.experiments:
            exp_name = experiment.header.name
            if getattr(experiment.config, 'shots', 1) != 1:
                logger.info(
                    '"%s" only supports 1 shot. '
                    'Setting shots=1 for circuit "%s".', name, exp_name)
                experiment.config.shots = 1
