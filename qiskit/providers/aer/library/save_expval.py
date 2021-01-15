# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Simulator instruction to save exact operator expectation value.
"""

from qiskit.quantum_info import Pauli, SparsePauliOp, Operator
from qiskit.circuit import QuantumCircuit
from qiskit.extensions.exceptions import ExtensionError
from .save_data import SaveAverageData


class SaveExpval(SaveAverageData):
    """Save expectation value of an operator."""
    def __init__(self,
                 key,
                 operator,
                 conditional=False,
                 pershot=False,
                 unnormalized=False):
        """Create new a instruction to save simulator expectation value.

        Args:
            data (str): the type of data being saved.
            key (str): the key for retrieving saved data from results.
            operator (Pauli or SparsePauliOp or Operator): the operator to save the
                                                           expectation value of.
            pershot (bool): if True save a list of expectation values for each shot
                            of the simulation rather than the average over
                            all shots [Default: False].
            conditional (bool): if True save the average or pershot data
                                conditional on the current classical register
                                values [Default: False].
            unnormalized (bool): If True return save the unnormalized accumulated
                                 or conditional accumulated expectation value
                                 over all shot [Default: False].

        Raises:
            ExtensionError: if the input operator is not valid.

        .. note ::

            In cetain cases the list returned by ``pershot=True`` may only
            contain a single value, rather than the number of shots. This
            happens when a run circuit supports measurement sampling because
            it is either

            1. An ideal simulation with all measurements at the end.

            2. A noisy simulation using the density matrix method with all
            measurements at the end.

            In both these cases only a single shot is actually simulated and
            measurement samples for all shots are calculated from the final
            state.
        """
        # Convert to sparse pauli op
        if isinstance(operator, Pauli):
            operator = SparsePauliOp(operator)
        elif not isinstance(operator, SparsePauliOp):
            operator = SparsePauliOp.from_operator(Operator(operator))
        if not isinstance(operator, SparsePauliOp):
            raise ExtensionError("Invalid input operator")
        # Convert SparsePauliOp to old Pauli-snapshot style params
        num_qubits = operator.num_qubits
        params = [[coeff, label] for label, coeff in operator.label_iter()]
        super().__init__("expval",
                         key,
                         num_qubits,
                         conditional=conditional,
                         pershot=pershot,
                         unnormalized=unnormalized,
                         params=params)


def save_expval(self,
                key,
                operator,
                qubits,
                conditional=False,
                pershot=False,
                unnormalized=False):
    """Save the measurement probabilities of the current simulator state.

    Args:
        key (str): the key for retrieving saved data from results.
        operator (Pauli or SparsePauliOp or Operator): the operator to save the
                                                       expectation value of.
        pershot (bool): if True save a list of expectation values for each shot
                        of the simulation rather than the average over
                        all shots [Default: False].
        conditional (bool): if True save the average or pershot data
                            conditional on the current classical register
                            values [Default: False].
        unnormalized (bool): If True return save the unnormalized accumulated
                             or conditional accumulated expectation value
                             over all shot [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    .. note ::

        In cetain cases the list returned by ``pershot=True`` may only
        contain a single value, rather than the number of shots. This
        happens when a run circuit supports measurement sampling because
        it is either

        1. An ideal simulation with all measurements at the end.

        2. A noisy simulation using the density matrix method with all
           measurements at the end.

        In both these cases only a single shot is actually simulated and
        measurement samples for all shots are calculated from the final
        state.
    """
    instr = SaveExpval(key,
                       operator,
                       conditional=conditional,
                       pershot=pershot,
                       unnormalized=unnormalized)
    return self.append(instr, qubits)


QuantumCircuit.save_expval = save_expval
