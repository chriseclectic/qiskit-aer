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
# that they have been altered from the originals

"""
Simulator command to snapshot internal simulator representation.
"""

from qiskit import QuantumCircuit
from qiskit.providers.aer.extensions import Snapshot


class SnapshotProbabilities(Snapshot):
    """Snapshot instruction for all methods of Qasm simulator."""

    def __init__(self, label, num_qubits, single_shot=False, conditional=False):
        """Create a probability snapshot instruction.

        Args:
            label (str): the snapshot label.
            num_qubits (int): the number of qubits to snapshot.
            single_shot (bool): return list for each shot rather than average [Default: False]
            conditional (bool): If True return conditional snapshot [Default: False].

        Raises:
            ExtensionError: if snapshot is invalid.
        """
        snapshot_type = 'probabilities'
        if single_shot:
            snapshot_type += '_single_shot'
        elif conditional:
            snapshot_type += '_conditional'
        super().__init__(label, snapshot_type=snapshot_type,
                         num_qubits=num_qubits)


def snapshot_probabilities(self, label, qubits, single_shot=False, conditional=False):
    """Take a probability snapshot of the simulator state.

    Args:
        label (str): a snapshot label to report the result
        qubits (list): the qubits to snapshot.
        single_shot (bool): return list for each shot rather than average [Default: False]
        conditional (bool): If True return conditional snapshot [Default: False].

    Returns:
        QuantumCircuit: with attached instruction.

    Raises:
        ExtensionError: if snapshot is invalid.
    """
    snapshot_register = Snapshot.define_snapshot_register(self, label, qubits)

    return self.append(
        SnapshotProbabilities(label,
                              num_qubits=len(snapshot_register),
                              single_shot=single_shot,
                              conditional=conditional),
        snapshot_register)


QuantumCircuit.snapshot_probabilities = snapshot_probabilities
