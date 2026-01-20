from collections import defaultdict

from flc.shared.pcaps import Layer4Proto

SYN = 0x02
ACK = 0x10
SYN_ACK = 0x12
FIN_ACK = 0x11
RST = 0x04
RST_ACK = 0x14
FIN = 0x01


class TcpHandshakeStateMachine:
    _current_state = None
    _handshake_started = False
    _handshake_finished = False

    def update(self, packet_tcp_flags):
        # TODO improve by taking into account the direction of the packet
        if packet_tcp_flags == SYN:
            self._handshake_started = True
            self._current_state = SYN
        elif packet_tcp_flags == SYN_ACK and self._current_state in [SYN, SYN_ACK]:
            self._current_state = SYN_ACK
        elif packet_tcp_flags == ACK and self._current_state in [SYN_ACK, ACK]:
            self._handshake_finished = True
            self._current_state = ACK
        else:
            self._current_state = None

    @property
    def started(self):
        return self._handshake_started

    @property
    def finished(self):
        return self._handshake_finished


class TcpTerminationStateMachine:
    _current_state = None
    _termination_started = False
    _termination_finished = False

    def update(self, packet_tcp_flags):
        # TODO improve by taking into account the direction of the packet
        if packet_tcp_flags == FIN_ACK:
            self._termination_started = True
            self._current_state = FIN_ACK
        elif packet_tcp_flags == ACK and self._current_state in [FIN_ACK, ACK]:
            self._current_state = ACK
            self._termination_finished = True
        else:
            self._current_state = None

    @property
    def started(self):
        return self._termination_started

    @property
    def finished(self):
        return self._termination_finished


class TcpState:
    def __init__(self):
        self._handshake_state = TcpHandshakeStateMachine()
        self._termination_state = TcpTerminationStateMachine()
        self._flags = defaultdict(int)
        self._payload_bytes = 0

    def update(self, packet):
        if packet.transport_layer != Layer4Proto.TCP:
            raise ValueError("Only TCP packets are supported")

        # measure how much data was exchanged
        self._payload_bytes += int(packet.payload_size)

        if packet.tcp_flags == SYN_ACK:
            self._flags["SYN/ACK"] += 1
        elif packet.tcp_flags == SYN:
            self._flags["SYN"] += 1
        elif packet.tcp_flags == ACK:
            self._flags["ACK"] += 1
        elif packet.tcp_flags == RST:
            self._flags["RST"] += 1
        elif packet.tcp_flags == RST_ACK:
            self._flags["RST/ACK"] += 1
        elif packet.tcp_flags == FIN_ACK:
            self._flags["FIN/ACK"] += 1
        else:
            self._flags["OTHER"] += 1

        self._handshake_state.update(packet.tcp_flags)
        self._termination_state.update(packet.tcp_flags)

    @property
    def flags(self):
        return dict(self._flags)

    @property
    def handshake_finished(self):
        return self._handshake_state.finished

    @property
    def termination_finished(self):
        return self._termination_state.finished

    @property
    def payload_bytes(self):
        return self._payload_bytes
