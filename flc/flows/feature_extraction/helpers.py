from flc.shared.pcaps import Layer4Proto


def is_tcp_control_packet(packet):
    if packet is None:
        return False

    # If no TCP flags, it's not a TCP packet at all
    if packet.transport_layer != Layer4Proto.TCP or packet.tcp_flags is None:
        return False

    # TCP control bits:
    # FIN (1), SYN (2), RST (4), PSH (8), ACK (16), URG (32)

    # Check for SYN, FIN, RST, or ACK flags
    is_syn = packet.tcp_flags & 2 != 0
    is_fin = packet.tcp_flags & 1 != 0
    is_rst = packet.tcp_flags & 4 != 0
    is_ack = packet.tcp_flags & 16 != 0
    is_psh = packet.tcp_flags & 8 != 0

    # Control packets typically have no payload
    has_payload = packet.payload_size is not None and packet.payload_size > 0

    # A packet is considered a control packet if:
    # 1. It has SYN, FIN, or RST flags set AND no payload, OR
    # 2. It has only ACK flag set (no PSH) AND no payload
    return ((is_syn or is_fin or is_rst) and not has_payload) or (is_ack and not is_psh and not has_payload)


def is_tcp(transport_layers):
    return transport_layers.keys() == {Layer4Proto.TCP}


def is_udp(transport_layers):
    return transport_layers.keys() == {Layer4Proto.UDP}


def is_sctp(transport_layers):
    return transport_layers.keys() == {Layer4Proto.SCTP}
