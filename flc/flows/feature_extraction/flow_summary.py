import os
from collections import defaultdict
from enum import IntEnum
from typing import Optional

import pickle
import numpy as np

from flc.shared.pcaps.tshark import Packet, Layer4Proto, parse_tshark_csv, parse_pcap
from flc.flows.feature_extraction.flow_tcp_state import SYN, SYN_ACK, TcpState
from flc.flows.feature_extraction.helpers import is_tcp_control_packet


class Direction(IntEnum):
    CLIENT_TO_SERVER = 0
    SERVER_TO_CLIENT = 1

    @staticmethod
    def to_idx(direction: str) -> int:
        return Direction[direction].value


class FlowId:
    address_src = None
    address_dst = None
    port_src = None
    port_dst = None

    def __init__(self, packet: Packet):
        if packet.ip_src is not None:
            self.address_src = packet.ip_src
            self.address_dst = packet.ip_dst
        else:
            self.address_src = packet.eth_src
            self.address_dst = packet.eth_dst

        self.port_src = packet.port_src
        self.port_dst = packet.port_dst

    def __eq__(self, __value):
        return (
            self.address_src == __value.address_src
            and self.address_dst == __value.address_dst
            and self.port_src == __value.port_src
            and self.port_dst == __value.port_dst
        )

    def __ne__(self, __value):
        return (
            self.address_src != __value.address_src
            or self.address_dst != __value.address_dst
            or self.port_src != __value.port_src
            or self.port_dst != __value.port_dst
        )


def direction_detection(first_packet: Packet) -> Direction:
    # start by assuming this is the actual first packet of the flow
    direction = Direction.CLIENT_TO_SERVER

    # if the ports are different we assume the lower port is the server
    if first_packet.port_src != first_packet.port_dst and first_packet.port_src < first_packet.port_dst:
        direction = Direction.SERVER_TO_CLIENT

    # if it's a TCP flow we check whether we are observing a handshake
    if first_packet.transport_layer == Layer4Proto.TCP:
        if first_packet.tcp_flags == SYN:
            # start of the handshake
            direction = Direction.CLIENT_TO_SERVER
        elif first_packet.tcp_flags == SYN_ACK:
            # response to the initial SYN
            direction = Direction.SERVER_TO_CLIENT

    return direction


class PacketCounter:
    def __init__(self):
        self.packets = 0
        self.bytes = 0
        self.bytes_list = []

    def update(self, packet: Packet):
        self.packets += 1
        self.bytes += packet.length
        self.bytes_list.append(packet.length)


class FlowState:
    def __init__(self, first_packet: Packet):
        self.direction = direction_detection(first_packet)
        self.flow_id = FlowId(first_packet)
        self.packet_counter_norm = PacketCounter()
        self.packet_counter_back = PacketCounter()
        self.inter_times_norm = []
        self.inter_times_back = []


class FlowSummary:
    def __init__(self, transport_layers, protocols, tcp_state, details, features, malformed_packets, vlan_etypes):
        self.transport_layers = transport_layers
        self.protocols = protocols
        self.tcp_state = tcp_state
        self.details = details
        self.features = features
        self.malformed_packets = malformed_packets
        self.vlan_etypes = vlan_etypes


def extract_features(flow_pcap: str, ignore_tcp_control_packets: bool = True) -> Optional[FlowSummary]:
    """
    Extracts protocols and TCP flags from a PCAP by running tshark on the input. We only care about the highest
    protocol in the stack, e.g., Ethernet > IPv4 > TCP > HTTP will count as "HTTP". For TCP flags we are only
    interested in specific combinations currently (values are exclusive).
    :param flow_pcap: file path to the PCAP file containing the flow to be parsed
    :param ignore_tcp_control_packets: if set to True packets like TCP ACK, SYN, RST and FIN will be ignored
    :return: protocols and TCP flags as dicts or (None, None) if there was an error with tshark
    """

    protocols = defaultdict(int)
    tcp_state = None
    transport_layers = defaultdict(int)
    malformed_packets = 0
    vlan_etypes = defaultdict(int)

    csv_file = flow_pcap.replace(".pcap", ".csv")
    if os.path.exists(csv_file):
        packets = parse_tshark_csv(csv_file)
    else:
        packets = parse_pcap(flow_pcap)

    flow_state = FlowState(packets[0])
    start_ts = packets[0].sniff_time
    end_ts = start_ts
    last_ts = start_ts

    for i, packet in enumerate(packets):
        protocols[packet.highest_layer] += 1

        for layer in packet.layers:
            if layer == "HTTP":
                protocols["HTTP"] += 1
                break
            elif layer == "FTP":
                protocols["FTP"] += 1
                break
            elif layer == "SMB":
                protocols["SMB"] += 1
                break
            elif layer == "ICMP":
                protocols["ICMP"] += 1
                break
            elif layer == "TLS":
                protocols["TLS"] += 1
                break
            elif layer == "XMPP":
                protocols["XMPP"] += 1
                break
            elif layer == "DB-LSP-DISC":
                protocols["DB-LSP-DISC"] += 1
                break
            elif layer == "ICMP":
                protocols["ICMP"] += 1
                break

        if packet.malformed:
            malformed_packets += 1

        if packet.vlan_etype is not None:
            vlan_etypes[packet.vlan_etype] += 1

        if packet.transport_layer is not None:
            transport_layers[packet.transport_layer] += 1
            if packet.transport_layer == Layer4Proto.TCP:
                if tcp_state is None:
                    tcp_state = TcpState()
                tcp_state.update(packet)
        else:
            transport_layers["NONE"] += 1

        end_ts = packet.sniff_time

        # ignore TCP control packets for feature calculation if config option is enabled
        if ignore_tcp_control_packets and is_tcp_control_packet(packet):
            continue

        # check if this packet follows the initial flow direction (norm=True)
        norm_direction = flow_state.flow_id == FlowId(packet)

        # depending on the norm direction we count the packet information as sent or recv
        if norm_direction:
            flow_state.packet_counter_norm.update(packet)
            if i != 0:
                flow_state.inter_times_norm.append((packet.sniff_time - last_ts).total_seconds())
        else:
            flow_state.packet_counter_back.update(packet)
            if i != 0:
                flow_state.inter_times_back.append((packet.sniff_time - last_ts).total_seconds())

        last_ts = packet.sniff_time

    duration = end_ts - start_ts
    if duration.total_seconds() > 0:
        packets_per_second_send = flow_state.packet_counter_norm.packets / duration.total_seconds()
        packets_per_second_recv = flow_state.packet_counter_back.packets / duration.total_seconds()
        packets_per_second_total = (
            flow_state.packet_counter_norm.packets + flow_state.packet_counter_back.packets / duration.total_seconds()
        )
    else:
        packets_per_second_send = 1 if flow_state.packet_counter_norm.packets > 0 else 0
        packets_per_second_recv = 1 if flow_state.packet_counter_back.packets > 0 else 0
        packets_per_second_total = 1

    # convert to numpy arrays to allow for simple computation of min/max/avg/mean/std/variance
    bytes_send = np.array(flow_state.packet_counter_norm.bytes_list)
    bytes_recv = np.array(flow_state.packet_counter_back.bytes_list)
    bytes_total = np.array(flow_state.packet_counter_norm.bytes_list + flow_state.packet_counter_back.bytes_list)
    inter_times_send = np.array(flow_state.inter_times_norm)
    inter_times_recv = np.array(flow_state.inter_times_back)
    inter_times_total = np.array(flow_state.inter_times_norm + flow_state.inter_times_back)

    return FlowSummary(
        transport_layers=transport_layers,
        protocols=protocols,
        tcp_state=tcp_state,
        details={
            "ip_src": packets[0].ip_src,
            "ip_dst": packets[0].ip_dst,
            "eth_src": packets[0].eth_src,
            "eth_dst": packets[0].eth_dst,
        },
        features={
            "packets.send": flow_state.packet_counter_norm.packets,
            "packets.recv": flow_state.packet_counter_back.packets,
            "packets.total": flow_state.packet_counter_norm.packets + flow_state.packet_counter_back.packets,
            #
            "bytes.send": flow_state.packet_counter_norm.bytes,
            "bytes.send.min": np.min(bytes_send) if len(bytes_send) > 0 else 0,
            "bytes.send.max": np.max(bytes_send) if len(bytes_send) > 0 else 0,
            "bytes.send.mean": np.mean(bytes_send) if len(bytes_send) > 0 else 0,
            "bytes.send.std": np.std(bytes_send) if len(bytes_send) > 0 else 0,
            "bytes.send.var": np.var(bytes_send) if len(bytes_send) > 0 else 0,
            #
            "bytes.recv": flow_state.packet_counter_back.bytes,
            "bytes.recv.min": np.min(bytes_recv) if len(bytes_recv) > 0 else 0,
            "bytes.recv.max": np.max(bytes_recv) if len(bytes_recv) > 0 else 0,
            "bytes.recv.mean": np.mean(bytes_recv) if len(bytes_recv) > 0 else 0,
            "bytes.recv.std": np.std(bytes_recv) if len(bytes_recv) > 0 else 0,
            "bytes.recv.var": np.var(bytes_recv) if len(bytes_recv) > 0 else 0,
            #
            "bytes.total": flow_state.packet_counter_norm.bytes + flow_state.packet_counter_back.bytes,
            "bytes.total.min": np.min(bytes_total) if len(bytes_total) > 0 else 0,
            "bytes.total.max": np.max(bytes_total) if len(bytes_total) > 0 else 0,
            "bytes.total.mean": np.mean(bytes_total) if len(bytes_total) > 0 else 0,
            "bytes.total.std": np.std(bytes_total) if len(bytes_total) > 0 else 0,
            "bytes.total.var": np.var(bytes_total) if len(bytes_total) > 0 else 0,
            #
            "packets-per-second.send": packets_per_second_send,
            "packets-per-second.recv": packets_per_second_recv,
            "packets-per-second.total": packets_per_second_total,
            #
            "protocols": sorted(list(protocols.keys())),
            "direction": (
                "CLIENT_TO_SERVER" if flow_state.direction is Direction.CLIENT_TO_SERVER else "SERVER_TO_CLIENT"
            ),
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "duration": duration.total_seconds(),
            #
            "inter-time.send.min": np.min(inter_times_send) if len(inter_times_send) > 0 else 0,
            "inter-time.send.max": np.max(inter_times_send) if len(inter_times_send) > 0 else 0,
            "inter-time.send.avg": np.average(inter_times_send) if len(inter_times_send) > 0 else 0,
            "inter-time.send.mean": np.mean(inter_times_send) if len(inter_times_send) > 0 else 0,
            "inter-time.send.std": np.std(inter_times_send) if len(inter_times_send) > 0 else 0,
            "inter-time.send.var": np.var(inter_times_send) if len(inter_times_send) > 0 else 0,
            #
            "inter-time.recv.min": np.min(inter_times_recv) if len(inter_times_recv) > 0 else 0,
            "inter-time.recv.max": np.max(inter_times_recv) if len(inter_times_recv) > 0 else 0,
            "inter-time.recv.avg": np.average(inter_times_recv) if len(inter_times_recv) > 0 else 0,
            "inter-time.recv.mean": np.mean(inter_times_recv) if len(inter_times_recv) > 0 else 0,
            "inter-time.recv.std": np.std(inter_times_recv) if len(inter_times_recv) > 0 else 0,
            "inter-time.recv.var": np.var(inter_times_recv) if len(inter_times_recv) > 0 else 0,
            #
            "inter-time.total.min": np.min(inter_times_total) if len(inter_times_total) > 0 else 0,
            "inter-time.total.max": np.max(inter_times_total) if len(inter_times_total) > 0 else 0,
            "inter-time.total.avg": np.average(inter_times_total) if len(inter_times_total) > 0 else 0,
            "inter-time.total.mean": np.mean(inter_times_total) if len(inter_times_total) > 0 else 0,
            "inter-time.total.std": np.std(inter_times_total) if len(inter_times_total) > 0 else 0,
            "inter-time.total.var": np.var(inter_times_total) if len(inter_times_total) > 0 else 0,
        },
        malformed_packets=malformed_packets,
        vlan_etypes=vlan_etypes,
    )


def save_flow_summaries(summaries: dict[str, FlowSummary], file: str):
    pickle.dump(summaries, open(file, "wb"), protocol=pickle.HIGHEST_PROTOCOL)


def load_flow_summaries(file: str) -> dict[str, FlowSummary]:
    if not os.path.exists(file):
        return {}
    return pickle.load(open(file, "rb"))


def extract_features_dict(flow_summary: FlowSummary) -> dict:
    """
    Extract features from FlowSummary.features as a flat dictionary.
    Raises an exception if there are no features.

    Args:
        flow_summary: FlowSummary object

    Returns:
        Dictionary with all features from flow_summary.features
    """
    if flow_summary.features:
        features_dict = {}
        for key, value in flow_summary.features.items():
            if key == "protocols":
                # Convert protocols list to string
                features_dict[key] = "|".join(value) if isinstance(value, list) else str(value)
            elif key in ["start_timestamp", "end_timestamp"]:
                # Convert timestamps to string
                features_dict[key] = str(value)
            else:
                # Keep values as-is
                features_dict[key] = value
        return features_dict

    raise ValueError("FlowSummary does not contain features to extract.")
