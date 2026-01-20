import csv
import io
import subprocess
import sys
from datetime import datetime
from enum import Enum
from ipaddress import ip_address, IPv4Address
from typing import Optional


class Layer4Proto(Enum):
    TCP = 6
    UDP = 17
    SCTP = 132


class Packet:
    time_epoch: float = None
    length: int = None
    capture_length: int = None
    eth_src: Optional[str] = None
    eth_dst: Optional[str] = None
    ip_src: Optional[IPv4Address] = None
    ip_dst: Optional[IPv4Address] = None
    ip_proto: Optional[int] = None
    port_src: Optional[int] = None
    port_dst: Optional[int] = None
    tcp_flags: Optional[int] = None
    payload_size: Optional[int] = None
    protocols: list[str] = None
    vlan_etype: Optional[int] = None
    malformed: Optional[bool] = None

    @property
    def highest_layer(self):
        return self.protocols[-1]

    @property
    def layers(self):
        return self.protocols

    @property
    def sniff_time(self):
        return datetime.fromtimestamp(self.time_epoch)

    @property
    def transport_layer(self):
        # TODO support more types
        if self.ip_proto == 6:
            return Layer4Proto.TCP
        elif self.ip_proto == 17:
            return Layer4Proto.UDP
        elif self.ip_proto == 132:
            return Layer4Proto.SCTP
        else:
            return None

    def __eq__(self, other):
        if not isinstance(other, Packet):
            return False

        return (
            self.length == other.length
            and self.capture_length == other.capture_length
            and self.eth_src == other.eth_src
            and self.eth_dst == other.eth_dst
            and self.ip_src == other.ip_src
            and self.ip_dst == other.ip_dst
            and self.ip_proto == other.ip_proto
            and self.port_src == other.port_src
            and self.port_dst == other.port_dst
            and self.tcp_flags == other.tcp_flags
            and self.payload_size == other.payload_size
            and self.protocols == other.protocols
            and self.vlan_etype == other.vlan_etype
            and self.malformed == other.malformed
        )

    def __hash__(self):
        return hash(
            (
                self.length,
                self.capture_length,
                self.eth_src,
                self.eth_dst,
                str(self.ip_src) if self.ip_src else None,  # Convert IPv4Address to string
                str(self.ip_dst) if self.ip_dst else None,  # Convert IPv4Address to string
                self.ip_proto,
                self.port_src,
                self.port_dst,
                self.tcp_flags,
                self.payload_size,
                tuple(self.protocols) if self.protocols else None,
                self.vlan_etype,
                self.malformed,
            )
        )


def parse_packets_from_csv(csv_dict_reader: csv.DictReader) -> list[Packet]:
    packets = []
    for row in csv_dict_reader:
        packet = Packet()

        packet.time_epoch = float(row["frame.time_epoch"])
        packet.length = int(row["frame.len"])
        packet.capture_length = int(row["frame.cap_len"])

        # TODO parse MAC addresses
        if len(row["eth.src"]) > 0:
            packet.eth_src = row["eth.src"]
        if len(row["eth.dst"]) > 0:
            packet.eth_dst = row["eth.dst"]

        if len(row["ip.src"]) > 0:
            packet.ip_src = ip_address(row["ip.src"])
        if len(row["ip.dst"]) > 0:
            packet.ip_dst = ip_address(row["ip.dst"])
        if len(row["ip.dst"]) > 0:
            packet.ip_proto = int(row["ip.proto"])

        if len(row["tcp.srcport"]) > 0:
            packet.port_src = int(row["tcp.srcport"])
            packet.ip_proto = 6  # fix because ip_proto field is named differently in IPv6
        if len(row["tcp.dstport"]) > 0:
            packet.port_dst = int(row["tcp.dstport"])

        if len(row["tcp.len"]) > 0:
            packet.payload_size = int(row["tcp.len"])
        else:
            packet.payload_size = 0
        if len(row["tcp.flags"]) > 0:
            packet.tcp_flags = int(row["tcp.flags"], base=16)
        else:
            packet.tcp_flags = None

        if len(row["udp.srcport"]) > 0:
            packet.port_src = int(row["udp.srcport"])
            packet.ip_proto = 17  # fix because ip_proto field is named differently in IPv6

            if len(row["udp.length"]) > 0:
                # udp.length includes the size of the UPD header, so we subtract 8 bytes
                packet.payload_size = int(row["udp.length"]) - 8
            else:
                packet.payload_size = 0

        if len(row["udp.dstport"]) > 0:
            packet.port_dst = int(row["udp.dstport"])

        if row["frame.protocols"] is not None:
            packet.protocols = list(map(lambda x: x.upper(), row["frame.protocols"].split(":")))
        else:
            # FIXME not sure how this happens
            continue

        if row["vlan.etype"] is not None and len(row["vlan.etype"]) > 0:
            packet.vlan_etype = int(row["vlan.etype"], 16)

        if row["_ws.malformed"] is not None and len(row["_ws.malformed"]) > 0:
            packet.malformed = True

        packets.append(packet)

    # sort packets by time
    packets.sort(key=lambda x: x.time_epoch)

    return packets


def parse_pcap(filepath: str) -> list[Packet]:
    cmd = [
        "tshark",
        "-T",
        "fields",
        "-e",
        "frame.time_epoch",
        "-e",
        "frame.len",
        "-e",
        "frame.cap_len",
        "-e",
        "eth.src",
        "-e",
        "eth.dst",
        "-e",
        "ip.src",
        "-e",
        "ip.dst",
        "-e",
        "ip.proto",
        "-e",
        "tcp.srcport",
        "-e",
        "tcp.dstport",
        "-e",
        "tcp.flags",
        "-e",
        "tcp.len",
        "-e",
        "udp.srcport",
        "-e",
        "udp.dstport",
        "-e",
        "udp.length",
        "-e",
        "frame.protocols",
        "-e",
        "vlan.etype",
        "-e",
        "_ws.malformed",
        "-E",
        "header=y",
        "-E",
        "separator=,",
        "-E",
        "quote=d",
        "-E",
        "occurrence=f",
        "-r",
        filepath,
    ]
    process = subprocess.run(cmd, capture_output=True)
    if process.returncode != 0:
        print(process.stdout.decode("ascii"))
        print(process.stderr.decode("ascii"), file=sys.stderr)
        raise RuntimeError(
            f'executing "{" ".join(cmd)}" returned with non-zero exit code: {process.returncode}, '
            f"stdout and stderr output is printed above"
        )

    result = process.stdout.decode("utf-8")
    csv_dict_reader = csv.DictReader(io.StringIO(result))
    return parse_packets_from_csv(csv_dict_reader)


def parse_tshark_csv(flow_csv: str) -> list[Packet]:
    try:
        with open(flow_csv) as fp:
            csv_dict_reader = csv.DictReader(fp)
            return parse_packets_from_csv(csv_dict_reader)
    except Exception as e:
        print(f"Error parsing flow csv: {flow_csv}", file=sys.stderr)
        raise e
