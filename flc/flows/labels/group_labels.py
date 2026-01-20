from enum import Enum, auto

import pandas as pd

from flc.flows.labels.flow_labels import FlowLabel
from flc.flows.labels.label_source import Label


class FlowGroupLabel(Enum):
    Chat = 0  # start counting from 0
    WebTraffic = auto()
    ApplicationService = auto()
    Mail = auto()
    Telephony_VoIP = auto()
    Printing = auto()
    Streaming = auto()
    FileTransfer = auto()
    ResourceSharing = auto()
    RemoteAccess_Administration = auto()
    Routing = auto()
    NameAndServiceDiscovery = auto()
    NetworkInfrastructureService = auto()
    GeneralDataFlow = auto()
    Malformed = auto()
    IncompleteFlow = auto()
    Others = auto()
    Industrial = auto()

    @staticmethod
    def labels_from_csv(file: str, add_enum_col: bool = True) -> pd.DataFrame:
        """
        Load flow labels from a csv file into a pd.DataFrame.
        The csv file should have the following columns:
        - flow_pcap: the flow pcap file name
        - label_idxs: a list of integers representing the label indexes
        - label_names: a list of strings representing the label names
        """
        data = pd.read_csv(file, index_col=None)

        def convert_csv_str_list_to_list(x):
            # x looks like "['str1', 'str2', ...]"
            # so we have parse the string correctly to a list
            x = x[1:-1]  # remove brackets
            x = x.split(",")  # split into elements
            x = [i.strip() for i in x]
            x = [i[1:-1] for i in x]  # remove quotes
            x = [i for i in x if len(i) > 0]  # remove empty strings
            return x

        def convert_csv_int_list_to_list(x):
            return [int(i) for i in x[1:-1].split(",") if len(i) > 0]

        # convert the fields to the correct types
        data["label_idxs"] = data["label_idxs"].apply(convert_csv_int_list_to_list)
        data["label_names"] = data["label_names"].apply(convert_csv_str_list_to_list)

        if add_enum_col:
            data["labels"] = data["label_idxs"].apply(lambda x: [FlowGroupLabel(i) for i in x])

        return data

    @staticmethod
    def labels_to_csv(labels: dict[str, list["FlowGroupLabel"]], file: str, zip: bool = True):
        """
        Save flow labels to a csv file.
        The csv file will have the following columns:
        - flow_id: the flow pcap file name
        - label_idxs: a list of integers representing the label indexes
        - label_names: a list of strings representing the label names
        """

        # convert to pd.DataFrame
        df_data = dict(flow_id=[], label_idxs=[], label_names=[])
        for flow_id, flow_labels in labels.items():
            df_data["flow_id"].append(flow_id)
            df_data["label_idxs"].append([label.value for label in flow_labels])
            df_data["label_names"].append([label.name for label in flow_labels])

        df = pd.DataFrame(df_data)

        # save to csv
        df.to_csv(file, index=False, compression="zip" if zip else None)


def _group2labels_mapping():
    return {
        FlowGroupLabel.Chat: {
            FlowLabel.CHAT_XMPP,
            FlowLabel.CHAT_XMPP_ENCRYPTED,
        },
        FlowGroupLabel.WebTraffic: {
            FlowLabel.WEB_HTTP,
            FlowLabel.WEB_TLS,
            FlowLabel.UDP_GQUIC,
            FlowLabel.NETOPS_WSP,
            FlowLabel.TCP_XML,
        },
        FlowGroupLabel.ApplicationService: {
            FlowLabel.WEB_JAVARMI,
            FlowLabel.DCERPC,
        },
        FlowGroupLabel.Mail: {
            FlowLabel.EMAIL_SMTP,
            FlowLabel.EMAIL_IMAP,
        },
        FlowGroupLabel.Telephony_VoIP: {
            FlowLabel.TELEPHONY_IPSICTL,
            FlowLabel.TELEPHONY_SIP,
            FlowLabel.TELEPHONY_IAX2,
        },
        FlowGroupLabel.Printing: {
            FlowLabel.PRINTER_CANON_BJNP,
        },
        FlowGroupLabel.Streaming: {
            FlowLabel.STREAM_SCTP,
        },
        FlowGroupLabel.FileTransfer: {
            FlowLabel.FILE_TRANSFER_FTP,
            FlowLabel.FILE_TRANSFER_TFTP,
        },
        FlowGroupLabel.ResourceSharing: {
            FlowLabel.NETOPS_SMB,
            FlowLabel.NETOPS_BROWSER_SMB,
            FlowLabel.NETOPS_NBSS,
        },
        FlowGroupLabel.RemoteAccess_Administration: {
            FlowLabel.REMOTE_ACCESS_SSH,
            FlowLabel.REMOTE_ACCESS_X11,
            FlowLabel.REMOTE_ACCESS_DEC,
            FlowLabel.NETOPS_SOCKS,
            FlowLabel.NETOPS_DTLS_UDP,
            FlowLabel.NETOPS_STUN_TCP,
            FlowLabel.NETOPS_STUN_UDP,
            FlowLabel.NETOPS_STUN_DTLS_UDP,
        },
        FlowGroupLabel.Routing: {
            FlowLabel.NETOPS_EIGRP,
            FlowLabel.NETOPS_RIP,
            FlowLabel.NETOPS_BFD,
            FlowLabel.NETOPS_IGMP,  # could also belong to NetworkInfrastructureService
            FlowLabel.DROPBOX_DISCOVERY,
            FlowLabel.IPX,
        },
        FlowGroupLabel.NameAndServiceDiscovery: {
            FlowLabel.NETOPS_DNS,
            FlowLabel.NETOPS_MDNS,
            FlowLabel.NETOPS_LLMNR,
            FlowLabel.NETOPS_NBNS,
            FlowLabel.NETOPS_SSDP,
            FlowLabel.NETOPS_SRVLOC,
            FlowLabel.NETOPS_BROWSER,
            FlowLabel.NETOPS_LSD,
            FlowLabel.NETOPS_BTDHT,
        },
        FlowGroupLabel.NetworkInfrastructureService: {
            FlowLabel.NETOPS_DHCP,
            FlowLabel.NETOPS_DHCPV6,
            FlowLabel.NETOPS_NATPMP,
            FlowLabel.NETOPS_LLTD,
            FlowLabel.NETOPS_LLDP,
            FlowLabel.NETOPS_SNMP,
            FlowLabel.NETOPS_CDP,
            FlowLabel.MONITORING_SYSLOG,
            FlowLabel.NETOPS_ICMP,
            FlowLabel.NETOPS_NTP,
            FlowLabel.NETOPS_ARP,
            FlowLabel.STP,
            FlowLabel.NETOPS_RADIUS,
            FlowLabel.NETOPS_L2TP,
            FlowLabel.NETOPS_KERBEROS,
            FlowLabel.NETOPS_GPRS,
            FlowLabel.NETOPS_ECHO,
            FlowLabel.NETOPS_IAPP,
            FlowLabel.NETOPS_DAYTIME,
            FlowLabel.NETOPS_RX,
        },
        FlowGroupLabel.GeneralDataFlow: {
            FlowLabel.TCP_DATA,
            FlowLabel.UDP,
            FlowLabel.UDP_XML,
            FlowLabel.UDP_DATA,
        },
        FlowGroupLabel.Malformed: {
            FlowLabel.MALFORMED,
            FlowLabel.MALFORMED_ONLY_IP,
        },
        FlowGroupLabel.IncompleteFlow: {
            FlowLabel.UNSUCCESSFUL_PORT_CLOSED,
            FlowLabel.INCOMPLETE_RESET_AFTER_HANDSHAKE,
            FlowLabel.INCOMPLETE_ONLY_TCP_SYN,
            FlowLabel.INCOMPLETE_ONLY_TCP_SYN_SYNACK,
            FlowLabel.INCOMPLETE_ONLY_TCP_SYNACK,
            FlowLabel.INCOMPLETE_NO_DATA,
            FlowLabel.INCOMPLETE_ONLY_ONE_PACKET,
        },
        FlowGroupLabel.Others: {
            FlowLabel.OTHER_KNET,
            FlowLabel.OTHER_LTP,
            FlowLabel.OTHER_CHARGEN,
            FlowLabel.OTHER_DIS,
            FlowLabel.OTHER_RPC,
        },
        FlowGroupLabel.Industrial: {
            FlowLabel.INDUSTRIAL_ETHERNET,
        },
    }


def group2labels(group: FlowGroupLabel) -> set[FlowLabel]:
    mapping = _group2labels_mapping()
    return mapping[group]


def label2group(label: FlowLabel | Label | int) -> "FlowGroupLabel":
    group2label_mapping = _group2labels_mapping()

    # reverse the mapping
    label2group_mapping = {label: group for group, labels in group2label_mapping.items() for label in labels}

    # sanity check: all labels should be in exactly one group
    assert len(label2group_mapping) == len(FlowLabel), f"Missing value for {set(FlowLabel) - set(label2group_mapping)}"

    if isinstance(label, Label):
        label = label.label
    elif isinstance(label, int):
        label = FlowLabel(label)

    return label2group_mapping[label]
