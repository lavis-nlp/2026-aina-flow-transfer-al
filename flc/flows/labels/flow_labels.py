from enum import Enum, auto

import pandas as pd


class FlowLabel(Enum):
    MALFORMED = 0  # start counting from 0
    MALFORMED_ONLY_IP = auto()
    UDP = auto()
    UDP_XML = auto()
    UDP_DATA = auto()
    STREAM_SCTP = auto()
    NETOPS_ARP = auto()
    NETOPS_ICMP = auto()
    NETOPS_DNS = auto()
    NETOPS_NTP = auto()
    NETOPS_CDP = auto()
    NETOPS_BROWSER = auto()
    NETOPS_BROWSER_SMB = auto()
    NETOPS_SMB = auto()
    DCERPC = auto()
    TELEPHONY_IPSICTL = auto()
    TELEPHONY_SIP = auto()
    CHAT_XMPP = auto()
    CHAT_XMPP_ENCRYPTED = auto()
    WEB_TLS = auto()
    WEB_HTTP = auto()
    EMAIL_SMTP = auto()
    UNSUCCESSFUL_PORT_CLOSED = auto()
    INCOMPLETE_RESET_AFTER_HANDSHAKE = auto()
    INCOMPLETE_ONLY_TCP_SYN = auto()
    INCOMPLETE_ONLY_TCP_SYN_SYNACK = auto()
    INCOMPLETE_ONLY_TCP_SYNACK = auto()
    NETOPS_NBNS = auto()
    NETOPS_SSDP = auto()
    NETOPS_LLMNR = auto()
    NETOPS_DHCP = auto()
    MONITORING_SYSLOG = auto()
    INCOMPLETE_NO_DATA = auto()
    NETOPS_MDNS = auto()
    NETOPS_LLDP = auto()
    NETOPS_SRVLOC = auto()
    NETOPS_SNMP = auto()
    NETOPS_NATPMP = auto()
    NETOPS_LLTD = auto()
    NETOPS_EIGRP = auto()
    NETOPS_RIP = auto()
    REMOTE_ACCESS_SSH = auto()
    REMOTE_ACCESS_X11 = auto()
    FILE_TRANSFER_FTP = auto()
    NETOPS_SOCKS = auto()
    WEB_JAVARMI = auto()
    INCOMPLETE_ONLY_ONE_PACKET = auto()
    NETOPS_STUN_TCP = auto()
    NETOPS_IGMP = auto()
    UDP_GQUIC = auto()
    NETOPS_STUN_DTLS_UDP = auto()
    NETOPS_DTLS_UDP = auto()
    NETOPS_STUN_UDP = auto()
    NETOPS_DHCPV6 = auto()
    NETOPS_BTDHT = auto()
    TCP_DATA = auto()
    NETOPS_LSD = auto()
    STP = auto()
    DROPBOX_DISCOVERY = auto()
    IPX = auto()
    PRINTER_CANON_BJNP = auto()
    EMAIL_IMAP = auto()
    FILE_TRANSFER_TFTP = auto()
    NETOPS_ECHO = auto()
    NETOPS_RADIUS = auto()
    NETOPS_KERBEROS = auto()
    NETOPS_BFD = auto()
    NETOPS_L2TP = auto()
    NETOPS_GPRS = auto()
    NETOPS_IAPP = auto()
    TELEPHONY_IAX2 = auto()
    NETOPS_DAYTIME = auto()
    TCP_XML = auto()
    NETOPS_NBSS = auto()
    INDUSTRIAL_ETHERNET = auto()
    NETOPS_WSP = auto()
    NETOPS_RX = auto()
    OTHER_DIS = auto()
    OTHER_KNET = auto()
    OTHER_RPC = auto()
    OTHER_LTP = auto()
    OTHER_CHARGEN = auto()
    REMOTE_ACCESS_DEC = auto()

    def __str__(self) -> str:
        if self.value == FlowLabel.MALFORMED.value:
            return "Malformed"
        elif self.value == FlowLabel.MALFORMED_ONLY_IP.value:
            return "Malformed > Only IP nothing more"
        elif self.value == FlowLabel.UDP.value:
            return "UDP"
        elif self.value == FlowLabel.UDP_XML.value:
            return "UDP > XML"
        elif self.value == FlowLabel.UDP_DATA.value:
            return "UDP > DATA"
        elif self.value == FlowLabel.STREAM_SCTP.value:
            return "Stream > SCTP"
        elif self.value == FlowLabel.NETOPS_ARP.value:
            return "Network Operations > ARP"
        elif self.value == FlowLabel.NETOPS_ICMP.value:
            return "Network Operations > ICMP"
        elif self.value == FlowLabel.NETOPS_DNS.value:
            return "Network Operations > DNS"
        elif self.value == FlowLabel.NETOPS_NTP.value:
            return "Network Operations > NTP"
        elif self.value == FlowLabel.NETOPS_CDP.value:
            return "Network Operations > Cisco Discovery Protocol"
        elif self.value == FlowLabel.NETOPS_BROWSER.value:
            return "Network Operations > Microsoft Windows Browser Protocol"
        elif self.value == FlowLabel.NETOPS_BROWSER_SMB.value:
            return "Network Operations > Microsoft Windows SMB and Browser Protocol"
        elif self.value == FlowLabel.NETOPS_SMB.value:
            return "Network Operations > SMB"
        elif self.value == FlowLabel.DCERPC.value:
            return "DCERPC"
        elif self.value == FlowLabel.TELEPHONY_IPSICTL.value:
            return "Telephony > IPSICTL"
        elif self.value == FlowLabel.TELEPHONY_SIP.value:
            return "Telephony > SIP"
        elif self.value == FlowLabel.CHAT_XMPP.value:
            return "Chat > XMPP"
        elif self.value == FlowLabel.CHAT_XMPP_ENCRYPTED.value:
            return "Chat > XMPP Encrypted"
        elif self.value == FlowLabel.WEB_TLS.value:
            return "Web > TLS Encrypted"
        elif self.value == FlowLabel.WEB_HTTP.value:
            return "Web > HTTP"
        elif self.value == FlowLabel.EMAIL_SMTP.value:
            return "E-Mail > SMTP"
        elif self.value == FlowLabel.UNSUCCESSFUL_PORT_CLOSED.value:
            return "Unsuccessful > Port is closed"
        elif self.value == FlowLabel.INCOMPLETE_RESET_AFTER_HANDSHAKE.value:
            return "Incomplete > Reset directly after handshake / port scan"
        elif self.value == FlowLabel.INCOMPLETE_ONLY_TCP_SYN.value:
            return "Incomplete > Only TCP SYNs / SYN flood attack"
        elif self.value == FlowLabel.INCOMPLETE_ONLY_TCP_SYN_SYNACK.value:
            return "Incomplete > Only TCP SYNs and SYN/ACKs"
        elif self.value == FlowLabel.INCOMPLETE_ONLY_TCP_SYNACK.value:
            return "Incomplete > Only TCP SYN/ACKs"
        elif self.value == FlowLabel.NETOPS_NBNS.value:
            return "Network Operations > NetBIOS Name Service"
        elif self.value == FlowLabel.NETOPS_SSDP.value:
            return "Network Operations > SSDP"
        elif self.value == FlowLabel.NETOPS_LLMNR.value:
            return "Network Operations > LLMNR"
        elif self.value == FlowLabel.NETOPS_DHCP.value:
            return "Network Operations > DHCP"
        elif self.value == FlowLabel.MONITORING_SYSLOG.value:
            return "Monitoring > SYSLOG"
        elif self.value == FlowLabel.INCOMPLETE_NO_DATA.value:
            return "Incomplete > TCP connection with handshake and termination, but without data exchange"
        elif self.value == FlowLabel.NETOPS_MDNS.value:
            return "Network Operations > MDNS"
        elif self.value == FlowLabel.NETOPS_LLDP.value:
            return "Network Operations > LLDP"
        elif self.value == FlowLabel.NETOPS_SRVLOC.value:
            return "Network Operations > SRVLOC"
        elif self.value == FlowLabel.NETOPS_SNMP.value:
            return "Network Operations > SNMP"
        elif self.value == FlowLabel.NETOPS_NATPMP.value:
            return "Network Operations > NAT-PMP"
        elif self.value == FlowLabel.NETOPS_LLTD.value:
            return "Network Operations > LLTD"
        elif self.value == FlowLabel.NETOPS_EIGRP.value:
            return "Network Operations > EIGRP"
        elif self.value == FlowLabel.NETOPS_RIP.value:
            return "Network Operations > RIP"
        elif self.value == FlowLabel.REMOTE_ACCESS_SSH.value:
            return "Remote Access > SSH"
        elif self.value == FlowLabel.REMOTE_ACCESS_X11.value:
            return "Remote Access > X11"
        elif self.value == FlowLabel.FILE_TRANSFER_FTP.value:
            return "File Transfer > FTP"
        elif self.value == FlowLabel.NETOPS_SOCKS.value:
            return "Network Operations > SOCKS Proxy"
        elif self.value == FlowLabel.WEB_JAVARMI.value:
            return "Web > Java RMI"
        elif self.value == FlowLabel.INCOMPLETE_ONLY_ONE_PACKET.value:
            return "Incomplete > Only 1 packet"
        elif self.value == FlowLabel.NETOPS_STUN_TCP.value:
            return "Network Operations > STUN over TCP"
        elif self.value == FlowLabel.NETOPS_IGMP.value:
            return "Network Operations > IGMP"
        elif self.value == FlowLabel.UDP_GQUIC.value:
            return "UDP > GQUIC"
        elif self.value == FlowLabel.NETOPS_STUN_DTLS_UDP.value:
            return "Network Operations > STUN/DTLS over UDP"
        elif self.value == FlowLabel.NETOPS_DTLS_UDP.value:
            return "Network Operations > DTLS over UDP"
        elif self.value == FlowLabel.NETOPS_STUN_UDP.value:
            return "Network Operations > STUN over UDP"
        elif self.value == FlowLabel.NETOPS_DHCPV6.value:
            return "Network Operations > DHCPv6"
        elif self.value == FlowLabel.NETOPS_BTDHT.value:
            return "Network Operations > BT-DHT"
        elif self.value == FlowLabel.TCP_DATA.value:
            return "TCP > DATA"
        elif self.value == FlowLabel.NETOPS_LSD.value:
            return "Network Operations > Local Service Discovery"
        elif self.value == FlowLabel.STP.value:
            return "STP"
        elif self.value == FlowLabel.DROPBOX_DISCOVERY.value:
            return "Dropbox Discovery"
        elif self.value == FlowLabel.PRINTER_CANON_BJNP.value:
            return "Printer > Canon BJNP"
        elif self.value == FlowLabel.EMAIL_IMAP.value:
            return "Email > IMAP"
        elif self.value == FlowLabel.FILE_TRANSFER_TFTP.value:
            return "File Transfer > TFTP"
        elif self.value == FlowLabel.NETOPS_ECHO.value:
            return "Network Operations > Echo"
        elif self.value == FlowLabel.NETOPS_RADIUS.value:
            return "Network Operations > RADIUS"
        elif self.value == FlowLabel.NETOPS_KERBEROS.value:
            return "Network Operations > Kerberos"
        elif self.value == FlowLabel.NETOPS_BFD.value:
            return "Network Operations > BFD"
        elif self.value == FlowLabel.NETOPS_L2TP.value:
            return "Network Operations > L2TP"
        elif self.value == FlowLabel.NETOPS_GPRS:
            return "Network Operations > GPRS"
        elif self.value == FlowLabel.NETOPS_IAPP.value:
            return "Network Operations > IAPP"
        elif self.value == FlowLabel.TELEPHONY_IAX2.value:
            return "Telephony > IAX2"
        elif self.value == FlowLabel.NETOPS_DAYTIME.value:
            return "Network Operations > Daytime"
        elif self.value == FlowLabel.TCP_XML.value:
            return "TCP > XML"
        elif self.value == FlowLabel.NETOPS_NBSS.value:
            return "Network Operations > Netbios Session Service"
        elif self.value == FlowLabel.INDUSTRIAL_ETHERNET.value:
            return "Industrial > Ethernet"
        elif self.value == FlowLabel.NETOPS_WSP.value:
            return "Network Operations > WSP"
        elif self.value == FlowLabel.NETOPS_RX.value:
            return "Network Operations > RX"
        elif self.value == FlowLabel.OTHER_KNET.value:
            return "Network Operations > kNet"
        elif self.value == FlowLabel.OTHER_RPC.value:
            return "Remote Procedure Call"
        elif self.value == FlowLabel.OTHER_LTP.value:
            return "Other > Licklider Transmission Protocol (LTP)"
        elif self.value == FlowLabel.OTHER_CHARGEN.value:
            return "Other > Chargen Transmission Protocol (CHARGEN)"
        elif self.value == FlowLabel.MALFORMED.value:
            return "Malformed"
        elif self.value == FlowLabel.REMOTE_ACCESS_DEC.value:
            return "Remote Access > DEC"
        else:
            raise NotImplementedError(f"FlowLabel.str() not implemented for {self.value}")

    @staticmethod
    def invalid_classes() -> set["FlowLabel"]:
        return {
            FlowLabel.UNSUCCESSFUL_PORT_CLOSED,
            FlowLabel.INCOMPLETE_RESET_AFTER_HANDSHAKE,
            FlowLabel.INCOMPLETE_ONLY_TCP_SYN,
            FlowLabel.INCOMPLETE_ONLY_TCP_SYN_SYNACK,
            FlowLabel.INCOMPLETE_ONLY_TCP_SYNACK,
            FlowLabel.INCOMPLETE_NO_DATA,
            FlowLabel.INCOMPLETE_ONLY_ONE_PACKET,
            FlowLabel.MALFORMED,
            FlowLabel.MALFORMED_ONLY_IP,
        }

    @staticmethod
    def labels_from_csv(file: str, add_enum_col: bool = True) -> pd.DataFrame:
        """
        Load flow labels from a csv file into a pd.DataFrame.
        The csv file should have the following columns:
        - flow_id: the flow pcap file name
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
            data["labels"] = data["label_idxs"].apply(lambda x: [FlowLabel(i) for i in x])

        return data

    @staticmethod
    def labels_to_csv(labels: dict[str, list["FlowLabel"]], file: str, zip: bool = True):
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
