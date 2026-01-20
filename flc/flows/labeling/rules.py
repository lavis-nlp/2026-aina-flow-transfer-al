from flc.flows.labels import Label, LabelSource, FlowLabel
from flc.flows.feature_extraction import FlowSummary
from flc.flows.feature_extraction import is_udp, is_tcp, is_sctp


def cdx2009_rules(flow: FlowSummary) -> list[Label]:
    protocols = flow.protocols
    tcp_state = flow.tcp_state

    labels = []

    if flow.malformed_packets > 0:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.MALFORMED))

    if flow.features["packets.total"] == 1:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_ONLY_ONE_PACKET))

    """
    Flows which are only UDP nothing more.
    """
    if protocols.keys() == {"_WS.MALFORMED"} or protocols.keys() == {"TCP", "_WS.MALFORMED"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.MALFORMED))

    if tcp_state is None:
        tcp_flags = {}
    else:
        tcp_flags = tcp_state.flags

    """
    Flows which are only UDP nothing more.
    """
    if is_udp(flow.transport_layers) and protocols.keys() == {"UDP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.UDP))

    """
    Flows which are only SCTP nothing more.
    """
    if is_sctp(flow.transport_layers) or protocols.keys() == {"SCTP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.STREAM_SCTP))

    """
    Flows which are only UDP nothing more.
    """
    if protocols.keys() == {"IP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.MALFORMED_ONLY_IP))

    """
    Flows which are only TCP with data.
    """
    if protocols.keys() == {"TCP", "DATA-TEXT-LINES"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.TCP_DATA))

    """
    Flows which transport XML via UDP.
    """
    if is_udp(flow.transport_layers) and protocols.keys() == {"XML"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.UDP_XML))

    """
    Flows which transport undefined DATA via UDP.
    """
    if is_udp(flow.transport_layers) and protocols.keys() == {"DATA"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.UDP_DATA))

    """
    Flows which contain only Address Resolution Protocol (ARP) traffic.
    """
    if protocols.keys() == {"ARP"} or protocols.keys() == {"ARP", "DATA"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_ARP))

    """
    Flows which contain only Internet Control Message Protocol (ICMP) traffic.
    """
    if "ICMP" in protocols.keys() or "ICMPV6" in protocols.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_ICMP))

    """
    Flows which contain only Domain Name System (DNS) traffic.
    """
    if protocols.keys() == {"DNS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_DNS))

    """
    Flows which contain only TCP and DNS are probably DNS traffic.
    """
    if protocols.keys() == {"TCP", "DNS"}:
        # TODO we could differentiate between DNS over UDP and DNS over TCP
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_DNS))

    """
    Flows which contain only NTP traffic.
    """
    if protocols.keys() == {"NTP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_NTP))

    """
    flows which contain only Cisco Discovery Protocol (CDP) traffic

    Cisco Discovery Protocol (CDP) is a proprietary data link layer protocol developed by Cisco Systems in 1994[1] 
    by Keith McCloghrie and Dino Farinacci. It is used to share information about other directly connected Cisco 
    equipment, such as the operating system version and IP address. CDP can also be used for On-Demand Routing, 
    which is a method of including routing information in CDP announcements so that dynamic routing protocols do 
    not need to be used in simple networks. 
    """
    if protocols.keys() == {"CDP"} or protocols.keys() == {"CDP", "IPX"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_CDP))

    """
    Flows which contain only Microsoft Browser traffic.

    Browser service or Computer Browser Service[1] is a feature of Microsoft Windows to let users easily browse and 
    locate shared resources in neighboring computers. This is done by aggregating the information in a single 
    computer "Browse Master" (or "Master Browser"). All other computers contact this computer for information and 
    display in the Network Neighborhood window. 
    """
    if protocols.keys() == {"BROWSER"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_BROWSER))
    if protocols.keys() == {"BROWSER", "SMB", "SMB_NETLOGON"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_BROWSER_SMB))

    """
    Flows that contain SMB.
    """
    if "SMB" in protocols.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_SMB))

    """
    Flows that contain DCERPC.
    """
    if protocols.keys() == {"TCP", "DCERPC"} or protocols.keys() == {"DCERPC"}:
        # TODO find a better place in the hierarchy for this
        labels.append(Label(LabelSource.CDX2009, FlowLabel.DCERPC))

    """
    Flows that contain IPSICTL. 

    IPSICTL (IP Server Interface Control) is a proprietary protocol used by Avaya for communication between components 
    in their telephony systems.
    """
    if protocols.keys() == {"TCP", "IPSICTL"} or protocols.keys() == {"IPSICTL"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.TELEPHONY_IPSICTL))

    """
    Flows with SIP packets are most probably telephony.
    """
    if protocols.keys() == {"SIP"} or protocols.keys() == {"TCP", "SIP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.TELEPHONY_SIP))

    """
    Flows which contain only TCP and XMPP traffic are probably belonging to the chat protocol XMPP.
    """
    if protocols.keys() == {"TCP", "XMPP"} or protocols.keys() == {"TCP", "XMPP", "XML"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.CHAT_XMPP))

    """
    Flows which contain some XMPP and also TLS/DATA are probably encrypted XMPP.
    """
    if protocols.keys() == {"TCP", "XMPP", "TLS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.CHAT_XMPP_ENCRYPTED))
    if protocols.keys() == {"TCP", "XMPP", "TLS", "DATA"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.CHAT_XMPP_ENCRYPTED))
    if protocols.keys() == {"TCP", "XML", "XMPP", "TLS", "PKIXQUALIFIED"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.CHAT_XMPP_ENCRYPTED))

    """
    Flows which contain TLS are probably encrypted web traffic.
    """
    if "TLS" in protocols.keys():
        # TODO does TLS automatically has to be Web?
        labels.append(Label(LabelSource.CDX2009, FlowLabel.WEB_TLS))

    """
    Flows which contain TCP, HTTP and derivatives are probably unencrypted web traffic.
    """
    # TODO maybe we want to distinguish between payload types such as XML, images etc.
    if "HTTP" in protocols.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.WEB_HTTP))

    """
    Flows which contain SMTP are probably email traffic.
    """
    if "SMTP" in protocols.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.EMAIL_SMTP))

    """
    Flows which contain only TCP SYNs and RST/ACK are unsuccessful connections (port is closed).
    """
    if protocols.keys() == {"TCP"} and tcp_flags.keys() == {"SYN", "RST/ACK"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.UNSUCCESSFUL_PORT_CLOSED))

    """
    Flows which start a handshake which is then abruptly reset by the initiating side.
    """
    if protocols.keys() == {"TCP"} and tcp_flags.keys() == {"SYN", "SYN/ACK", "RST"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_RESET_AFTER_HANDSHAKE))
    if protocols.keys() == {"TCP"} and tcp_flags.keys() == {"SYN/ACK", "RST"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_RESET_AFTER_HANDSHAKE))

    """
    Flows which contain only TCP SYNs are at least incomplete at worst an attack.
    """
    if protocols.keys() == {"TCP"} and tcp_flags.keys() == {"SYN"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_ONLY_TCP_SYN))

    """
    Flows which contain only TCP SYNs and/or SYN/ACKs are at least incomplete.
    """
    if protocols.keys() == {"TCP"} and tcp_flags.keys() == {"SYN", "SYN/ACK"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_ONLY_TCP_SYN_SYNACK))
    if protocols.keys() == {"TCP"} and tcp_flags.keys() == {"SYN/ACK"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_ONLY_TCP_SYNACK))

    """
    Flows which contain only NetBIOS Name Service (NBNS).
    """
    if protocols.keys() == {"NBNS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_NBNS))

    """
    Flows which contain only Simple Service Discovery Protocol (SSDP).
    """
    if protocols.keys() == {"SSDP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_SSDP))

    """
    Flows which contain only Link-local Multicast Name Resolution (LLMNR).
    """
    if protocols.keys() == {"LLMNR"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_LLMNR))

    """
    Flows which contain only Dynamic Host Configuration Protocol (DHCP).
    """
    if protocols.keys() == {"DHCP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_DHCP))

    """
    Flows which contain only syslog.
    """
    if protocols.keys() == {"SYSLOG"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.MONITORING_SYSLOG))

    """
    TCP flows that have been established and closed regularly, but did not exchange any data
    """
    if tcp_state and tcp_state.handshake_finished and tcp_state.termination_finished and tcp_state.payload_bytes == 0:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_NO_DATA))

    """
    TCP flows that have been established without any data exchange
    """
    if tcp_state and tcp_state.handshake_finished and tcp_state.flags.keys() == {"ACK", "SYN", "SYN/ACK"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INCOMPLETE_NO_DATA))

    """
    Flows which contain only multicast DNS.
    """
    if protocols.keys() == {"MDNS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_MDNS))

    """
    Flows which contain only multicast DNS.
    """
    if protocols.keys() == {"LLDP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_LLDP))

    """
    Flows which contain only Service Location Protocol (SRVLOC).
    """
    if protocols.keys() == {"SRVLOC"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_SRVLOC))

    """
    Flows which contain only Simple Network Management Protocol (SNMP).
    """
    if protocols.keys() == {"SNMP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_SNMP))

    """
    Flows which contain only NAT Port Mapping Protocol (NAT-PMP).
    """
    if protocols.keys() == {"NAT-PMP"} or protocols.keys() == {"PORTCONTROL"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_NATPMP))

    """
    Flows which contain only Link Layer Topology Discovery (LLTD).
    """
    if protocols.keys() == {"LLTD"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_LLTD))

    """
    Flows which contain only Cisco Enhance Interior Gateway Routing Protocol (EIGRP).
    """
    if protocols.keys() == {"EIGRP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_EIGRP))

    """
    Flows which contain only Routing Information Protocol (RIP).
    """
    if protocols.keys() == {"RIP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_RIP))

    """
    Flows which contain only SSH and TCP are probably SSH connections.
    """
    if protocols.keys() == {"TCP", "SSH"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.REMOTE_ACCESS_SSH))

    """
    Flows which contain X11 are probably X11.
    """
    if "X11" in protocols.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.REMOTE_ACCESS_X11))

    """
    Flows over TCP that contain FTP or FTP DATA packets are probably File Transfer Protocol (FTP).
    """
    if "FTP" in protocols.keys() or "FTP-DATA" in protocols.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.FILE_TRANSFER_FTP))

    """
    Flows which contain only TCP and/or SOCKS are probably SOCKS.
    """
    if protocols.keys() == {"TCP", "SOCKS"} or protocols.keys() == {"SOCKS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_SOCKS))

    """
    Flows which contain Java Remote Method Invocation (Java RMI).
    """
    if protocols.keys() == {"TCP", "RMI"} or protocols.keys() == {"RMI"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.WEB_JAVARMI))

    """
    Flows which contain Trivial File Transfer Protocol (TFTP).
    """
    if protocols.keys() == {"TFTP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.FILE_TRANSFER_TFTP))

    """
    Flows with Echo protocol, see RFC 862.
    """
    if protocols.keys() == {"ECHO"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_ECHO))

    """
    Flows with Remote Authentication Dial-In User Service (RADIUS).
    """
    if protocols.keys() == {"RADIUS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_RADIUS))

    """
    Flows with Kerberos.
    """
    if protocols.keys() == {"KRB4"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_KERBEROS))

    """
    Flows with Bidirectional Forwarding Detection (BFD) protocol.
    """
    if protocols.keys() == {"BFD_ECHO"} or protocols.keys() == {"BFD"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_BFD))

    """
    Flows with Layer 2 Tunneling Protocol (L2TP).
    """
    if protocols.keys() == {"L2TP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_L2TP))

    """
    Flows with GPRS related protocols (like BSSGP).
    """
    if protocols.keys() == {"BSSGP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_GPRS))

    """
    Flows with Inter-Access-Point Protocol.
    """
    if protocols.keys() == {"IAPP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_IAPP))

    """
    Flows with Inter-Asterisk eXchange v2 (IAX2) VoIP protocol.
    """
    if protocols.keys() == {"IAX2"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.TELEPHONY_IAX2))

    """
    Flows with Daytime protocol.
    """
    if protocols.keys() == {"DAYTIME"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_DAYTIME))

    """
    Flows with TCP that transports XML.
    """
    if protocols.keys() == {"TCP", "XML"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.TCP_XML))

    """
    Flows with Netbios Session Service.
    """
    if protocols.keys() == {"TCP", "NBSS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_NBSS))

    """
    Flows with Industrial Ethernet.
    """
    if protocols.keys() == {"CIPIO"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.INDUSTRIAL_ETHERNET))

    """
    Flows with Wireless Session Protocol.
    """
    if protocols.keys() == {"WSP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_WSP))

    """
    Flows with RX Protocol.
    """
    if protocols.keys() == {"RX"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.NETOPS_RX))

    """
    Flows with Distributed Interactive Simulation Protocol.
    """
    if protocols.keys() == {"DIS"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.OTHER_DIS))

    """
    Flows with kNet connection-oriented network protocol.
    """
    if protocols.keys() == {"KNET"} or protocols.keys() == {"TCP", "KNET"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.OTHER_KNET))

    """
    Flows with kNet connection-oriented network protocol.
    """
    if protocols.keys() == {"PORTMAP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.OTHER_RPC))

    """
    Flows with Licklider Transmission Protocol (LTP).
    """
    if protocols.keys() == {"LTP"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.OTHER_LTP))

    """
    Flows with Character Generation Protocol (CHARGEN).
    """
    if protocols.keys() == {"CHARGEN"}:
        labels.append(Label(LabelSource.CDX2009, FlowLabel.OTHER_CHARGEN))

    if 0x6002 in flow.vlan_etypes.keys():
        labels.append(Label(LabelSource.CDX2009, FlowLabel.REMOTE_ACCESS_DEC))

    return labels


def iscxvpn2016_rules(flow: FlowSummary) -> list[Label]:
    protocols = flow.protocols
    transport_layers = flow.transport_layers

    labels = []

    """
    Flows containing Session Traversal Utilities for NAT (STUN).
    """
    if is_tcp(transport_layers) and (protocols.keys() == {"STUN"} or protocols.keys() == {"TCP", "STUN"}):
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_STUN_TCP))

    """
    Flows containing Internet Group Management Protocol (IGMP).
    """
    if protocols.keys() == {"IGMP"}:
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_IGMP))

    """
    Flows that contain BROWSER and SMB.
    """
    if protocols.keys() == {"BROWSER", "SMB"}:
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_BROWSER_SMB))

    """
    Flows that contain Google Quick UDP Internet Connections (GQUIC).
    """
    if is_udp(transport_layers) and protocols.keys() == {"GQUIC"}:
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.UDP_GQUIC))

    """
    Flows that contain STUN and Datagram Transport Layer Security (DTLS) traffic.
    """
    if is_udp(transport_layers) and protocols.keys() == {"STUN", "DTLS", "DATA"}:
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_STUN_DTLS_UDP))
    elif is_udp(transport_layers) and "DTLS" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_DTLS_UDP))

    """
    Flows that contain STUN over UDP.
    """
    if is_udp(transport_layers) and (protocols.keys() == {"STUN"} or protocols.keys() == {"STUN", "DATA"}):
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_STUN_UDP))

    """
    Flows that contain DHCPv6.
    """
    if protocols.keys() == {"DHCPV6"}:
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_DHCPV6))

    """
    Flows that contain BT-DHT.
    """
    if "BT-DHT" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_BTDHT))

    """
    Flows that contain LSD.
    """
    if is_udp(transport_layers) and "LSD" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.NETOPS_LSD))

    # TODO STP is transport layer
    if "STP" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.STP))

    """
    Flows that contain Dropbox Discovery
    """
    if is_udp(transport_layers) and "DB-LSP-DISC" in protocols.keys() and "JSON" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.DROPBOX_DISCOVERY))

    """
    Flows based on the IPX protocol.
    """
    if "IPX" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.IPX))

    """
    Flows that transport data over UDP.
    """
    if is_udp(transport_layers) and "DATA" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.UDP_DATA))

    """
    Flows that contain the Canon BJNP printer protocol.
    """
    if "BJNP" in protocols.keys():
        labels.append(Label(LabelSource.ISCXVPN2016, FlowLabel.PRINTER_CANON_BJNP))

    return labels


def cicids2012_rules(flow: FlowSummary) -> list[Label]:
    labels = []

    if "IMAP" in flow.protocols.keys():
        labels.append(Label(LabelSource.CICIDS2012, FlowLabel.EMAIL_IMAP))

    return labels
