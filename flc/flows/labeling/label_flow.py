import time

from flc.flows.feature_extraction import extract_features
from flc.flows.feature_extraction.flow_summary import FlowSummary
from flc.flows.labels import Label
from flc.flows.labeling.rules import cdx2009_rules, iscxvpn2016_rules, cicids2012_rules


def label_flow(flow_summary: FlowSummary) -> list[Label]:
    """
    Label a particular flow based on the input features. Currently, the features are protocols and TCP flags.
    :param flow_summary:
    :return: label for the flow as string or None if no label was found according to the rules
    """

    cdx2009_label = cdx2009_rules(flow=flow_summary)
    iscxvpn2016_label = iscxvpn2016_rules(flow=flow_summary)
    cicids2012_label = cicids2012_rules(flow=flow_summary)

    return cdx2009_label + iscxvpn2016_label + cicids2012_label


def label_flow_pcap(
    flow_pcap, print_unlabeled: bool = False, ignore_tcp_control_packets: bool = True
) -> tuple[str, list[Label] | None]:
    """
    Tries to assign a label to a PCAP that presumably contains a single flow by first parsing the PCAP file, extracting
    meaningful input features and then applying a set of rules to determine a possible label.
    :param flow_pcap: file path to the PCAP file containing a single flow
    :param print_unlabeled: if set to True flows that were not labeled successfully are printed to the console
    :param ignore_tcp_control_packets: if set to True packets like TCP ACK, SYN, RST and FIN will be ignored
    :return: tuple of the file path and the labels or None if no label was found
    """

    start_time = time.time()

    # step 1: extract features from the flow
    flow_summary = extract_features(flow_pcap, ignore_tcp_control_packets=ignore_tcp_control_packets)
    if flow_summary is None:
        return flow_pcap, None

    # step 2: use rules to classify the flow based on the features
    flow_labels = label_flow(flow_summary=flow_summary)

    if print_unlabeled and len(flow_labels) == 0:
        if flow_summary.tcp_state is None:
            print("file://" + flow_pcap, dict(flow_summary.protocols))
        elif flow_summary.protocols.keys() != {"TCP", "DATA"} and flow_summary.protocols.keys() != {"TCP"}:
            print(
                "file://" + flow_pcap,
                dict(flow_summary.protocols),
                flow_summary.tcp_state.flags,
                f"tcp_handshake={flow_summary.tcp_state.handshake_finished}",
                f"tcp_termination={flow_summary.tcp_state.termination_finished}",
                f"tcp_payload_bytes={flow_summary.tcp_state.payload_bytes}",
            )

    elapsed_time = time.time() - start_time
    if elapsed_time > 10:
        print(f"file://{flow_pcap} took more than 10s to analyze")

    return flow_pcap, flow_labels
