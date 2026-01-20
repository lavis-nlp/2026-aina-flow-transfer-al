import os
import multiprocessing
import signal
from functools import partial

import pandas as pd
from tqdm import tqdm

from flc.flows.scripts.flow_labeling.config import load_config_from_args, Config, Dataset
from flc.flows.labeling.label_flow import label_flow
from flc.flows.labels.label_source import load_labels, save_labels
from flc.flows.feature_extraction import extract_features
from flc.flows.feature_extraction.flow_summary import extract_features_dict
from flc.shared.pcaps.get_pcaps import get_pcap_files

# allow graceful shutdown
running = True


def _process_pcap(
    flow_pcap: str,
    print_unlabeled: bool,
    ignore_tcp_control_packets: bool,
):
    # step 1: always extract features from the flow
    flow_summary = extract_features(
        flow_pcap=flow_pcap,
        ignore_tcp_control_packets=ignore_tcp_control_packets,
    )

    if flow_summary is None:
        # flow could not be processed
        return flow_pcap, None, None

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

    return flow_pcap, flow_labels, flow_summary


def save_features(features: dict, features_file: str):
    """Save extracted features to a CSV file."""
    if not features:
        return

    # convert the features dictionary to a DataFrame
    features_df = pd.DataFrame.from_dict(features, orient="index")

    # set the index name to 'flow_id'
    features_df.index.name = "flow_id"

    # ensure the directory exists
    os.makedirs(os.path.dirname(features_file), exist_ok=True)

    # save the DataFrame to a CSV file
    features_df.to_csv(features_file)


def main():
    config = load_config_from_args()

    for dataset in config.datasets:
        if not running:
            break

        if not dataset.enabled:
            print(f"Skipping {dataset.name}")
            continue

        print(f"{dataset.name} -> {dataset.flow_pcaps_folder}")

        # create dataset folder
        labels_dir = os.path.dirname(dataset.labels_file)
        os.makedirs(labels_dir, exist_ok=True)

        # retrieve a list of all flow PCAP file paths
        flow_pcaps = get_pcap_files(root_folder=dataset.flow_pcaps_folder, cache_file=dataset.filepaths_cache_file)

        print("flows:            ", len(flow_pcaps))

        labels = dict()
        features = dict()

        func = partial(
            _process_pcap,
            print_unlabeled=config.print_unlabeled,
            ignore_tcp_control_packets=config.ignore_tcp_control_packets,
        )

        with tqdm(total=len(flow_pcaps), unit="flows") as pbar:
            for i in range(0, len(flow_pcaps), config.batch_size):
                if not running:
                    break

                batch_pcap_files = flow_pcaps[i : i + config.batch_size]

                # label multiple flow PCAP file simultaneously
                with multiprocessing.Pool(config.parallel_processes) as pool:
                    for _, (pcap_file, flow_labels, flow_summary) in enumerate(
                        pool.imap_unordered(func, batch_pcap_files)
                    ):

                        if flow_labels is None or flow_summary is None:
                            # flow could not be processed
                            print(f"Skipping {pcap_file} due to processing error.")
                            pbar.update(1)
                            continue

                        labels[pcap_file] = flow_labels
                        features[pcap_file] = extract_features_dict(flow_summary)

                        pbar.update(1)

        # sanity check
        assert len(labels) == len(features)

        print("Processed flows:   ", len(labels))
        print("Unprocessed flows: ", len(flow_pcaps) - len(labels))

        # persist results to disk after each batch
        save_labels(labels=dict(labels), labels_file=dataset.labels_file)
        save_features(features=dict(features), features_file=dataset.features_file)


def signal_handler(signum, frame):
    """Gracefully handle termination signals."""
    global running
    print(f"\nReceived signal {signum}. Shutting down...")
    running = False


if __name__ == "__main__":
    # register signal handlers for graceful shutdown
    # signal.signal(signal.SIGINT, signal_handler)  # handle Ctrl+C
    # signal.signal(signal.SIGTERM, signal_handler)  # handle systemctl stop/restart

    main()
