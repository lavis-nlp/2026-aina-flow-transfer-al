import os

import pickle
from tqdm import tqdm


def get_pcap_files(root_folder: str, cache_file: str) -> list[str]:
    """
    Retrieves a sorted list of all .pcap files within a certain folder and its subfolders recursively. To avoid
    re-computation every time the program is run, the results are stored in a cache file and loaded from there if the
    file is present.
    :param root_folder: path to the folder where to start looking for .pcap files
    :param cache_file: path to the cache file that will contain the retrieved file paths
    :return: sorted list of filepaths to all .pcap files within the root_folder
    """

    if os.path.exists(cache_file):
        # load from cache if file already exists
        with open(cache_file, "rb") as fp:
            return pickle.load(fp)

    # iterate over all subdirectories recursively and gather filepaths of PCAPs
    flow_pcaps = []
    for root, _, files in tqdm(os.walk(root_folder)):
        for filename in files:
            file_path = os.path.join(root, filename)
            if filename.endswith(".pcap") and "._" not in filename:
                flow_pcaps.append(file_path)
    flow_pcaps.sort()

    # save detected filepaths to cache
    with open(cache_file, "wb") as fp:
        pickle.dump(flow_pcaps, fp, pickle.HIGHEST_PROTOCOL)

    return flow_pcaps
