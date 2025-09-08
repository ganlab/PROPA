import os.path
from typing import Tuple
import networkx as nx
import h5py
import numpy as np
from tqdm import tqdm


def create_image_info_from_names(images_names):
    image_info = {}
    id_count = 1
    for image_name in images_names:
        image_id = id_count
        image_info[image_id] = image_name
        id_count += 1
    return image_info


def create_image_ids_from_names(images_names):
    image_ids = {}
    id_count = 1
    for image_name in images_names:
        image_id = id_count
        image_ids[image_name] = image_id
        id_count += 1
    return image_ids


def get_keypoints(path, name) -> np.ndarray:
    with h5py.File(str(path), 'r') as hfile:
        p = hfile[name]['keypoints'].__array__()
    return p


def names_to_pair(name0, name1, separator='/'):
    return separator.join((name0.replace('/', '-'), name1.replace('/', '-')))


def names_to_pair_old(name0, name1):
    return names_to_pair(name0, name1, separator='_')


def find_pair(hfile: h5py.File, name0: str, name1: str):
    pair = names_to_pair(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair(name1, name0)
    if pair in hfile:
        return pair, True
    # older, less efficient format
    pair = names_to_pair_old(name0, name1)
    if pair in hfile:
        return pair, False
    pair = names_to_pair_old(name1, name0)
    if pair in hfile:
        return pair, True
    raise ValueError(
        f'Could not find pair {(name0, name1)}... '
        'Maybe you matched with a different list of pairs? ')


def get_matches(path, name0, name1) -> Tuple[np.ndarray]:
    with h5py.File(str(path), 'r') as hfile:
        pair, reverse = find_pair(hfile, name0, name1)
        matches = hfile[pair]['matches0'].__array__()
        scores = hfile[pair]['matching_scores0'].__array__()
    idx = np.where(matches != -1)[0]
    matches = np.stack([idx, matches[idx]], -1)
    if reverse:
        matches = np.flip(matches, -1)
    scores = scores[idx]
    return matches, scores


def get_pair_connections(points_matches):
    match_list = []
    pair_list = []
    for imageid, match_points in points_matches.items():
        imageid_l = imageid[0]
        imageid_r = imageid[1]
        for match_point in match_points:
            point_l_s = str(imageid_l) + '-' + str(match_point[0])
            point_r_s = str(imageid_r) + '-' + str(match_point[1])
            pair_list.append([point_l_s, point_r_s])

    G = nx.Graph()
    edge_count = {}

    for lst in tqdm(pair_list):
        for i in range(len(lst)):
            id1 = lst[i]
            parts = id1.split("-")
            label1 = parts[0]

            G.add_node(id1, label=label1)

            for j in range(i + 1, len(lst)):
                id2 = lst[j]
                parts = id2.split("-")
                label2 = parts[0]

                G.add_node(id2, label=label2)

                edge = (id1, id2)
                if edge not in edge_count:
                    edge_count[edge] = 0
                edge_count[edge] += 1

                G.add_edge(id1, id2, weight=edge_count[edge])

                subG = G.subgraph(nx.node_connected_component(G, id1))
                labels = [G.nodes[n]['label'] for n in subG]
                if len(set(labels)) < len(labels):
                    G.remove_edge(id1, id2)

    return_list = []
    connected_components = nx.connected_components(G)
    for component in connected_components:
        return_list.append(list(component))

    return return_list


def get_image_info(folder_path):
    features_path = os.path.join(folder_path, 'feats-superpoint-n4096-rmax1600.h5')
    with h5py.File(features_path, 'r') as h5file:
        image_names = list(h5file.keys())
        image_info = create_image_info_from_names(image_names)

    return image_info


def get_matches_and_points_from_h5(folder_path='C:/Users/d1595/Desktop/read_point_match/test/output'):
    pairs_path = os.path.join(folder_path, 'pairs-netvlad.txt')
    features_path = os.path.join(folder_path, 'feats-superpoint-n4096-rmax1600.h5')
    matches_path = os.path.join(folder_path, 'feats-superpoint-n4096-rmax1600_matches-superglue_pairs-netvlad.h5')

    image_info = {}
    with h5py.File(features_path, 'r') as h5file:
        image_names = list(h5file.keys())
        image_info = create_image_info_from_names(image_names)
        image_ids = create_image_ids_from_names(image_names)

    image_points = {}
    for image_id, image_name in image_info.items():
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5
        if image_id in image_points:
            image_points[image_id].extend(keypoints)
        else:
            image_points[image_id] = keypoints

    points_matches = {}
    with open(pairs_path, 'r') as f:
        pairs = [p.split() for p in f.readlines()]
    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)

        high_confidence = (scores > 0.9)
        matches = matches[high_confidence]

        key = (id0, id1)
        if key in points_matches:
            points_matches[key].extend(matches)
        else:
            points_matches[key] = matches

    return image_points, points_matches, image_info
