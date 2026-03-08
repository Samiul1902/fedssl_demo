# src/federated/partition.py
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
import re
import random

CASE_RE = re.compile(r"(case_\d{5})", re.IGNORECASE)

def extract_case_id(filename: str) -> str:
    m = CASE_RE.search(filename)
    if not m:
        raise ValueError(f"Could not extract case id from: {filename}")
    return m.group(1).lower()

def group_indices_by_case(meta_filename_list: List[str]) -> Dict[str, List[int]]:
    cases = defaultdict(list)
    for idx, fn in enumerate(meta_filename_list):
        cid = extract_case_id(fn)
        cases[cid].append(idx)
    return dict(cases)

def iid_case_split(case_ids: List[str], num_clients: int, seed: int = 42) -> List[List[str]]:
    rng = random.Random(seed)
    case_ids = list(case_ids)
    rng.shuffle(case_ids)
    return [case_ids[i::num_clients] for i in range(num_clients)]

def build_client_indices_from_cases(case_to_indices: Dict[str, List[int]],
                                   client_cases: List[List[str]]) -> List[List[int]]:
    client_indices = []
    for cases in client_cases:
        idxs = []
        for c in cases:
            idxs.extend(case_to_indices[c])
        client_indices.append(sorted(idxs))
    return client_indices

import random

def noniid_split_by_tumor_burden(case_ids: List[str],
                                case_burden: Dict[str, float],
                                num_clients: int,
                                seed: int = 42) -> List[List[str]]:
    """
    Non-IID split: sort cases by tumor burden, then assign in round-robin.
    This creates clients skewed toward low/high tumor burden.
    """
    rng = random.Random(seed)
    sorted_cases = sorted(case_ids, key=lambda c: case_burden[c])

    client_cases = [[] for _ in range(num_clients)]
    for i, c in enumerate(sorted_cases):
        client_cases[i % num_clients].append(c)

    # small shuffle within each client list (doesn't change burden distribution)
    for lst in client_cases:
        rng.shuffle(lst)

    return client_cases