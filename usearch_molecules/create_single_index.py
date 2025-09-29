import os
import argparse

import numpy as np
import pyarrow as pa

from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind

from usearch_molecules.metrics_numba import (
    tanimoto_maccs,
    tanimoto_ecfp4,
    tanimoto_fcfp4
)
from usearch_molecules.dataset import (
    FingerprintedDataset,
    FingerprintedEntry,
)
from dataset import (
    shape_maccs,
    shape_ecfp4,
    shape_fcfp4
)

from usearch_molecules.prep_smiles import export_smiles

def mono_index_maccs(dataset):
    index_path_maccs = os.path.join(dataset.dir, "index-maccs.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)
    index_maccs = Index(
        ndim=shape_maccs.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_maccs.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_maccs,
    )
    for shard_idx, shard in enumerate(dataset.shards):
        if shard.first_key in index_maccs:
            continue

        table = shard.load_table(["maccs"])
        n = len(table)

        # No need to shuffle the entries as they already are:
        keys = np.arange(shard.first_key, shard.first_key + n)
        maccs_fingerprints = [table["maccs"][i].as_buffer() for i in range(n)]

        # First construct the index just for MACCS representations
        vectors = np.vstack(
            [
                FingerprintedEntry.from_parts(
                    None,
                    maccs_fingerprints[i],
                    None,
                    None,
                    shape_maccs,
                ).fingerprint
                for i in range(n)
            ]
        )
        index_maccs.add(keys, vectors)
        dataset.shards[shard_idx].table_cached = None
        dataset.shards[shard_idx].index_cached = None

    index_maccs.save(index_path_maccs)
    index_maccs.reset()

def mono_index_ecfp4(dataset):
    index_path_ecfp4 = os.path.join(dataset.dir, "index-ecfp4.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)
    index_ecfp4 = Index(
        ndim=shape_ecfp4.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_ecfp4.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_maccs,
    )
    for shard_idx, shard in enumerate(dataset.shards):
        if shard.first_key in index_ecfp4:
            continue

        table = shard.load_table(["ecfp4"])
        n = len(table)

        # No need to shuffle the entries as they already are:
        keys = np.arange(shard.first_key, shard.first_key + n)
        ecfp4_fingerprints = [table["ecfp4"][i].as_buffer() for i in range(n)]

        # First construct the index just for MACCS representations
        vectors = np.vstack(
            [
                FingerprintedEntry.from_parts(
                    None,
                    None,
                    ecfp4_fingerprints[i],
                    None,
                    shape_ecfp4,
                ).fingerprint
                for i in range(n)
            ]
            )

        index_ecfp4.add(keys, vectors)
        dataset.shards[shard_idx].table_cached = None
        dataset.shards[shard_idx].index_cached = None

    index_ecfp4.save(index_path_ecfp4)
    index_ecfp4.reset()

def mono_index_fcfp4(dataset):
    index_path_fcfp4 = os.path.join(dataset.dir, "index-fcfp4.usearch")
    os.makedirs(os.path.join(dataset.dir), exist_ok=True)
    index_fcfp4 = Index(
        ndim=shape_fcfp4.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=tanimoto_fcfp4.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray,
        ),
        # path=index_path_maccs,
    )
    for shard_idx, shard in enumerate(dataset.shards):
        if shard.first_key in index_fcfp4:
            continue

        table = shard.load_table(["fcfp4"])
        n = len(table)

        # No need to shuffle the entries as they already are:
        keys = np.arange(shard.first_key, shard.first_key + n)
        fcfp4_fingerprints = [table["fcfp4"][i].as_buffer() for i in range(n)]

        # First construct the index just for MACCS representations
        vectors = np.vstack(
            [
                FingerprintedEntry.from_parts(
                    None,
                    None,
                    None,
                    fcfp4_fingerprints[i],
                    shape_fcfp4,
                ).fingerprint
                for i in range(n)
            ]
            )
        index_fcfp4.add(keys, vectors)
        dataset.shards[shard_idx].table_cached = None
        dataset.shards[shard_idx].index_cached = None

    index_fcfp4.save(index_path_fcfp4)
    index_fcfp4.reset()











