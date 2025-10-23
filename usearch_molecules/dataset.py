import warnings
from functools import cached_property
from typing import List
from dataclasses import dataclass
import os

import numpy as np
import pyarrow.parquet as pq
import stringzilla as sz

from usearch.index import Index, CompiledMetric, MetricKind, MetricSignature, ScalarKind

from usearch_molecules.metrics import tanimoto_maccs, tanimoto_ecfp4, tanimoto_fcfp4
from usearch_molecules.utils import smiles_to_maccs_ecfp4_fcfp4

class IncompleteIndexError(Exception):
    pass

class DataIntegrityError(Exception):
    pass

def mono_index(dataset, shape, metric):
    """
    while the library can support mixed indexing I am not really using them at the moment
    this is just a mono index for a fingerprint shape and the only metric we are using is tanimoto
    :param dataset: FingerPringDataset instance see below
    :param shape: FingerPringShape instance see below
    :param metric: this is an implementation of tanimoto using numba, see metrics.py
    :return: notihg but creates a file with index.<fingerprint-type>.usearch file
    """
    index_path = os.path.join(dataset.dir, f"index-{shape.name}.usearch")

    index=Index(
        ndim=shape.nbits,
        dtype=ScalarKind.B1,
        metric=CompiledMetric(
            pointer=metric.address,
            kind=MetricKind.Tanimoto,
            signature=MetricSignature.ArrayArray)
    )
    for shard_idx, shard in enumerate(dataset.shards):
        table = shard.load_table()
        start=int(list(os.path.split(shard.shard_path)).pop().split("-")[0])
        n = len(table)
        end=start+n
        keys = np.arange(start, end)
        vectors = np.vstack(
            [
                FingerprintedEntry.from_table_row(table, i, shape).fingerprint
                for i in range(n)
            ]
        )
        index.add(keys, vectors)
        dataset.shards[shard_idx].table_cached = None
        dataset.shards[shard_idx].index_cached = None
    index.save(index_path)
    index.reset()
    return index_path

class FingerprintShape:
    def __init__(self, include_maccs=False,
                 include_ecfp4=False,
                 include_fcfp4=False,
                 nbytes_padding=0):
        """
        basic allocation of bits and bytes for the fingerprint, this can be a single fingerprint
        or a combination of maccs, ecfp4 and/or fcpf4 while the mixed shape is not used currenly with a little bit of
        refactoring it's possible
        :param include_maccs: whether to include this in the shape
        :param include_ecfp4: same as above
        :param include_fcfp4: same as above
        """

        self.include_maccs = include_maccs
        self.include_ecfp4 = include_ecfp4
        self.include_fcfp4 = include_fcfp4
        self.nbytes_padding=nbytes_padding

    @property
    def nbytes(self):
        return (self.include_maccs * 21
            +self.nbytes_padding
            + self.include_ecfp4 * 256
            + self.include_fcfp4 * 256)

    @property
    def nbits(self):
        return self.nbytes * 8

    @property
    def name(self):
        parts = []
        if self.include_maccs:
            parts.append("maccs")
        if self.include_ecfp4:
            parts.append("ecfp4")
        if self.include_fcfp4:
            parts.append("fcfp4")
        return "-".join(parts)

@dataclass
class FingerprintedEntry:
    """
    a molecule's representation with the fingerprints
    """
    smiles:str
    fingerprint: np.ndarray

    @staticmethod
    def from_table_row(table, row, shape):
        fingerprint = np.zeros(shape.nbytes, dtype=np.uint8)
        progress = 0
        if shape.include_maccs:
            fingerprint[progress: progress + 21] = table["maccs"][row].as_buffer()
            progress += 21
        if shape.include_ecfp4:
            fingerprint[progress: progress + 256] = table["ecfp4"][row].as_buffer()
            progress += 256
        if shape.include_fcfp4:
            fingerprint[progress: progress + 256] = table["fcfp4"][row].as_buffer()
            progress += 256
        return FingerprintedEntry(smiles=table["smiles"][row], fingerprint=fingerprint)

    @staticmethod
    def from_parts(smiles, maccs, ecfp4, fcfp4, shape):
        progress = 0
        fingerprint = np.zeros(shape.nbytes, dtype=np.uint8)
        if shape.include_maccs:
            fingerprint[progress: progress + 21] = maccs
            progress += 21
        if shape.nbytes_padding:
            progress += shape.nbytes_padding
        if shape.include_ecfp4:
            fingerprint[progress: progress + 256] = ecfp4
            progress += 256
        if shape.include_fcfp4:
            fingerprint[progress: progress + 256] = fcfp4
            progress += 256
        return FingerprintedEntry(smiles=smiles, fingerprint=fingerprint)


class FingerprintedShard:
    """
    this is to be used with a fingerpringdataset see below, each shard contains a subset (defined by rawdataset) of all the
    molecules
    """
    def __init__(self, shard_path, smiles_path, table_cached=None, smiles_caches=None):
        self.shard_path = shard_path
        self.smiles_path = smiles_path
        self.table_cached = table_cached
        self.smiles_caches = smiles_caches
        self.start=int(list(os.path.split(shard_path)).pop().split("-")[0])
        self.end=self.start+len(self.smiles)-1

    @property
    def is_complete(self):
        return os.path.exists(self.shard_path) and os.path.exists(self.smiles_path)

    @property
    def table(self):
        return self.load_table()

    @property
    def smiles(self):
        return self.load_smiles()

    def load_table(self, columns=None, view=False):
        if not self.table_cached:
            self.table_cached = pq.read_table(
                self.shard_path,
                memory_map=view,
                columns=columns,
            )
        return self.table_cached

    def load_smiles(self):
        if not self.smiles_caches:
            self.smiles_caches = sz.File(self.smiles_path).splitlines()
        return self.smiles_caches


class FingerprintedDataset:
    def __init__(self, data_dir, shapes:List[FingerprintShape]):
        """
        This will convert a raw dataset to a fingerprinted dataset but creating indices and loading shards
        :param raw_dataset: rawdataset instance from usearch_molecules.raw_dataset
        """
        self.dir = data_dir
        self.shapes = shapes
        self.indexed = False
        self.indices=self.get_indices()
        self.shards=self.get_shards()


    def get_shards(self):
        if not os.path.exists(os.path.join(self.dir, "parquet")):
            raise NotADirectoryError("Parquet director not found")

        if not os.path.exists(os.path.join(self.dir, "smiles")):
            raise NotADirectoryError("SMILES directory not found")

        data = []
        smiles=[os.path.join(self.dir, "smiles", file) for file in os.listdir(os.path.join(self.dir, "smiles"))]
        smiles.sort()
        parquets=[os.path.join(self.dir, "parquet", file) for file in os.listdir(os.path.join(self.dir, "parquet"))]
        parquets.sort()

        if len(smiles) != len(parquets):
            raise ValueError("Number of smiles and parquet files does not match")

        for smi, parquet in zip(smiles, parquets):
            if smi.replace("smiles", "")==parquet.replace("parquet", ""):
                sh=FingerprintedShard(parquet, smi)
                data.append(sh)
            else:
                raise DataIntegrityError(f"{smi} and {parquet} names do not match")
        return data

    @property
    def is_indexed(self):
        indices={}
        for shape in self.shapes:
            index_path=os.path.join(self.dir, "index-"+shape.name+".usearch")
            if os.path.exists(index_path):
                indices[shape.name] = index_path
            else:
                indices[shape.name]=None
        if all(indices.values()):
            return True
        else:
            return False

    def get_indices(self):
        indices={}
        if self.is_indexed:
            for shape in self.shapes:
                index_path = os.path.join(self.dir, shape.name)
                indices[shape.name] = index_path
        else:
            warnings.warn("The dataset is not indexed")

        return indices

    def index(self):
        if self.indexed:
            return(f"{self.dir} is already indexed")
        else:
            # this is hardcoded but how many different kinds can you come up with here
            for shape in self.shapes:
                if shape.name=="maccs":
                    metric=tanimoto_maccs
                    self.indices["maccs"]=mono_index(self, shape, metric=metric)
                elif shape.name=="ecfp4":
                    metric=tanimoto_ecfp4
                    self.indices["ecfp4"]=mono_index(self, shape, metric=metric)
                elif shape.name=="fcfp4":
                    metric=tanimoto_fcfp4
                    self.indices["fcfp4"]=mono_index(self, shape, metric=metric)
                else:
                    raise NotImplementedError(f"{shape.name} indexing is not implemented")


    def search(self, smiles, shape, n=1000):
        if not self.is_indexed:
            raise IncompleteIndexError("The dataset is not full indexed please run index before searching")

        fingers = smiles_to_maccs_ecfp4_fcfp4(smiles)
        entry = FingerprintedEntry.from_parts(
            smiles,
            fingers[0],
            fingers[1],
            fingers[2],
            shape,
        )
        index=Index.restore(os.path.join(self.dir, "index-"+shape.name+".usearch"))
        results=index.search(entry.fingerprint, n)


        filtered_results = []
        for match in results:
            shard = self._shard_containing(match.key)
            row = int(match.key - shard.start)
            result = str(shard.smiles[row])
            filtered_results.append((match.key, result, match.distance))

        return filtered_results

    def open(self):
        for data in self.shards:
            data.table()
            data.smiles()

    def _shard_containing(self, key: int):
        to_ret=None
        for shard in self.shards:
            if shard.start <= key <= shard.end:
                to_ret=shard

        return to_ret




shape_maccs = FingerprintShape(
    include_maccs=True,
    nbytes_padding=0,
)

shape_ecfp4 = FingerprintShape(
    include_ecfp4=True,
    nbytes_padding=0,
)

shape_fcfp4 = FingerprintShape(
    include_fcfp4=True,
    nbytes_padding=0,
)

shape_mixed = FingerprintShape(
    include_maccs=True,
    include_ecfp4=True,
    include_fcfp4=True,
    nbytes_padding=3,
)