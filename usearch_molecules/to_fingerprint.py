from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random


import numpy as np


try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, MACCSkeys
    from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage
except ImportError:
    print("Can't fingerprint molecules without RDKit and JPype")


from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from usearch.index import Index, Matches, Key
import stringzilla as sz

from usearch_molecules.to_fingerprint import (
    smiles_to_maccs_ecfp4_fcfp4,
    FingerprintShape,
    shape_maccs,
    shape_mixed,
)

SEED = 42  # For reproducibility
SHARD_SIZE = 1_000_000  # This would result in files between 150 and 300 MB
BATCH_SIZE = 100_000  # A good threshold to split insertions


@dataclass
class FingerprintedEntry:
    """SMILES string augmented with a potentially hybrid fingerprint of known `FingerprintShape` shape."""

    smiles: str
    fingerprint: np.ndarray
    key: Optional[int] = None

    @staticmethod
    def from_table_row(
        table: pa.Table, row: int, shape: FingerprintShape
    ) -> FingerprintedEntry:
        fingerprint = np.zeros(shape.nbytes, dtype=np.uint8)
        progress = 0
        if shape.include_maccs:
            fingerprint[progress : progress + 21] = table["maccs"][row].as_buffer()
            progress += 21
        if shape.nbytes_padding:
            progress += shape.nbytes_padding
        if shape.include_ecfp4:
            fingerprint[progress : progress + 256] = table["ecfp4"][row].as_buffer()
            progress += 256
        if shape.include_fcfp4:
            fingerprint[progress : progress + 256] = table["fcfp4"][row].as_buffer()
            progress += 256
        return FingerprintedEntry(smiles=table["smiles"][row], fingerprint=fingerprint)

    @staticmethod
    def from_parts(
        smiles: str,
        maccs: np.ndarray,
        ecfp4: np.ndarray,
        fcfp4: np.ndarray,
        shape: FingerprintShape,
    ) -> FingerprintedEntry:
        fingerprint = np.zeros(shape.nbytes, dtype=np.uint8)
        progress = 0
        if shape.include_maccs:
            fingerprint[progress : progress + 21] = maccs
            progress += 21
        if shape.nbytes_padding:
            progress += shape.nbytes_padding
        if shape.include_ecfp4:
            fingerprint[progress : progress + 256] = ecfp4
            progress += 256
        if shape.include_fcfp4:
            fingerprint[progress : progress + 256] = fcfp4
            progress += 256
        return FingerprintedEntry(smiles=smiles, fingerprint=fingerprint)


def shard_name(dir: str, from_index: int, to_index: int, kind: str):
    return os.path.join(dir, kind, f"{from_index:0>10}-{to_index:0>10}.{kind}")


def write_table(table: pa.Table, path_out: os.PathLike):
    return pq.write_table(
        table,
        path_out,
        # Without compression the file size may be too large.
        # compression="NONE",
        write_statistics=False,
        store_schema=True,
        use_dictionary=False,
    )


@dataclass
class FingerprintedShard:
    """Potentially cached table and smiles path containing up to `SHARD_SIZE` entries."""

    first_key: int
    name: str

    table_path: os.PathLike
    smiles_path: os.PathLike
    table_cached: Optional[pa.Table] = None
    smiles_caches: Optional[sz.Strs] = None

    @property
    def is_complete(self) -> bool:
        return os.path.exists(self.table_path) and os.path.exists(self.smiles_path)

    @property
    def table(self) -> pa.Table:
        return self.load_table()

    @property
    def smiles(self) -> sz.Strs:
        return self.load_smiles()

    def load_table(self, columns=None, view=False) -> pa.Table:
        if not self.table_cached:
            self.table_cached = pq.read_table(
                self.table_path,
                memory_map=view,
                columns=columns,
            )
        return self.table_cached

    def load_smiles(self) -> sz.Strs:
        if not self.smiles_caches:
            self.smiles_caches = sz.Str(sz.File(self.smiles_path)).splitlines()
        return self.smiles_caches


@dataclass
class FingerprintedDataset:
    dir: os.PathLike
    shards: List[FingerprintedShard]
    shape: Optional[FingerprintShape] = None
    index: Optional[Index] = None

    @staticmethod
    def open(
        dir: os.PathLike,
        shape: Optional[FingerprintShape] = None,
        max_shards: Optional[int] = None,
    ) -> FingerprintedDataset:
        """Gather a list of files forming the dataset."""

        if dir is None:
            return FingerprintedDataset(dir=None, shards=[], shape=shape)

        shards = []
        filenames = sorted(os.listdir(os.path.join(dir, "parquet")))
        if max_shards:
            filenames = filenames[:max_shards]

        for filename in tqdm(filenames, unit="shard"):
            if not filename.endswith(".parquet"):
                continue

            filename = filename.replace(".parquet", "")
            first_key = int(filename.split("-")[0])
            table_path = os.path.join(dir, "parquet", filename + ".parquet")
            smiles_path = os.path.join(dir, "smiles", filename + ".smi")

            shard = FingerprintedShard(
                first_key=first_key,
                name=filename,
                table_path=table_path,
                smiles_path=smiles_path,
            )
            shards.append(shard)

        print(f"Fetched {len(shards)} shards")

        index = None
        if shape:
            index_path = os.path.join(dir, shape.index_name)
            if os.path.exists(index_path):
                index = Index.restore(index_path)

        return FingerprintedDataset(dir=dir, shards=shards, shape=shape, index=index)

    def shard_containing(self, key: int) -> FingerprintedShard:
        for shard in self.shards:
            if shard.first_key <= key and key <= (shard.first_key + SHARD_SIZE):
                return shard

    def head(
        self,
        max_rows: int,
        shape: Optional[FingerprintShape] = None,
        shuffle: bool = False,
    ) -> Tuple[List[str], List[int], np.ndarray]:
        """Load the first part of the dataset. Mostly used for preview and testing."""

        if self.dir is None:
            return None

        if not shape:
            shape = self.shape
        exported_rows = 0
        smiles = []
        keys = []
        fingers = []
        for shard in self.shards:
            table = shard.load_table()
            chunk_size = len(table)
            for i in range(chunk_size):
                entry = FingerprintedEntry.from_table_row(table, i, shape)
                keys.append(exported_rows)
                smiles.append(entry.smiles)
                fingers.append(entry.fingerprint)
                exported_rows += 1

                if exported_rows >= max_rows:
                    break

            if exported_rows >= max_rows:
                break

        smiles = np.array(smiles, dtype=object)
        keys = np.array(keys, dtype=Key)
        fingers = np.vstack(fingers)
        if shuffle:
            permutation = np.arange(len(keys))
            np.random.shuffle(permutation)
            smiles = smiles[permutation]
            keys = keys[permutation]
            fingers = fingers[permutation]
        return smiles, keys, fingers

    def search(
        self,
        smiles: str,
        count: int = 10,
        log: bool = False,
    ) -> List[Tuple[int, str, float]]:
        """Search for similar molecules in the whole dataset."""

        fingers: tuple = smiles_to_maccs_ecfp4_fcfp4(smiles)
        entry = FingerprintedEntry.from_parts(
            smiles,
            fingers[0],
            fingers[1],
            fingers[2],
            self.shape,
        )
        results: Matches = self.index.search(entry.fingerprint, count, log=log)

        filtered_results = []
        for match in results:
            shard = self.shard_containing(match.key)
            row = int(match.key - shard.first_key)
            result = str(shard.smiles[row])
            filtered_results.append((match.key, result, match.distance))

        return filtered_results

    def __len__(self) -> int:
        return len(self.index)

    def random_smiles(self) -> str:
        shard_idx = random.randint(0, len(self.shards) - 1)
        shard = self.shards[shard_idx]
        row = random.randint(0, len(shard.smiles) - 1)
        return str(shard.smiles[row])


if __name__ == "__main__":
    dataset = FingerprintedDataset.open("data/pubchem", shape=shape_maccs)
    dataset.search("C")


def molecule_to_maccs(x):
    return MACCSkeys.GenMACCSKeys(x)


def molecule_to_ecfp4(x):
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048)


def molecule_to_fcfp4(x):
    return AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=2048, useFeatures=True)


def smiles_to_maccs_ecfp4_fcfp4(
    smiles: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uses RDKit to simultaneously compute MACCS, ECFP4, and FCFP4 representations."""

    molecule = Chem.MolFromSmiles(smiles)
    return (
        np.packbits(molecule_to_maccs(molecule)),
        np.packbits(molecule_to_ecfp4(molecule)),
        np.packbits(molecule_to_fcfp4(molecule)),
    )


_cdk = None
_cdk_smiles_parser = None
_cdk_fingerprinter = None


def smiles_to_pubchem(smiles: str) -> Tuple[np.ndarray]:
    """Uses Chemistry Development Kit to compute PubChem representations."""
    global _cdk
    global _cdk_smiles_parser
    global _cdk_fingerprinter

    if not isJVMStarted():
        cdk_path = os.path.join(os.getcwd(), "cdk-2.2.jar")
        startJVM(getDefaultJVMPath(), "-Djava.class.path=%s" % cdk_path)
        _cdk = JPackage("org").openscience.cdk

    if _cdk_smiles_parser is None:
        _cdk_smiles_parser = _cdk.smiles.SmilesParser(
            _cdk.DefaultChemObjectBuilder.getInstance()
        )

    if _cdk_fingerprinter is None:
        _cdk_fingerprinter = _cdk.fingerprint.PubchemFingerprinter(
            _cdk.silent.SilentChemObjectBuilder.getInstance()
        )

    molecule = _cdk_smiles_parser.parseSmiles(smiles)
    cdk_fingerprint = _cdk_fingerprinter.getBitFingerprint(molecule)
    cdk_set_bits = list(cdk_fingerprint.getSetbits())
    bitset = np.zeros(881, dtype=np.uint8)
    bitset[cdk_set_bits] = 1
    bitset = np.packbits(bitset)
    return (bitset,)


@dataclass
class FingerprintShape:
    """Represents the shape of a hybrid fingerprint, potentially containing multiple concatenated bit-vectors."""

    include_maccs: bool = False
    include_ecfp4: bool = False
    include_fcfp4: bool = False
    nbytes_padding: int = 0

    @property
    def nbytes(self) -> int:
        return (
            self.include_maccs * 21
            + self.nbytes_padding
            + self.include_ecfp4 * 256
            + self.include_fcfp4 * 256
        )

    @property
    def nbits(self) -> int:
        return self.nbytes * 8

    @property
    def index_name(self) -> str:
        parts = ["index"]
        if self.include_maccs:
            parts.append("maccs")
        if self.include_ecfp4:
            parts.append("ecfp4")
        if self.include_fcfp4:
            parts.append("fcfp4")
        return "-".join(parts) + ".usearch"


shape_maccs = FingerprintShape(
    include_maccs=True,
    nbytes_padding=3,
)

shape_mixed = FingerprintShape(
    include_maccs=True,
    include_ecfp4=True,
    nbytes_padding=3,
)

shape_ecfp4 = FingerprintShape(
    include_ecfp4=True,
    nbytes_padding=3,
)

shape_fcfp4 = FingerprintShape(
    include_fcfp4=True,
    nbytes_padding=3,
)


