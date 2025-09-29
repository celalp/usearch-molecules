"""Exports all molecules from the PubChem, GDB13 and Enamine REAL datasets into Parquet shards, with up to 1 Million molecules in every granule."""

import os
import logging
from dataclasses import dataclass
from typing import List, Callable, Optional, Tuple
from multiprocessing import Process, cpu_count

import pyarrow as pa
from stringzilla import File, Strs, Str

from usearch_molecules.dataset import shard_name, write_table, SHARD_SIZE, SEED

logger = logging.getLogger(__name__)


@dataclass
class RawDataset:
    lines: Strs
    extractor: Callable

    def count_lines(self) -> int:
        return len(self.lines)

    def smiles(self, row_idx: int) -> Optional[str]:
        return self.extractor(str(self.lines[row_idx]))

    def smiles_slice(self, count_to_skip: int, max_count: int) -> List[Tuple[int, str]]:
        result = []

        count_lines = len(self.lines)
        for row_idx in range(count_to_skip, count_lines):
            smiles = self.smiles(row_idx)
            if smiles:
                result.append((row_idx, smiles))
                if len(result) >= max_count:
                    return result

        return result


def example(dir: os.PathLike) -> RawDataset:
    """
    gzip -d CID-SMILES.gz
    """

    lines = Strs()
    filenames = (
        "0000000000-0001000000.smi",
        "0001000000-0002000000.smi",
    )
    for filename in filenames:
        file = File(os.path.join(dir, "smiles", filename))
        lines.extend(file.splitlines())

    # Let's shuffle across all the files
    lines.shuffle(SEED)

    def extractor(row: str) -> Optional[str]:
        row = row.strip("\n")
        if len(row) > 0:
            return row
        return None

    return RawDataset(
        lines=lines,
        extractor=extractor,
    )


def pubchem(dir: os.PathLike, filename: str = "CID-SMILES") -> RawDataset:
    """
    gzip -d CID-SMILES.gz
    """

    file = File(os.path.join(dir, filename))
    file = file.splitlines().shuffled(SEED)

    def extractor(row: str) -> Optional[str]:
        row = row.strip("\n")
        if "\t" in row:
            row = row.split("\t")[-1]
            return row
        return None

    return RawDataset(
        lines=file,
        extractor=extractor,
    )


def gdb13(dir: os.PathLike) -> RawDataset:
    """GDB13 dataset only requires concatenating

    tar -xvzf gdb13.tgz
    """

    lines = Strs()
    for i in range(1, 14):
        file = File(os.path.join(dir, f"{i}.smi"))
        lines.extend(file.splitlines())

    # Let's shuffle across all the files
    lines.shuffle(SEED)

    def extractor(row: str) -> Optional[str]:
        row = row.strip("\n")
        if len(row) > 0:
            return row
        return None

    return RawDataset(
        lines=lines,
        extractor=extractor,
    )


def real(dir: os.PathLike):
    """Enamine REAL dataset only requires both cleanup and concatenating

    This dataset is so large, that we split the process into a few steps.

    1.  Decompress `.cxsmiles.bz2` files into `.cxsmiles`.
    2.  Wipe metadata, exporting `.smiles` files, reducing size by 3x.
    3.  Load all the `.smiles` files with StringZilla, split, random shuffle.
    """

    filenames = [
        "Enamine_REAL_HAC_6_21_420M_CXSMILES",
        "Enamine_REAL_HAC_22_23_471M_CXSMILES",
        "Enamine_REAL_HAC_24_394M_CXSMILES",
        "Enamine_REAL_HAC_25_557M_CXSMILES",
        "Enamine_REAL_HAC_26_833M_Part_1_CXSMILES",
        "Enamine_REAL_HAC_26_833M_Part_2_CXSMILES",
        "Enamine_REAL_HAC_27_1.1B_Part_1_CXSMILES",
        "Enamine_REAL_HAC_27_1.1B_Part_2_CXSMILES",
        "Enamine_REAL_HAC_28_1.2B_Part_1_CXSMILES",
        "Enamine_REAL_HAC_28_1.2B_Part_2_CXSMILES",
        "Enamine_REAL_HAC_29_38_988M_Part_1_CXSMILES",
        "Enamine_REAL_HAC_29_38_988M_Part_2_CXSMILES",
    ]

    # Decompress all the files
    for filename in filenames:
        has_archieve = os.path.exists(os.path.join(dir, filename + ".cxsmiles.bz2"))
        has_decompressed = os.path.exists(os.path.join(dir, filename + ".cxsmiles"))
        if has_decompressed or not has_archieve:
            continue

        print(f"Will decompress {filename}.bz2, may take a while...")
        decompress = f"pbzip2 -d data/real/{filename}.cxsmiles.bz2"
        os.system(decompress)

    # Wipe metadata
    for filename in filenames:
        has_decompressed = os.path.exists(os.path.join(dir, filename + ".cxsmiles"))
        has_wiped = os.path.exists(os.path.join(dir, filename + ".smiles"))
        if has_wiped or not has_decompressed:
            continue

        logger.info(f"Loading dataset: {filename}")
        file = File(os.path.join(dir, filename))
        file_contents: Str = file.load()
        logger.info(f"Loaded dataset: {filename}")
        file_lines: Strs = file_contents.splitlines()
        logger.info(f"Will filter {len(file_lines):,} lines in: {filename}")

        count_preserved = 0
        with open(os.path.join(dir, filename + ".smiles"), "w") as file_smiles:
            for line in file_lines[1:]:
                tab_offset = line.find("\t")
                if tab_offset > 1:
                    file_smiles.write(str(line)[:tab_offset] + "\n")
                    count_preserved += 1
                    if count_preserved % 100_000_000 == 0:
                        logger.info(f"Passed {count_preserved:,} lines from {filename}")
        logger.info(f"Kept {count_preserved:,} lines from {filename}")

    lines = Strs()
    for filename in filenames:
        filename = filename + ".smiles"
        logger.info(f"Loading dataset: {filename}")
        file = File(os.path.join(dir, filename))
        logger.info(f"Loaded dataset: {filename}")
        file_lines: Strs = file.splitlines()
        lines.extend(file_lines)
        logger.info(f"Added {len(file_lines):,} molecules, reaching {len(lines):,}")

    # Let's shuffle across all the files
    lines.shuffle(SEED)

    def extractor(row: str) -> str:
        return row

    return RawDataset(
        lines=lines,
        extractor=extractor,
    )


def export_parquet_shard(
    dataset: RawDataset,
    dir: os.PathLike,
    shard_index: int,
    shards_count: int,
    rows_per_part: int = SHARD_SIZE,
):
    os.makedirs(os.path.join(dir, "parquet"), exist_ok=True)

    try:
        lines_count = dataset.count_lines()
        first_epoch_offset = shard_index * rows_per_part
        epoch_size = shards_count * rows_per_part

        for start_row in range(first_epoch_offset, lines_count, epoch_size):
            end_row = start_row + rows_per_part

            rows_and_smiles = dataset.smiles_slice(start_row, rows_per_part)
            path_out = shard_name(dir, start_row, end_row, "parquet")
            if os.path.exists(path_out):
                continue

            try:
                dicts = []
                for _, smiles in rows_and_smiles:
                    try:
                        dicts.append({"smiles": smiles})
                    except Exception:
                        continue

                schema = pa.schema([pa.field("smiles", pa.string(), nullable=False)])
                table = pa.Table.from_pylist(dicts, schema=schema)
                write_table(table, path_out)

            except KeyboardInterrupt as e:
                raise e

            shard_description = "Molecules {:,}-{:,} / {:,}. Process # {} / {}".format(
                start_row,
                end_row,
                lines_count,
                shard_index,
                shards_count,
            )

            logger.info(f"Passed {shard_description}")

    except KeyboardInterrupt as e:
        logger.info(f"Stopping shard {shard_index} / {shards_count}")
        raise e


def export_parquet_shards(dataset: RawDataset, dir: os.PathLike, processes: int = 1):
    dataset_size = dataset.count_lines()
    logger.info(f"Loaded {dataset_size:,} lines")
    logger.info(f"First one is {str(dataset.smiles(0))}")
    logger.info(f"Mid one is {str(dataset.smiles(dataset_size // 2))}")
    logger.info(f"Last one is {str(dataset.smiles(dataset_size - 1))}")

    # Produce new fingerprints
    os.makedirs(dir, exist_ok=True)
    if processes > 1:
        process_pool = []
        for i in range(processes):
            p = Process(target=export_parquet_shard, args=(dataset, dir, i, processes))
            p.start()
            process_pool.append(p)

        for p in process_pool:
            p.join()
    else:
        export_parquet_shard(dataset, dir, 0, 1)

