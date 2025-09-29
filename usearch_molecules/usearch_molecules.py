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
    """
    simple dataclass that loads all the smiles, how you process the smiles is up to you, all you need
    is a just a list of simles.
    """
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

def export_parquet_shard(
    dataset: RawDataset,
    dir: os.PathLike,
    shard_index: int,
    shards_count: int,
    rows_per_part: int = SHARD_SIZE,
):
    """
    Export a shard of the dataset to parquet files.
    :param dataset: this is the smiles dataset from aboe, you can shard it as you like
    current default is 1M
    :param dir: where to put the shards
    :param shard_index:
    :param shards_count:
    :param rows_per_part:
    :return:
    """
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
