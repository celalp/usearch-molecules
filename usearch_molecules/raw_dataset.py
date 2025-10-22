import os
from multiprocessing import Process

from math import ceil
import pyarrow as pa
from stringzilla import File, Strs

from usearch_molecules.utils import write_table, shard_name, augment_with_rdkit

from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)

class RawDataset:
    """raw dataset takes a list of files that contains smiles, splits them into shards and
    and calculates fingerprints. This is the first step to prepare the dataset
    This will then get passed to fingerprints.FingerprintedDataset to searched. This part
    needs to be done only once"""
    def __init__(self, files, data_dir, extractor=None):
        """
        :param files: list of files to read, these files contain the smiles strings for the molecules
        :param output_dir: directory to write output to
        :param extractor: a callable, if there are more things in the file per line use this function to extract it from the line
        """
        self.files = files
        self.extractor = extractor
        self.lines=None
        self.num_lines=None
        self.data_dir=data_dir
        self.shards=[]
        self.smiles=[]

    def prep_shards(self):
        """just collect shards and memory map them see stringzilla for an explanation"""
        lines=Strs()
        for file in self.files:
            f=File(file)
            if self.extractor is None:
                lines.extend(f.splitlines())
            else:
                lines.extend(self.extractor(f.splitlines()))
        self.lines=lines
        self.num_lines=len(lines)

    def export_shards(self, shard_size=1_000_000, start_row=0):
        """
        split the datasae into n shards of specific shard size the size is the number of smiles
        that are going to be in each shard
        :param shard_size: number of smiles
        :param start_row: if there is header of somesuch
        :return: names of files generated for smiles and shards, initially these are identical but
        the shards will get augmented by rdkit
        """
        os.makedirs(os.path.join(self.data_dir, "parquet"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "smiles"), exist_ok=True)
        num_shards = ceil(len(self.lines) / shard_size)
        shards = []
        smiles = []
        for i in range(num_shards):
            dicts = []
            start = (i + start_row) * shard_size
            end = start + shard_size
            path_out = shard_name(self.data_dir, start, end, "parquet")
            smiles_out = shard_name(self.data_dir, start, end, "smiles")
            shards.append(path_out)
            smiles.append(smiles_out)
            rows = [{"smiles": str(row)} for row in self.lines[start:end]]
            dicts.extend(rows)
            schema = pa.schema([pa.field("smiles", pa.string(), nullable=False)])
            table = pa.Table.from_pylist(dicts, schema=schema)
            write_table(table, path_out)
            with open(smiles_out, "w") as f:
                for smile in self.lines[start:end]:
                    f.write(str(smile) + "\n")

        self.shards=shards
        self.smiles=smiles



def augment_parquet_shard(shard_subset, augmentation):
    """
    this may seem convoluted but the augmentation is an expensive process and I need to have an out-of-scope function
    so I can call mutiprocessing w/o worrying about changing things in self. This is mostly a me limitation as i do not know
    how to make this part of the class
    :param shard_subset: shard subset, multiprocessing will deal with it
    :param augmentation: augmentation callable
    :return:
    """
    for file in shard_subset:
        augmentation(os.path.join(file))


def augment_parquet_shards(dataset, augmentation=augment_with_rdkit, processes=1):
    """This is where we call multiprocessing"""
    filenames = sorted(dataset.shards)
    shard_chunks = [
        filenames[i::processes] for i in range(processes)
    ]

    if processes > 1:
        process_pool = []
        for i in range(processes):
            p = Process(
                target=augment_parquet_shard,
                args=(shard_chunks[i], augmentation),
            )
            p.start()
            process_pool.append(p)

        for p in process_pool:
            p.join()
    else:
        augment_parquet_shard(filenames, augmentation=augmentation)

