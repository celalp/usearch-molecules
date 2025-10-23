import os

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetMorganFeatureAtomInvGen

import pyarrow.parquet as pq
import pyarrow as pa

fcfp_invariants = GetMorganFeatureAtomInvGen()
fcfp_generator = GetMorganGenerator(radius=2, fpSize=2048, atomInvariantsGenerator=fcfp_invariants)
ecfp_generator = GetMorganGenerator(radius=2, fpSize=2048)

def molecule_to_maccs(x):
    return rdMolDescriptors.GetMACCSKeysFingerprint(x)


def molecule_to_ecfp4(x):
    return list(ecfp_generator.GetFingerprint(x))


def molecule_to_fcfp4(x):
    return list(fcfp_generator.GetFingerprint(x))


def smiles_to_maccs_ecfp4_fcfp4(smiles: str):
    """Uses RDKit to simultaneously compute MACCS, ECFP4, and FCFP4 representations."""

    molecule = Chem.MolFromSmiles(smiles)
    return (
        np.packbits(molecule_to_maccs(molecule)),
        np.packbits(molecule_to_ecfp4(molecule)),
        np.packbits(molecule_to_fcfp4(molecule)),
    )


def shard_name(dir, from_index, to_index, kind):
    """create file name to indicate the star and end number for the shards"""
    return os.path.join(dir, kind, f"{from_index}-{to_index}.{kind}")

def write_table(table, path_out):
    """write parquet table"""
    return pq.write_table(
        table,
        path_out,
        write_statistics=False,
        store_schema=True,
        use_dictionary=False,
        )

def augment_with_rdkit(parquet_path: os.PathLike):
    """
    augment parquet file with all the fingerprints, uses functions defined above
    :param parquet_path: path of the file
    :return: nothing just adds data to file i.e. the fingerprints
    """
    meta = pq.read_metadata(parquet_path)
    column_names = meta.schema.names

    table: pa.Table = pq.read_table(parquet_path)
    maccs_list = []
    ecfp4_list = []
    fcfp4_list = []
    for smiles in table["smiles"]:
        try:
            fingers = smiles_to_maccs_ecfp4_fcfp4(str(smiles))
            maccs_list.append(fingers[0].tobytes())
            ecfp4_list.append(fingers[1].tobytes())
            fcfp4_list.append(fingers[2].tobytes())
        except Exception:
            maccs_list.append(bytes(bytearray(21)))
            ecfp4_list.append(bytes(bytearray(256)))
            fcfp4_list.append(bytes(bytearray(256)))

    maccs_list = pa.array(maccs_list, pa.binary(21))
    ecfp4_list = pa.array(ecfp4_list, pa.binary(256)) #because 256*8 is 2048 (bytes to bits)
    fcfp4_list = pa.array(fcfp4_list, pa.binary(256))
    maccs_field = pa.field("maccs", pa.binary(21), nullable=False)
    ecfp4_field = pa.field("ecfp4", pa.binary(256), nullable=False)
    fcfp4_field = pa.field("fcfp4", pa.binary(256), nullable=False)

    table = table.append_column(maccs_field, maccs_list)
    table = table.append_column(ecfp4_field, ecfp4_list)
    table = table.append_column(fcfp4_field, fcfp4_list)
    write_table(table, parquet_path)


