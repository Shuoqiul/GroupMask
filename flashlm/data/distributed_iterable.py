import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union
import datasets
from datasets import IterableDataset, load_dataset
from datasets.iterable_dataset import VerticallyConcatenatedMultiSourcesExamplesIterable
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from itertools import cycle, islice
import random


def get_files_by_extensions(directory, extensions):
    file_list = []
    for folder, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_list.append(os.path.join(folder, file))
    return file_list


class DistributedFolderIterable(VerticallyConcatenatedMultiSourcesExamplesIterable):
    def __init__(self, data_folder: str, world_size: int, rank: int, num_workers: Optional[int]=1, source_column_name: Optional[str]=None, target_column_name: Optional[str]='text'):
        assert datasets.__version__>='2.14.4', 'datasets lib is too old. Please do: pip install datasets --upgrade'
        self.data_folder = data_folder
        self.step = world_size
        self.offset = rank
        self.file_mapping = {}
        files = get_files_by_extensions(data_folder, ['json', 'jsonl', 'parquet', 'zst'])
        ex_iterables = []
        file_list = []
        for file in tqdm(files):
            file_list.append(file)
            if len(file_list)==num_workers:
                ds = load_dataset(data_folder, data_files=file_list, streaming=True)
                ds = ds['train']
                if source_column_name is not None:
                    ds = ds.rename_column(source_column_name, target_column_name)
                ds = ds.select_columns(target_column_name)
                self.file_mapping[ds._ex_iterable] = file_list
                file_list = []
        if len(file_list)>0:
            print(f"[WARNING] These files are discarded because the total number of file`s ({len(files)}) is not divisible by num_workers ({num_workers}):", file_list)
        super().__init__(list(self.file_mapping.keys())) # setup self.ex_iterables
    
    def __iter__(self):
        for ex_iterable in self.ex_iterables:
            ex_iterator = iter(ex_iterable)
            while True:
                batch = list(islice(ex_iterator, self.step))
                if len(batch) > self.offset:
                    yield batch[self.offset]
                else:
                    break

    def _iter_arrow(self):
        raise NotImplementedError(f"{type(self)} doesn't implement _iter_arrow yet")

    def shuffle_data_sources(self, generator: np.random.Generator):
        rng = deepcopy(generator)
        rng.shuffle(self.ex_iterables)
        return self

    def shard_data_sources(self, worker_id: int, num_workers: int):
        ex_iterables = [iterable.shard_data_sources(worker_id, num_workers) for iterable in self.ex_iterables]
        self.ex_iterables = ex_iterables
        return self

    def shuffle(self, rnd_seed):
        random.seed(rnd_seed)
        random.shuffle(self.ex_iterables)


def DistributedFolderIterableDataset(*args):
    it = DistributedFolderIterable(*args)
    ds = IterableDataset(it)
    return ds