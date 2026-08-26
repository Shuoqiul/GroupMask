import datasets
import os
from .distributed_iterable import DistributedFolderIterableDataset

# The datasets here shuffle the order of files for each thread.
# The datasets in hugginface_access.py can not have different file shufflings for different threads

class dataset_info:
    def __init__(self, name, path, token, source_column_name=None, target_column_name='text'):
        self.name = name
        self.path = path
        self.token = token # number of token in trillion, ex: 0.3 T
        self.source_column_name = source_column_name # if a dataset does not use 'text' in their json, specify the name here.
        self.target_column_name = target_column_name # we always expect 'text' in our cases
    def __str__(self):
        return f"name:{self.name}, path:{self.path}, #token(T):{self.token}"

def _data_path(*parts):
    return os.path.join(os.environ.get("FLASHLM_DATA_ROOT", "datasets"), *parts)

dataset_dict = {
    'pile': dataset_info('pile', _data_path('the_pile_deduplicated', 'data'), 0.23),
    'korean': dataset_info('korean', _data_path('corpus_pt_korean_split'), 0.3),
    'slimpajama': dataset_info('slimpajama', _data_path('SlimPajama-627B', 'train'), 0.6),
    'refinedweb': dataset_info('refinedweb', _data_path('falcon-refinedweb', 'data'), 0.6, source_column_name='content'),
}

def distributed_pile(world_size, rank, num_workers):
    ds = DistributedFolderIterableDataset(dataset_dict['pile'].path,world_size, rank, num_workers)
    return ds

def distributed_korean(world_size, rank, num_workers):
    ds = DistributedFolderIterableDataset(dataset_dict['korean'].path,world_size, rank, num_workers)
    return ds

def distributed_slimpajama(world_size, rank, num_workers):
    ds = DistributedFolderIterableDataset(dataset_dict['slimpajama'].path,world_size, rank, num_workers)
    return ds

def distributed_refinedweb(world_size, rank, num_workers):
    ds = DistributedFolderIterableDataset(dataset_dict['refinedweb'].path,world_size, rank, num_workers, 'content')
    return ds

def distributed_mixed_datasets(world_size, rank, num_workers, dataset_name_list, prob_list):
    assert sum(prob_list)==1
    assert len(dataset_name_list)==len(prob_list)
    ds_list = []
    for name in dataset_name_list:
        ds = DistributedFolderIterableDataset(dataset_dict[name].path, world_size, rank, num_workers, 
                dataset_dict[name].source_column_name, dataset_dict[name].target_column_name)
        ds_list.append(ds)
    merged_ds = datasets.interleave_datasets(ds_list, prob_list, stopping_strategy='all_exhausted')
    return merged_ds
