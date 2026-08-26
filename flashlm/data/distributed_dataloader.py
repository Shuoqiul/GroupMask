import torch
from itertools import chain
from datasets.distributed import split_dataset_by_node
from datasets import IterableDataset
from torch.utils.data import DataLoader
from .distributed_iterable import DistributedFolderIterable


def is_distirbuted_dataset(iterable):
    if hasattr(iterable, '_ex_iterable'):
        if isinstance(iterable._ex_iterable, DistributedFolderIterable):
            return True
        else:
            return is_distirbuted_dataset(iterable._ex_iterable)
    elif hasattr(iterable, 'ex_iterable'):
        if isinstance(iterable.ex_iterable, DistributedFolderIterable):
            return True
        else:
            return is_distirbuted_dataset(iterable.ex_iterable)
    elif hasattr(iterable, 'ex_iterables'):
        for i in iterable.ex_iterables:
            if isinstance(i, DistributedFolderIterable):
                return True
            else:
                return is_distirbuted_dataset(i)
    else:
        return False


def dataloader_creator(dataset, tokenizer, batch_size, block_size, rank, world_size, 
                        num_workers=1, cycling=False, shuffle_seed=1, shuffle_buffer=0, sample_group_size=50, ignored_token=None):
    
    print(type(dataset))
    assert type(dataset)==IterableDataset, "The input dataset must be type of IterableDataset"

    torch.multiprocessing.set_sharing_strategy("file_system")

    if is_distirbuted_dataset(dataset):
        # The dataset with no_split attribute can not be split again here.
        print('This dataset was already initialized distributedly')
        world_size = 0 # turn off split_dataset_by_node
        shuffle_seed += rank # only the distributed dataset can use different shuffle seed to shuffle the files of each thread

    if shuffle_buffer>0:
        # Recommand shuffle_buffer=10 for the best throughput (for text)
        dataset = dataset.shuffle(seed=shuffle_seed, buffer_size=shuffle_buffer)

    if world_size>1:
        dataset = split_dataset_by_node(dataset, rank, world_size)

    block_size = block_size+1 # for the shift of target
    if ignored_token is None:
        ignored_token = tokenizer.pad_id

    def pad_list(x):
        if len(x)<block_size:
            # print('!!! WARNING: short example detected, please check block_size or sample_group_size:', len(x))
            x += [ignored_token] * (block_size - len(x))
        return x

    def group_tokens(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [pad_list(t[i : i + block_size]) for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    dataset = dataset.map(lambda x: {'input_ids': tokenizer.encode(x["text"])}, remove_columns='text')
    dataset = dataset.map(group_tokens, batched=True, batch_size=sample_group_size)
    dataset = dataset.map(lambda x: {'input_ids': torch.LongTensor(x['input_ids'])})
    dataset = dataset.map(lambda x: {'labels':x['input_ids'][1:], 'input_ids': x['input_ids'][:-1]})

    def collate_fn(batch):
        # This is to avoid some weird errors happened with the default collate_fn
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch])
        }

    def set_worker_sharing_strategy(worker_id: int) -> None:
        # This is to avoid the error "Too many open files. Communication with the workers is no longer possible."
        torch.multiprocessing.set_sharing_strategy("file_system")

    def cycle(dataloader):
        # This is to avoid the /dev/shm exploding issue caused by itertools.cycle
        # See this thread for the detailed discussion about /dev/shm: https://github.com/pytorch/pytorch/issues/13246
        dataloader_iterator = iter(dataloader)
        while True:
            try:
                yield next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(dataloader)
                yield next(dataloader_iterator)

    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False, drop_last=True, worker_init_fn=set_worker_sharing_strategy)

    if cycling:
        # create an infinity cycling iterator
        # This is useful when each shard have very different lengths,
        # which can cause some workers/threads to terminate much earilier than others.
        dataloader = cycle(dataloader)
    return dataloader
