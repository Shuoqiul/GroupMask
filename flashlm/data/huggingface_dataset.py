import datasets
from datasets import load_dataset, load_from_disk, concatenate_datasets, interleave_datasets
import os, errno
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Type, Union

FLASHLM_DATA_ROOT_ENV = "FLASHLM_DATA_ROOT"
FLASHLM_HF_CACHE_ENV = "FLASHLM_HF_CACHE_DIR"

def _data_root() -> str:
    return os.environ.get(FLASHLM_DATA_ROOT_ENV, "datasets")

def _data_path(*parts: str) -> str:
    return os.path.join(_data_root(), *parts)

def _hf_cache_dir() -> Optional[str]:
    return os.environ.get(FLASHLM_HF_CACHE_ENV)

# <<How to save huggingface datasets>>
# Step 1: Use huggingface to download a dataset. 
#   Ex: 
#   from datasets import load_dataset
#   dataset = load_dataset("EleutherAI/the_pile_deduplicated")
# Step 2: Find the cache and upload it to object storage.
#   Ex: scp -r ~/.cache/huggingface/datasets/EleutherAI___parquet <DATASET_STORAGE_PATH>
# Step 3: Create the function that overlay the cache stored in object storage.
#   Ex: load_hf_dataset_pile_dedup()

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
            # Change the symlink will cause race condition - so avoid to do it
            # os.remove(link_name)
            # os.symlink(target, link_name)
        else:
            raise e

def load_hf_dataset(hf_path: str, data_cache_dir: Optional[str]=None, default_cache_dir: Optional[str]='~/.cache/huggingface/datasets', **kwargs):
    # This function load a specific cached hf dataset from a different storage
    # This function is useful to load the cached hf dataset from a read-only storage like object storage
    # Huggingface datasets requires the dataset cache folder to be writable, but the object storage is read-only. So we created this function.
    # Example data_cache_dir = os.path.join(os.environ["FLASHLM_DATA_ROOT"], "EleutherAI___the_pile")
    if data_cache_dir is not None:
        sym_name = os.path.basename(data_cache_dir)
        os.makedirs(os.path.expanduser(default_cache_dir), exist_ok=True)
        symlink_force(data_cache_dir, os.path.join(os.path.expanduser(default_cache_dir),sym_name))
    return load_dataset(hf_path, **kwargs)

def load_hf_dataset_pile_dedup(split, n_shards=None):
    # Off-the-shelf function to get the HF pile deduplicated dataset
    if split=='train':
        ds = load_dataset(_data_path("the_pile_deduplicated"), streaming=True)
        ds = ds['train']
        # ds = load_from_disk(_data_path("pile", "train"))
        ds = ds.select_columns("text")
        # ds = ds.to_iterable_dataset(num_shards=n_shards)
    if split=='validation':
        ds = load_from_disk(_data_path("pile", "val"))
        #ds = ds.select_columns("text")
        ds = ds.to_iterable_dataset(num_shards=n_shards)
    if split=='test':
        ds = load_from_disk(_data_path("pile", "test"))
        ds = ds.select_columns("text")
        ds = ds.to_iterable_dataset(num_shards=n_shards)
    return ds

def load_hf_dataset_slimpajama(split=None, n_shards=None):
    ds = load_dataset(_data_path("SlimPajama-627B", split), streaming=True)
    ds = ds['train']
    ds = ds.select_columns("text")
    return ds

def load_hf_dataset_korean(split=None, n_shards=None):
    ds = load_dataset(_data_path("corpus_pt_korean_split"), streaming=True)
    ds = ds['train']
    ds = ds.select_columns("text")
    return ds

def load_hf_dataset_refinedweb(split=None, n_shards=None):
    ds = load_dataset(_data_path("falcon-refinedweb"), streaming=True)
    ds = ds['train']
    ds = ds.rename_column('content', 'text')
    ds = ds.select_columns("text")
    return ds

def load_hf_dataset_minipile(split='train', n_shards=None):
    # SJC MLP group dataset
    # train split number of token: 1690681344 (1.7B or ~0.6% of original Pile of ~0.7% or pile_dedup)
    ds = load_dataset(_data_path("minipile"))
    ds = ds[split]
    ds = ds.to_iterable_dataset(num_shards=n_shards)
    return ds

def load_hf_dataset_wiki(split='train', n_shards=None, seed=777):
    # Off-the-shelf function to get the HF pile deduplicated dataset
    # if split=='train':
    #     # ds = load_hf_dataset("EleutherAI/the_pile_deduplicated", data_cache_dir=_data_path("EleutherAI___parquet"))
    #     ds = load_from_disk(_data_path("pile", "train"))
    # if split=='validation':
    #     ds = load_from_disk(_data_path("pile", "val"))
    # if split=='test':
    #     ds = load_from_disk(_data_path("pile", "test"))
    # wiki = load_dataset("wikipedia", "20220301.en", cache_dir=_hf_cache_dir(), split="train")
    # bookcorpus = load_dataset("bookcorpus", cache_dir=_hf_cache_dir(), split="train")
    # wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column
    cache_dir = _hf_cache_dir()
    if cache_dir is None and os.path.isdir("./wikitext"):
        cache_dir = os.path.abspath("./wikitext")

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")
    # ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    

    # ds = load_from_disk("wikitext")
    if hasattr(ds, "keys") and "train" in ds:
        wikitext = ds["train"]
    else:
        wikitext = ds
    
    wikitext = wikitext.remove_columns([col for col in wikitext.column_names if col != "text"])
    raw_datasets = wikitext
    raw_datasets.shuffle(seed=seed)
    # if return_raw:
    #     return raw_datasets
    # else:
    return raw_datasets.to_iterable_dataset(num_shards=n_shards)


# use this dataset for training 

def load_hf_dataset_alpaca(split='train', n_shards=None, seed=777):
    ds = load_dataset("tatsu-lab/alpaca", split=split)
    ds = ds.select_columns("text").shuffle(seed=seed)
    return ds.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_orca_dpo(n_shards=None, seed=777):
    ds = load_from_disk(_data_path("orca_dpo_pairs.hf"))
    ds = ds.select_columns("text").shuffle(seed=seed)
    return ds.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_wizardlMv2(n_shards=None, seed=777):
    ds = load_from_disk(_data_path("WizardLM_evol_instruct_V2_196k.hf"))
    ds = ds.select_columns("text").shuffle(seed=seed)
    return ds.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_new_mixed(n_shards=None, seed=42, splits=[0.25,0.20,0.15,0.25,0.05,0.05,0.05]):
    cache_dir = _hf_cache_dir()
    # ds0 = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")
    # ds0 = ds0.select_columns("text").shuffle(seed=seed)

    ds0 = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    ds0 = ds0.select_columns("text").shuffle(seed=seed)

    ds1 = load_from_disk(_data_path("CodeAlpaca-20k.hf"))
    ds1 = ds1.select_columns("text").shuffle(seed=seed)

    ds2 = load_from_disk(_data_path("WizardLM_evol_instruct_V2_196k.hf"))
    ds2 = ds2.select_columns("text").shuffle(seed=seed)

    ds3 = load_from_disk(_data_path("hellaswag.hf"))
    ds3 = ds3.select_columns("text").shuffle(seed=seed)

    ds4 = load_from_disk(_data_path("arc_c.hf"))
    ds4 = ds4.select_columns("text").shuffle(seed=seed)

    ds5 = load_from_disk(_data_path("arc_e.hf"))
    ds5 = ds5.select_columns("text").shuffle(seed=seed)

    ds6 = load_from_disk(_data_path("winogrande.hf"))
    ds6 = ds6.select_columns("text").shuffle(seed=seed)
    # ds6 = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")
    # ds6 = ds6.select_columns("text").shuffle(seed=seed)


    dsc = interleave_datasets([ds0, ds1, ds2, ds3, ds4, ds5, ds6], probabilities=splits, seed=seed)

    return dsc.to_iterable_dataset(num_shards=n_shards)

def load_hf_dataset_mixed(n_shards=None, seed=777, splits=[0.25,0.25,0.25,0.25]):
    cache_dir = _hf_cache_dir()

    ds1 = load_dataset("tatsu-lab/alpaca", split="train", cache_dir=cache_dir)
    ds1 = ds1.select_columns("text").shuffle(seed=seed)

    ds2 = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir, split="train")
    ds2 = ds2.select_columns("text").shuffle(seed=seed)

    # ds3 = load_dataset("JeanKaddour/minipile", cache_dir=cache_dir, split="train")
    # ds3 = ds3.select_columns("text").shuffle(seed=seed)
    ds3 = load_from_disk(_data_path("CodeAlpaca-20k.hf"))
    ds3 = ds3.select_columns("text").shuffle(seed=seed)
    # ds3 = load_dataset("ssbuild/vicuna", cache_dir=cache_dir, split="train")
    # ds3 = ds3.select_columns("text").shuffle(seed=seed)

    ds4 = load_from_disk(_data_path("WizardLM_evol_instruct_V2_196k.hf"))
    ds4 = ds4.select_columns("text").shuffle(seed=seed)

    dsc = interleave_datasets([ds1, ds2, ds3, ds4], probabilities=splits, seed=seed)


    # dsc = concatenate_datasets(
    #     [
    #     #ds1.select(range(int(splits[0]*len(ds1)))),
    #     ds2.select(range(int(splits[1]*len(ds1)))),
    #     ds3.select(range(int(splits[2]*len(ds1))))
    #     ]).shuffle(seed=seed)
    
    return dsc.to_iterable_dataset(num_shards=n_shards)
