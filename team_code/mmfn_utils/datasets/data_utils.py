from torch.nn.utils.rnn import pad_sequence
import torch
import re
import collections
from torch._six import string_classes
import warnings

warnings.filterwarnings("ignore")
def collate_single_cpu(batch, now_key=""):
    
    r"""Puts each data field into a tensor with outer dimension batch size"""
    collate_single_cpu_err_msg_format = (
    "collate_single_cpu: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
    
    elem = batch[0]
    elem_type = type(elem)

    if now_key == "vectormaps" and isinstance(elem, torch.Tensor):
        lane_nums = []
        for lane in batch:
            lane_nums.append(lane.shape[0])
        out = pad_sequence(batch, batch_first=True)
        lane_nums = torch.tensor(lane_nums)
        return [out,lane_nums, int(torch.max(lane_nums).item())]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)

    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            np_str_obj_array_pattern = re.compile(r'[SaUO]')
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_single_cpu_err_msg_format.format(elem.dtype))
            return collate_single_cpu([torch.as_tensor(b) for b in batch], now_key)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_single_cpu([d[key] for d in batch], key) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_single_cpu(samples, now_key) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            print(elem)
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_single_cpu(samples, now_key) for samples in transposed]

    raise TypeError(collate_single_cpu_err_msg_format.format(elem_type))



def collate_single_cpu_fake(batch, now_key=""):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    collate_single_cpu_err_msg_format = (
    "collate_single_cpu: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
    
    elem = batch[0]
    elem_type = type(elem)
    if now_key == "vectormaps" and isinstance(elem, torch.Tensor):
        return torch.zeros(len(batch[0]), 5, 5)
    
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            np_str_obj_array_pattern = re.compile(r'[SaUO]')
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(collate_single_cpu_err_msg_format.format(elem.dtype))
            return collate_single_cpu_fake([torch.as_tensor(b) for b in batch], now_key)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)

    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: collate_single_cpu_fake([d[key] for d in batch], key) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(collate_single_cpu_fake(samples, now_key) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            print(elem)
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [collate_single_cpu_fake(samples, now_key) for samples in transposed]

    raise TypeError(collate_single_cpu_err_msg_format.format(elem_type))
