"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/misc.py
Mostly copy-paste from torchvision references.
"""

import time
import pickle
import datetime
from collections import defaultdict, deque
from typing import Dict

import torch
import torch.distributed as tdist

from pyasd.misc.dist_utils import is_dist_available_and_initialized, get_world_size

MB = 1024.0 * 1024.0

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        """Create an empty tracker.

        Args:
            window_size: Number of recent values kept for windowed statistics.
            fmt: Format string used by ``__str__``. May reference ``median``,
                ``avg``, ``global_avg``, ``max`` and ``value``.
        """
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        """Record a new value.

        Args:
            value: Value to add (e.g. a batch-mean loss).
            n: Weight of the value in the global average (e.g. batch size).
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Sum ``count`` and ``total`` across all processes.

        Warning: does not synchronize the deque!
        """
        if not is_dist_available_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """float: Median of the values in the window."""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """float: Mean of the values in the window."""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """float: Weighted mean of all values recorded so far."""
        return self.total / self.count

    @property
    def max(self):
        """Maximum of the values in the window."""
        return max(self.deque)

    @property
    def value(self):
        """Most recently recorded value."""
        return self.deque[-1]

    def __str__(self):
        """Format the statistics with ``self.fmt``.

        Returns:
            str: The formatted statistics.
        """
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    tdist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    tdist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True) -> Dict[str, torch.Tensor]:
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    """Collect named ``SmoothedValue`` meters and log progress over an iterable."""
    def __init__(self, delimiter="\t"):
        """Create a logger with no meters.

        Args:
            delimiter: String placed between meters in ``__str__``.
        """
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        """Add values to the named meters, creating meters as needed.

        Args:
            **kwargs: Meter name to value. Values must be numbers or one-element tensors.
        """
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """Return the meter named ``attr``, allowing ``logger.loss`` style access.

        Args:
            attr: Attribute name.

        Returns:
            SmoothedValue: The matching meter (or instance attribute).

        Raises:
            AttributeError: If no meter or attribute has that name.
        """
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        """Format all meters as ``name: value`` pairs, joined by the delimiter.

        Returns:
            str: The formatted meters.
        """
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                f"\033[92m{name}\033[0m: {str(meter)}"
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """Synchronize the global statistics of every meter across processes."""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """Register a meter under the given name.

        Args:
            name: Meter name.
            meter: ``SmoothedValue`` instance, e.g. with a custom window or format.
        """
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """Yield items from ``iterable`` and print progress every ``print_freq`` steps.

        Each progress line shows ETA, iteration and data-loading times, peak CUDA
        memory and the current meters. A timing summary is printed at the end.

        Args:
            iterable: Sized iterable to loop over (e.g. a DataLoader).
            print_freq: Print every this many iterations (and on the last one).
            header: Optional prefix for each progress line.

        Yields:
            The items of ``iterable``.
        """
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        
        # Determine padding for batch counter
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        
        # Fixed f-string syntax for nested formatting
        base_msg = [
            f"\033[1m{header}\033[0m [{{0{space_fmt}}}/{len(iterable)}]",
            "  \033[94mStats:\033[0m eta: {eta}  time: {time}  data: {data}  mem: {memory:.0f}MB  rsvd_mem: {rsvd_memory:.0f}MB",
            "  \033[92mMetrics:\033[0m {meters}"
        ]
        log_msg = "\n".join(base_msg)

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                memory = torch.cuda.max_memory_allocated() / MB if torch.cuda.is_available() else 0
                memory_reserved = torch.cuda.max_memory_reserved() / MB if torch.cuda.is_available() else 0
                # Correct .format call
                print(log_msg.format(i, eta=eta_string, meters=str(self),
                                   time=str(iter_time), data=str(data_time),
                                   memory=memory, rsvd_memory=memory_reserved))
            
            i += 1
            end = time.time()

        # --- EPOCH SUMMARY ---
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\n\033[1m{header} COMPLETED\033[0m")
        print(f"  Total Duration: {total_time_str}")
        print(f"  Average Speed:  {total_time / len(iterable):.4f} s/it")
        print("-" * 60)
