import sys

from torch import nn
from copy import copy
from abc import ABCMeta, abstractmethod


class BaseModel(nn.Module, metaclass=ABCMeta):
    default_conf = {}
    required_inputs = []

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        self.required_inputs = copy(self.required_inputs)
        self._init(conf)
        sys.stdout.flush()

    def forward(self, data):
        for key in self.required_inputs:
            assert key in data, "Missing key {} in data".format(key)
        return self._forward(data)

    @abstractmethod
    def _init(self, conf):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, data):
        raise NotImplementedError
