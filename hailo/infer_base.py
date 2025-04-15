from typing import Any
from abc import abstractmethod, ABCMeta


class BaseHailoInference(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def run(self, output_names, input_feed, run_options=None):
        """
        computer thr  predictions

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: options for the run

        return: list of results
        """
        raise NotImplementedError

    @abstractmethod
    def get_inputs(self) -> Any:
        """return the inputs metadata"""
        raise NotImplementedError

    def get_outputs(self):
        raise NotImplementedError
