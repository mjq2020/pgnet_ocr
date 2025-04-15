from typing import List
import os

os.environ["HAILO_MONITOR"] = "1"

import numpy as np
from hailo.infer_base import BaseHailoInference
from hailo_platform import (
    VDevice,
    HEF,
    ConfigureParams,
    HailoStreamInterface,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
    InferVStreams,
)
from hailo.format import NodeInfo


class HailoRTInference(BaseHailoInference):
    def __init__(self, hef_file=None, arch="hailo8"):
        """
        Initialize HailoRTInference instance.

        Parameters
        ----------
        hef_file : str, optional
            Path to the Hailo Engine File. If not provided, it will be generated from
            the onnx model (if provided) or the model in the Hailo SDK.
        arch : str, optional
            Target architecture name, default is "hailo8"

        Notes
        -----
        After initialization, the instance is ready to be used for inference.
        The instance is also a context manager, so it can be used in a with statement.
        When the instance is used in a with statement, the instance will be activated
        and deactivated automatically.
        """
        super().__init__()

        self.target = VDevice()

        self.hef_model = HEF(hef_file)

        self.config_params = ConfigureParams.create_from_hef(
            hef=self.hef_model, interface=HailoStreamInterface.PCIe
        )
        self.network_groups = self.target.configure(self.hef_model, self.config_params)

        self.network_group = self.network_groups[0]

        self.network_group_param = self.network_group.create_params()

        self.input_vstreams_params = InputVStreamParams.make(self.network_group)
        self.output_vstreams_params = OutputVStreamParams.make(self.network_group)

        self.infer = InferVStreams(
            self.network_group, self.input_vstreams_params, self.output_vstreams_params
        )

        self.activater = self.network_group.activate(self.network_group_param)

        self.__enter__()

    def __enter__(self) -> InferVStreams:
        """
        Activate the Hailo device and setup the Inference context.

        Returns
        -------
        InferVStreams
            The Inference context.
        """
        self.infer_ctx = self.infer.__enter__()
        self.activater.__enter__()
        return self.infer_ctx

    def __exit__(self) -> False:
        """
        Deactivate the Hailo device and clean up the Inference context.

        Notes
        -----
        This method is automatically called when exiting the with statement.

        Returns
        -------
        bool
            Always returns False.
        """
        self.infer_ctx.__exit__()
        self.activater.__exit__()
        return False

    def run(self, output_names, input_feed, run_options=None) -> dict:
        """
        Run inference on the input feed.

        Parameters
        ----------
        output_names : list of str
            List of output node names.
        input_feed : dict
            Input feed dictionary where key is the node name and value is the input data.
        run_options : dict, optional
            Additional options for the inference run.

        Returns
        -------
        dict
            Inference results, where key is the output node name and value is the output data.
        """
        infer_results = self.infer_ctx.infer(input_feed)
        return infer_results

    def get_inputs(self) -> List[NodeInfo]:
        """
        Get input node information.

        Returns
        -------
        List[NodeInfo]
            List of input node information.
        """
        input_infos = self.hef_model.get_input_stream_infos()
        res = []
        for info in input_infos:
            node = NodeInfo(info)
            res.append(node)
        return res

    def get_outputs(self) -> List[NodeInfo]:
        """
        Get output node information.

        Returns
        -------
        List[NodeInfo]
            List of output node information.
        """
        output_infos = self.hef_model.get_output_stream_infos()
        res = []
        for info in output_infos:
            node = NodeInfo(info)
            res.append(node)
        return res

    def __del__(self):
        """
        Destructor method. Calls the `__exit__` method to release resources.
        """
        self.__exit__()


if __name__ == "__main__":
    inference = HailoRTInference(
        hef_file="/home/hk/mjq/hailo_tools/onnx_pgnet_640x640_modify.hef"
    )
    inputs = inference.get_inputs()
    input_data = np.random.randn(1, 3, 640, 640).astype(np.uint8)
    outputs = inference.run(inputs[0].name, input_data)
    for key, output in outputs.items():
        print(key, output.shape)
