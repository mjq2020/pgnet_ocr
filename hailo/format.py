from hailo_platform.pyhailort._pyhailort import StreamInfo


class NodeInfo:
    __solts__ = ["_" + attr for attr in dir(StreamInfo) if "__" not in attr]

    def __init__(self, node_info):
        """
        Initialize a NodeInfo object from a StreamInfo object.

        :param node_info: StreamInfo object
        :type node_info: StreamInfo
        """
        for attr in self.__solts__:
            if "_shape" in attr:
                setattr(self, attr, ["-1"] + list(node_info.shape)[::-1])
                continue
            try:
                setattr(self, attr, getattr(node_info, attr[1:]))
            except:
                continue

    @property
    def name(self):
        """
        The name of the node.

        :rtype: str
        """
        return self._name

    @property
    def shape(self):
        """
        The shape of the node in the format (batch, height, width, channels).

        :rtype: list[int]
        """
        return self._shape

    @property
    def direction(self):
        """
        The direction of the node. Possible values are 'input' or 'output'.

        :rtype: str
        """
        return self._direction

    @property
    def data_bytes(self):
        """
        The data bytes of the node.

        :rtype: int
        """
        return self._data_bytes

    def __dir__(self):
        """
        List of valid attributes for the object.

        This method returns a list of valid attributes for the object. The list
        includes all public attributes (i.e. those that do not start with an
        underscore) and all special Python attributes (i.e. those that start with
        '__' and end with '__').

        :rtype: list[str]
        """
        return [
            attr
            for attr in super().__dir__()
            if not attr.startswith("_") or attr.startswith("__")
        ]
