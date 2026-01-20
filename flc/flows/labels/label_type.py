from enum import Enum

from flc.flows.labels.flow_labels import FlowLabel
from flc.flows.labels.group_labels import FlowGroupLabel


class FlowLabelType(str, Enum):
    FLOW = "flow"
    FLOW_GROUP = "flow-group"

    @staticmethod
    def from_str(s: str) -> "FlowLabelType":
        if s == FlowLabelType.FLOW.value:
            return FlowLabelType.FLOW
        elif s == FlowLabelType.FLOW_GROUP.value:
            return FlowLabelType.FLOW_GROUP

        raise ValueError(f"Unknown flow label type for string: {s}")

    @staticmethod
    def values() -> list[str]:
        return [lt.value for lt in FlowLabelType]

    @property
    def num_classes(self):
        if self is FlowLabelType.FLOW:
            return len(FlowLabel)
        elif self is FlowLabelType.FLOW_GROUP:
            return len(FlowGroupLabel)
        else:
            raise ValueError("No/unknown label source set")
