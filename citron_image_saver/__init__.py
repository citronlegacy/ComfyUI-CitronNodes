from typing import Any

from .nodes import ImageSaver, ImageSaverSimple, ImageSaverMetadata
from .nodes_literals import SeedGenerator, StringLiteral, SizeLiteral, IntLiteral, FloatLiteral, CfgLiteral, ConditioningConcatOptional
from .nodes_loaders import CheckpointLoaderWithName, UNETLoaderWithName
from .nodes_selectors import SamplerSelector, SchedulerSelector, SchedulerSelectorInspire, SchedulerSelectorEfficiency, SchedulerToString, SamplerToString, SchedulerInspireToString, SchedulerEfficiencyToString, InputParameters

NODE_CLASS_MAPPINGS: dict[str, Any] = {
    "Checkpoint Loader with Name (Citron Image Saver)": CheckpointLoaderWithName,
    "UNet loader with Name (Citron Image Saver)": UNETLoaderWithName,
    "Citron Image Saver": ImageSaver,
    "Citron Image Saver Simple": ImageSaverSimple,
    "Citron Image Saver Metadata": ImageSaverMetadata,
    "Sampler Selector (Citron Image Saver)": SamplerSelector,
    "Scheduler Selector (Citron Image Saver)": SchedulerSelector,
    "Scheduler Selector (inspire) (Citron Image Saver)": SchedulerSelectorInspire,
    "Scheduler Selector (Eff.) (Citron Image Saver)": SchedulerSelectorEfficiency,
    "Input Parameters (Citron Image Saver)": InputParameters,
    "Seed Generator (Citron Image Saver)": SeedGenerator,
    "String Literal (Citron Image Saver)": StringLiteral,
    "Width/Height Literal (Citron Image Saver)": SizeLiteral,
    "Cfg Literal (Citron Image Saver)": CfgLiteral,
    "Int Literal (Citron Image Saver)": IntLiteral,
    "Float Literal (Citron Image Saver)": FloatLiteral,
    "Conditioning Concat Optional (Citron Image Saver)": ConditioningConcatOptional,
    "SchedulerToString (Citron Image Saver)": SchedulerToString,
    "SchedulerInspireToString (Citron Image Saver)": SchedulerInspireToString,
    "SchedulerEfficiencyToString (Citron Image Saver)": SchedulerEfficiencyToString,
    "SamplerToString (Citron Image Saver)": SamplerToString,
}

WEB_DIRECTORY = None

__all__ = ['NODE_CLASS_MAPPINGS', 'WEB_DIRECTORY']
