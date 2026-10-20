# ComfyUI-CitronNodes (ComfyUI Custom Node)

A small collection of utility nodes for ComfyUI.

## Nodes

### Utility
| Node | Description |
|------|-------------|
| Get DateTime | Outputs the current date, time, and datetime as strings. Recomputed on every queue. |
| Get First and Final Frames | Outputs the first and final frames of an image batch. |
| Get GIF First and Final Frames | Takes a path to a `.gif` file and outputs its first and final frames. |

### Citron Image Saver (`Citron_ImageSaver`)
Based on [comfy-image-saver](https://github.com/giriss/comfy-image-saver), see [citron_image_saver/ATTRIBUTION.md](citron_image_saver/ATTRIBUTION.md).

| Node | Description |
|------|-------------|
| Citron Image Saver | Saves images with generation metadata. |
| Citron Image Saver Simple | Saves images with generation metadata, taking its metadata from the Metadata node. |
| Citron Image Saver Metadata | Prepares metadata for Citron Image Saver Simple. |

### Loaders (`Citron_ImageSaver/utils`)
| Node | Description |
|------|-------------|
| Checkpoint Loader with Name | Loads the U-Net, CLIP and VAE from a checkpoint and outputs its filename. |
| UNet loader with Name | Loads a U-Net model and outputs its filename. |

### Selectors (`Citron_ImageSaver/utils`)
| Node | Description |
|------|-------------|
| Sampler Selector | Provides one of the available ComfyUI samplers. |
| Scheduler Selector | Provides one of the standard KSampler schedulers. |
| Scheduler Selector (inspire) | Provides one of the KSampler (inspire) schedulers. |
| Scheduler Selector (Eff.) | Provides one of the KSampler (Eff.) schedulers. |
| Input Parameters | Combined node for seed, steps, cfg, sampler, scheduler and denoise. |
| SamplerToString | Outputs a standard sampler's name as a string. |
| SchedulerToString | Outputs a standard KSampler scheduler's name as a string. |
| SchedulerInspireToString | Outputs an Inspire Pack scheduler's name as a string. |
| SchedulerEfficiencyToString | Outputs an Efficiency pack scheduler's name as a string. |

### Literals (`Citron_ImageSaver/utils`)
| Node | Description |
|------|-------------|
| Seed Generator | Provides a seed as an integer. |
| String Literal | Provides a string. |
| Width/Height Literal | Provides width and height integers (step 8). |
| Int Literal | Provides an integer between 0 and 1000000. |
| Float Literal | Provides a floating point number (step 0.01). |
| Cfg Literal | Provides a CFG value between 0.0 and 100.0. |

### Conditioning
| Node | Description |
|------|-------------|
| Conditioning Concat Optional | Concatenates two conditionings; if `conditioning_from` is empty, `conditioning_to` passes through unchanged. |

## Install (manual)
```
git clone https://github.com/citronlegacy/Comfyui-CitronNodes.git ComfyUI/custom_nodes/ComfyUI-CitronNodes
# restart ComfyUI
```

## Install (ComfyUI Manager) [Coming soon]
Search "ComfyUI-CitronNodes" in Manager → Install.

## License
MIT
