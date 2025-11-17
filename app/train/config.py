from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, Literal, Annotated, Tuple
import yaml
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))


class DataConfig(BaseModel):
    """Configuration for input data."""

    # Preprocessing
    input_size: Tuple[
            Annotated[int, Field(strict=True, gt=0)],
            Annotated[int, Field(strict=True, gt=0)],
        ] = [224, 224]  # (H, W) > 0
    resize: Optional[Tuple[
            Annotated[int, Field(strict=True, gt=0)],
            Annotated[int, Field(strict=True, gt=0)],
        ]] = None  # (H, W) > 0
    normalize: bool = True
    dtype: Literal["float32", "float16"] = "float32"


class PipelineConfig(BaseModel):
    """Configuration for data pipeline."""

    # Pipeline
    cache: bool = False
    shuffle: bool = True
    shuffle_buffer_size: Annotated[int, Field(strict=True, gt=0)] = 1024
    batch_size: Annotated[int, Field(strict=True, gt=0)] = 64
    drop_remainder: bool = False
    prefetch: bool = True


class AugmentConfig(BaseModel):
    """Configuration for data augmentation."""

    flip_left_right: bool = False
    flip_up_down: bool = False
    random_crop: bool = False
    brightness: Optional[float] = None # e.g., 0.1 for ±10% brightness adjustment


class TrainConfig(BaseModel):
    """Configuration for training."""
    model_dir: Optional[str] = "./models/"
    checkpoint_dir: Optional[str] = "./models/checkpoints/"
    epochs: Annotated[int, Field(strict=True, gt=0)] = 10
    learning_rate: Annotated[float, Field(strict=True, gt=0)] = 0.001
    optimizer: str = "adam"
    loss: str = "sparse_categorical_crossentropy"
    metrics: list[str] = ["accuracy"]
    seed: Annotated[int, Field(strict=True, gt=0)] = 42
    mixed_precision: bool = False
    strategy: str = "auto"         # auto | single | mirrored | tpu


class ModelConfig(BaseModel):
    """Configuration for model architecture."""

    architecture: str = "resnet50"
    weights: Optional[str] = None
    num_classes: Annotated[int, Field(strict=True, gt=1)] = 1000
    drop_rate: Annotated[float, Field(strict=True, ge=0)] = 0.5
    freeze_backbone: bool = False
    

class ExperimentConfig(BaseModel):
    experiment_name: Optional[str] = None
    data: DataConfig
    pipeline: PipelineConfig
    augment: Optional[AugmentConfig] = None
    train: TrainConfig
    model: ModelConfig

    # Cross-field validation happens *after* all fields are parsed
    @model_validator(mode="after")
    def check_crop_and_resize(self):
        crop = bool(self.augment and getattr(self.augment, "random_crop", False))
        resize = bool(self.data and getattr(self.data, "resize", False))

        # They cannot both be enabled
        if crop and resize:
            raise ValueError("Cropping and resizing cannot be used together.")

        # At least one must be enabled
        if not crop and not resize:
            raise ValueError("You must enable either cropping and resizing.")

        return self
    

def load_config(path: str) -> ExperimentConfig:
    """Load preprocessing configuration from a YAML file."""

    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return ExperimentConfig(**cfg_dict)


if __name__ == "__main__":
    # Example usage
    try:
        config = load_config("./configs/config_test.yml")
        print(config)
    except ValidationError as e:
        print("Configuration Error:", e)
