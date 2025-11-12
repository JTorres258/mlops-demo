from pydantic import BaseModel, Field, model_validator, ValidationError
from typing import Optional, Literal, Annotated, Tuple
import yaml

class AugmentConfig(BaseModel):
    """Configuration for data augmentation."""

    flip_left_right: bool = False
    flip_up_down: bool = False
    random_crop: Optional[Tuple[Annotated[int, Field(strict=True, gt=0)],
                  Annotated[int, Field(strict=True, gt=0)]]] = None  # (H, W) > 0


class DataConfig(BaseModel):
    """Configuration for dataset preprocessing."""

    # Preprocessing
    resize: Optional[Tuple[Annotated[int, Field(strict=True, gt=0)],
                  Annotated[int, Field(strict=True, gt=0)]]] = None  # (H, W) > 0
    normalize: bool = True
    dtype: Literal["float32", "float16"] = "float32"
    
    # Data Augmentation
    augment: Optional[AugmentConfig] = None
    
    # Pipeline
    shuffle: bool = True
    shuffle_buffer_size: Annotated[int, Field(strict=True, gt=0)] = 1024
    batch_size: Annotated[int, Field(strict=True, gt=0)] = 64
    drop_remainder: bool = False
    prefetch: bool = True

    # Cross-field validation happens *after* all fields are parsed
    @model_validator(mode="after")
    def check_flip_and_resize(self):
        if self.augment and self.augment.random_crop and self.resize is not None:
            raise ValueError("resize cannot be used when flip_left_right is True.")
        return self
    

def load_config(path: str) -> DataConfig:
    """Load preprocessing configuration from a YAML file."""

    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return DataConfig(**cfg_dict)

if __name__ == "__main__":
    # Example usage
    try:
        config = load_config('/home/jorge/ComputerVisionProyects/mlops-demo/app/train/configs/config_test.yml')
        print(config)
    except ValidationError as e:
        print("Configuration Error:", e) 