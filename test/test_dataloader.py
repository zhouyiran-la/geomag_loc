from datasets import create_magnetic_dataset_v2_dataloaders, MagneticDataSetV2
from pathlib import Path
from typing import cast
from datasets.transforms import DefaultTransform

if __name__ == "__main__":
    val_dir = str(Path("data") / "data_for_train_test_v12" / "4.26-resample-zscore" / "eval")
    val_loader = create_magnetic_dataset_v2_dataloaders(
        val_dir,
        batch_size=16,
        pattern=".csv",
        num_workers=0,
        shuffle_train=False,
        pin_memory=False,
        transform=DefaultTransform(),
        seq_len=128,
        stride=10
    )
    assert val_loader

    batch = next(iter(val_loader))

    print(batch["x_mag"][1], batch["y"][1])
    # ds = cast(MagneticDataSetV2, val_loader.dataset)
    # ds.__getitem__
