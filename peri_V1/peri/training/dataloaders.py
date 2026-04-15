"""Dataset and DataLoader builders for PERI training."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from peri.data import create_emotic_dataset_from_config
from peri.data.emotic_dataset import EMOTICDataset, EMOTICRecord
from peri.preprocess import EMOTICPreprocessedDataset, LandmarkExtractor, PASDebugWriter, PASGenerator

from .config import TrainingConfig


def collate_emotic_batch(samples: list[dict[str, object]]) -> dict[str, object]:
    first = samples[0]
    batch: dict[str, object] = {}
    for key, value in first.items():
        values = [sample[key] for sample in samples]
        if isinstance(value, torch.Tensor):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(value, dict):
            nested: dict[str, object] = {}
            for nested_key in value:
                nested_values = [item[nested_key] for item in values]
                nested[nested_key] = torch.stack(nested_values, dim=0) if isinstance(value[nested_key], torch.Tensor) else nested_values
            batch[key] = nested
        else:
            batch[key] = values
    return batch


@dataclass
class DataLoaderBundle:
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    landmark_extractor: LandmarkExtractor | None = None

    def close(self) -> None:
        if self.landmark_extractor is not None:
            self.landmark_extractor.close()


def _limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None:
        return dataset
    return Subset(dataset, list(range(min(limit, len(dataset)))))


def _precomputed_pas_lookup_key(
    split: str,
    filename: str,
    bbox: tuple[int, int, int, int],
) -> tuple[str, str, tuple[int, int, int, int]]:
    return split, filename.replace("\\", "/"), bbox


def _load_precomputed_pas_index_map(precomputed_pas_root: Path, split: str) -> dict[tuple[str, str, tuple[int, int, int, int]], int]:
    csv_path = Path(precomputed_pas_root).resolve() / f"{split}.csv"
    if not csv_path.exists():
        return {}
    index_map: dict[tuple[str, str, tuple[int, int, int, int]], int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            bbox = (
                int(round(float(row["x1"]))),
                int(round(float(row["y1"]))),
                int(round(float(row["x2"]))),
                int(round(float(row["y2"]))),
            )
            key = _precomputed_pas_lookup_key(split, str(row["filename"]), bbox)
            index_map[key] = idx
    return index_map


def _get_base_emotic_dataset(dataset: Dataset) -> EMOTICDataset | None:
    """Recursively find the base EMOTICDataset."""
    if isinstance(dataset, EMOTICDataset):
        return dataset
    if hasattr(dataset, "base_dataset"):
        return _get_base_emotic_dataset(dataset.base_dataset)
    if hasattr(dataset, "dataset"):
        return _get_base_emotic_dataset(dataset.dataset)
    return None


def _record_precomputed_pas_key(record: EMOTICRecord) -> tuple[str, str, tuple[int, int, int, int]]:
    bbox = tuple(int(round(float(value))) for value in record.bbox.detach().cpu().tolist())
    return _precomputed_pas_lookup_key(record.source_split, record.filename, bbox)


def _validate_precomputed_pas_coverage(
    dataset: EMOTICDataset,
    index_map: dict[tuple[str, str, tuple[int, int, int, int]], int],
    *,
    split_name: str,
) -> None:
    missing = [record.sample_id for record in dataset.records if _record_precomputed_pas_key(record) not in index_map]
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(
            f"Precomputed PAS coverage is incomplete for split {split_name!r}: "
            f"{len(missing)}/{len(dataset.records)} official samples are missing from the PAS index. "
            f"Example sample_ids: {preview}"
        )


def _create_weighted_sampler(dataset: Dataset) -> WeightedRandomSampler | None:
    base_ds = _get_base_emotic_dataset(dataset)
    if base_ds is None:
        return None

    # Calculate global frequencies from the records we have in the current (possibly limited) dataset
    indices = list(range(len(dataset)))
    if isinstance(dataset, Subset):
        indices = dataset.indices

    all_emotions = []
    for idx in indices:
        all_emotions.append(base_ds.records[idx].emotion)
    
    emotions_tensor = torch.stack(all_emotions) # [N, 26]
    class_counts = emotions_tensor.sum(dim=0) # [26]
    
    # Weight per class: 1/count
    # Use max(count, 1) to avoid zero-division
    class_weights = 1.0 / torch.clamp(class_counts, min=1.0)
    
    # Sample weight = sum of weights of its positive classes
    sample_weights = (emotions_tensor * class_weights).sum(dim=1)
    
    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(indices),
        replacement=True,
    )


def build_dataloaders(config: TrainingConfig) -> DataLoaderBundle:
    train_dataset: Dataset = create_emotic_dataset_from_config(config, split="train")
    val_dataset: Dataset = create_emotic_dataset_from_config(config, split="val")
    test_dataset: Dataset = create_emotic_dataset_from_config(config, split="test")

    landmark_extractor = None
    if config.uses_pas:
        pas_debug_writer = None
        pas_generator = None
        precomputed_pas_index_maps = None
        if config.precomputed_pas_root is None:
            landmark_extractor = LandmarkExtractor(
                asset_root=config.mediapipe_asset_root,
                prefer_holistic=config.mode == "paper_faithful",
                use_full_image_fallback=config.mode == "experimental",
            )
            pas_debug_writer = PASDebugWriter(config.pas_debug_dir, max_samples=config.pas_debug_max_samples) if config.pas_debug and config.pas_debug_dir is not None else None
            pas_generator = PASGenerator(
                sigma=config.pas_sigma,
                rho=config.pas_rho,
                radius_scale=config.pas_radius_scale,
                binary_mask=config.pas_binary,
            )
        else:
            train_pas_index_map: dict[tuple[str, str, tuple[int, int, int, int]], int] = {}
            train_pas_splits = ("train", "extra_train") if config.include_extra_train else ("train",)
            for source_split in train_pas_splits:
                train_pas_index_map.update(_load_precomputed_pas_index_map(config.precomputed_pas_root, source_split))
            precomputed_pas_index_maps = {
                "train": train_pas_index_map,
                "val": _load_precomputed_pas_index_map(config.precomputed_pas_root, "val"),
                "test": _load_precomputed_pas_index_map(config.precomputed_pas_root, "test"),
            }
            if config.mode == "paper_faithful":
                _validate_precomputed_pas_coverage(train_dataset, precomputed_pas_index_maps["train"], split_name="train")
                _validate_precomputed_pas_coverage(val_dataset, precomputed_pas_index_maps["val"], split_name="val")
                _validate_precomputed_pas_coverage(test_dataset, precomputed_pas_index_maps["test"], split_name="test")
        # augment is passed only for training split; val/test never augment.
        # When precomputed PAS is used, the base EMOTICDataset has augment=False
        # and this wrapper applies synchronized augmentation to body crop + PAS.
        train_augment = config.augment and config.precomputed_pas_root is not None
        train_dataset = EMOTICPreprocessedDataset(
            train_dataset,
            landmark_extractor=landmark_extractor,
            pas_generator=pas_generator,
            pas_debug_writer=pas_debug_writer,
            landmark_cache_dir=config.landmark_cache_dir,
            precomputed_pas_root=config.precomputed_pas_root,
            precomputed_pas_index_map=precomputed_pas_index_maps["train"] if precomputed_pas_index_maps is not None else None,
            augment=train_augment,
        )
        val_dataset = EMOTICPreprocessedDataset(
            val_dataset,
            landmark_extractor=landmark_extractor,
            pas_generator=pas_generator,
            pas_debug_writer=pas_debug_writer,
            landmark_cache_dir=config.landmark_cache_dir,
            precomputed_pas_root=config.precomputed_pas_root,
            precomputed_pas_index_map=precomputed_pas_index_maps["val"] if precomputed_pas_index_maps is not None else None,
        )
        test_dataset = EMOTICPreprocessedDataset(
            test_dataset,
            landmark_extractor=landmark_extractor,
            pas_generator=pas_generator,
            pas_debug_writer=pas_debug_writer,
            landmark_cache_dir=config.landmark_cache_dir,
            precomputed_pas_root=config.precomputed_pas_root,
            precomputed_pas_index_map=precomputed_pas_index_maps["test"] if precomputed_pas_index_maps is not None else None,
        )

    train_dataset = _limit_dataset(train_dataset, config.max_train_samples)
    val_dataset = _limit_dataset(val_dataset, config.max_val_samples)
    test_dataset = _limit_dataset(test_dataset, config.max_test_samples)

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    
    train_sampler = None
    shuffle_train = True
    if config.use_weighted_sampler:
        train_sampler = _create_weighted_sampler(train_dataset)
        if train_sampler is not None:
            shuffle_train = False # Sampler and shuffle are mutually exclusive

    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "collate_fn": collate_emotic_batch,
        "pin_memory": True,
    }
    train_loader = DataLoader(
        train_dataset, 
        shuffle=shuffle_train, 
        sampler=train_sampler, 
        generator=generator, 
        **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)

    return DataLoaderBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        landmark_extractor=landmark_extractor,
    )
