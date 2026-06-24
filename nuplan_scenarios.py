"""Thin wrapper around NuPlan's scenario builder.

Replaces the dependency on ``driving_ruler_core/src/dr/nuplan_api/data.py`` so
notebooks under ``tutorials/buchi/`` need no external sys.path manipulation.

Environment variables (all optional — defaults match the project layout):

  NUPLAN_DATA_ROOT   root of the NuPlan dataset
  NUPLAN_MAPS_ROOT   directory containing the map files
  NUPLAN_MAP_VERSION map version string
  NUPLAN_SPLIT_ROOT  directory whose subdirs are per-split .db files
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Union

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_sequential import Sequential

# ── dataset paths (mirror dr/utils.py, overridable via environment) ───────────
_DATA_ROOT    = os.getenv("NUPLAN_DATA_ROOT",  "/home/pranav/Downloads/BTP_Suresh_KC/nuplan-v1.1_mini").rstrip("/")
_MAPS_ROOT    = os.getenv("NUPLAN_MAPS_ROOT",  "/home/pranav/Downloads/BTP_Suresh_KC/nuplan-maps-v1.0/maps")
_MAP_VERSION  = os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
_SPLIT_ROOT   = os.getenv("NUPLAN_SPLIT_ROOT",  os.path.join(_DATA_ROOT, "data", "cache"))


def get_scenarios(
    split: str,
    scenario_types: Optional[list[str]] = None,
    scenario_tokens: Optional[list[Sequence[str]]] = None,
    log_names: Optional[list[str]] = None,
    map_names: Optional[list[str]] = None,
    num_scenarios_per_type: Optional[int] = None,
    limit_total_scenarios: Optional[Union[int, float]] = None,
    expand_scenarios: bool = False,
    remove_invalid_goals: bool = False,
    shuffle: bool = False,
    timestamp_threshold_s: float = 2.0,
    ego_displacement_minimum_m: Optional[float] = None,
    ego_start_speed_threshold: Optional[float] = None,
    ego_stop_speed_threshold: Optional[float] = None,
    speed_noise_tolerance: Optional[float] = None,
    token_set_path: Optional[Path] = None,
    fraction_in_token_set_threshold: Optional[float] = None,
    worker_pool: Optional[WorkerPool] = None,
) -> list[AbstractScenario]:
    """Retrieve NuPlan scenarios matching the given filters.

    Args:
        split: Dataset split, e.g. ``'mini'``, ``'trainval'``, ``'test'``.
        scenario_types: Scenario type names to include.
        scenario_tokens: ``(log_name, token)`` pairs to include.
        log_names: Filter by log name.
        map_names: Filter by map name.
        num_scenarios_per_type: Cap per scenario type.
        limit_total_scenarios: Total cap (int) or fraction (float).
        expand_scenarios: Expand multi-sample scenarios.
        remove_invalid_goals: Drop scenarios with invalid mission goals.
        shuffle: Shuffle the result list.
        timestamp_threshold_s: Min interval between scenario timestamps (s).
        ego_displacement_minimum_m: Min ego travel distance (m).
        ego_start_speed_threshold: Ego speed must rise above this threshold.
        ego_stop_speed_threshold: Ego speed must fall to or below this threshold.
        speed_noise_tolerance: Speed-change noise floor.
        token_set_path: Path to a JSON file of lidarpc tokens.
        fraction_in_token_set_threshold: Token-set membership threshold.
        worker_pool: Worker pool for parallel scenario loading.

    Returns:
        List of matching :class:`AbstractScenario` instances.
    """
    split_data_root = os.path.join(_SPLIT_ROOT, split)

    scenario_builder = NuPlanScenarioBuilder(
        data_root=split_data_root,
        map_root=_MAPS_ROOT,
        sensor_root="",
        db_files=None,
        map_version=_MAP_VERSION,
    )

    scenario_filter = ScenarioFilter(
        scenario_types=scenario_types,
        scenario_tokens=scenario_tokens,
        log_names=log_names,
        map_names=map_names,
        num_scenarios_per_type=num_scenarios_per_type,
        limit_total_scenarios=limit_total_scenarios,
        expand_scenarios=expand_scenarios,
        remove_invalid_goals=remove_invalid_goals,
        shuffle=shuffle,
        timestamp_threshold_s=timestamp_threshold_s,
        ego_displacement_minimum_m=ego_displacement_minimum_m,
        ego_start_speed_threshold=ego_start_speed_threshold,
        ego_stop_speed_threshold=ego_stop_speed_threshold,
        speed_noise_tolerance=speed_noise_tolerance,
        token_set_path=token_set_path,
        fraction_in_token_set_threshold=fraction_in_token_set_threshold,
    )

    if worker_pool is None:
        worker_pool = Sequential()
    return scenario_builder.get_scenarios(scenario_filter, worker_pool)
