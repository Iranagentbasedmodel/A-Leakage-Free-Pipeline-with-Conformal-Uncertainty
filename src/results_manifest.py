"""
Run Manifest — v8.1
===================
Single source of truth for every number the manuscript quotes.

Root cause of the reviewer's worst finding (two numbers in one table cell,
"0.721 0.730"): the paper was assembled from TWO DIFFERENT RUNS and values
were typed in by hand. The manifest makes that structurally impossible —
every reported quantity is registered exactly once, tagged with the run
stamp, and the paper tables are EXPORTED from it (one number per cell).

Outputs (outputs/tables/paper_tables/):
    run_manifest_<stamp>.json         everything, machine-readable
    table2_performance_<stamp>.csv    held-out test performance
    table3_models_<stamp>.csv         baselines / ablations / seed statistics
    table4_logo_<stamp>.csv           leave-one-climate-out
    table5_counterfactual_<stamp>.csv student vs teacher vs ResStock
    seed_repetition_<stamp>.csv       raw per-seed scores
    shap_top_<tag>_<stamp>.json       SHAP values used by Fig. 3 (same run)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from config import TABLES_DIR

logger = logging.getLogger("results_manifest")


class RunManifest:
    def __init__(self, version: str, stamp: Optional[str] = None):
        self.stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = TABLES_DIR / "paper_tables"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data: Dict = {"run_stamp": self.stamp, "pipeline_version": version,
                           "generated": datetime.now().isoformat(timespec="seconds")}
        self._tables: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------ set
    def set(self, key: str, value) -> None:
        self.data[key] = value

    def update(self, d: Dict) -> None:
        self.data.update(d)

    def add_row(self, table: str, row: Dict) -> None:
        """Register one row of a paper table. One value per cell, by
        construction."""
        self._tables.setdefault(table, []).append(row)

    # ----------------------------------------------------------------- save
    def save(self) -> Path:
        p = self.dir / f"run_manifest_{self.stamp}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)
        for name, rows in self._tables.items():
            df = pd.DataFrame(rows)
            df.to_csv(self.dir / f"{name}_{self.stamp}.csv", index=False)
        logger.info(f"Manifest saved: {p} "
                    f"(+ {len(self._tables)} paper tables in {self.dir})")
        return p

    def table_df(self, name: str) -> pd.DataFrame:
        return pd.DataFrame(self._tables.get(name, []))
