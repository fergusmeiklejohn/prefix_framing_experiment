"""SQLite storage for experiment results."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .models import (
    AutomatedMetrics,
    ExperimentConfig,
    ExperimentSummary,
    JudgeRatings,
    PrefixCategory,
    PromptType,
    Trial,
)


class ExperimentStorage:
    """SQLite-based storage for experiment trials and results."""

    def __init__(self, db_path: str | Path = "results/experiment.db"):
        """Initialize storage with database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT,
                    config_json TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    timestamp TEXT,
                    model TEXT,
                    prompt_id TEXT,
                    prompt_text TEXT,
                    prompt_type TEXT,
                    prefix_id TEXT,
                    prefix_text TEXT,
                    prefix_category TEXT,
                    replication INTEGER,
                    temperature REAL,
                    full_response TEXT,
                    continuation_only TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    generation_time_ms INTEGER,
                    course_corrected INTEGER DEFAULT 0,
                    metrics_json TEXT,
                    judge_ratings_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
                );

                CREATE INDEX IF NOT EXISTS idx_trials_experiment ON trials(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_trials_prompt ON trials(prompt_id);
                CREATE INDEX IF NOT EXISTS idx_trials_prefix ON trials(prefix_id);
                CREATE INDEX IF NOT EXISTS idx_trials_replication ON trials(replication);
            """)

    def save_experiment(self, config: ExperimentConfig) -> str:
        """Save experiment configuration, return experiment_id."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (experiment_id, name, config_json, start_time)
                VALUES (?, ?, ?, ?)
                """,
                (
                    config.experiment_id,
                    config.name,
                    config.model_dump_json(),
                    datetime.now().isoformat(),
                ),
            )
        return config.experiment_id

    def update_experiment_end_time(self, experiment_id: str):
        """Update experiment end time."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE experiments SET end_time = ? WHERE experiment_id = ?",
                (datetime.now().isoformat(), experiment_id),
            )

    def save_trial(self, experiment_id: str, trial: Trial):
        """Save a single trial result."""
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trials (
                    trial_id, experiment_id, timestamp, model,
                    prompt_id, prompt_text, prompt_type,
                    prefix_id, prefix_text, prefix_category,
                    replication, temperature,
                    full_response, continuation_only,
                    input_tokens, output_tokens, generation_time_ms,
                    course_corrected, metrics_json, judge_ratings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial.trial_id,
                    experiment_id,
                    trial.timestamp.isoformat(),
                    trial.model,
                    trial.prompt_id,
                    trial.prompt_text,
                    trial.prompt_type.value,
                    trial.prefix_id,
                    trial.prefix_text,
                    trial.prefix_category.value,
                    trial.replication,
                    trial.temperature,
                    trial.full_response,
                    trial.continuation_only,
                    trial.input_tokens,
                    trial.output_tokens,
                    trial.generation_time_ms,
                    1 if trial.course_corrected else 0,
                    trial.metrics.model_dump_json() if trial.metrics else None,
                    trial.judge_ratings.model_dump_json() if trial.judge_ratings else None,
                ),
            )

    def get_trial(self, trial_id: str) -> Optional[Trial]:
        """Get a trial by ID."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM trials WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            if row:
                return self._row_to_trial(row)
        return None

    def get_trials(
        self,
        experiment_id: Optional[str] = None,
        prompt_id: Optional[str] = None,
        prefix_id: Optional[str] = None,
        prefix_category: Optional[PrefixCategory] = None,
    ) -> Iterator[Trial]:
        """Query trials with optional filters."""
        query = "SELECT * FROM trials WHERE 1=1"
        params = []

        if experiment_id:
            query += " AND experiment_id = ?"
            params.append(experiment_id)
        if prompt_id:
            query += " AND prompt_id = ?"
            params.append(prompt_id)
        if prefix_id:
            query += " AND prefix_id = ?"
            params.append(prefix_id)
        if prefix_category:
            query += " AND prefix_category = ?"
            params.append(prefix_category.value)

        with self._get_conn() as conn:
            for row in conn.execute(query, params):
                yield self._row_to_trial(row)

    def get_trial_count(self, experiment_id: str) -> int:
        """Get number of trials for an experiment."""
        with self._get_conn() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM trials WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            return result[0] if result else 0

    def get_completed_conditions(self, experiment_id: str) -> set[tuple[str, str, int]]:
        """Get set of (prompt_id, prefix_id, replication) that are completed."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT prompt_id, prefix_id, replication
                FROM trials WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchall()
            return {(r["prompt_id"], r["prefix_id"], r["replication"]) for r in rows}

    def get_experiment_summary(self, experiment_id: str) -> Optional[ExperimentSummary]:
        """Get summary statistics for an experiment."""
        with self._get_conn() as conn:
            exp_row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if not exp_row:
                return None

            stats = conn.execute(
                """
                SELECT
                    COUNT(*) as total,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output
                FROM trials WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()

            config = ExperimentConfig.model_validate_json(exp_row["config_json"])

            return ExperimentSummary(
                experiment_id=experiment_id,
                config=config,
                total_trials=stats["total"] or 0,
                completed_trials=stats["total"] or 0,
                failed_trials=0,
                total_input_tokens=stats["total_input"] or 0,
                total_output_tokens=stats["total_output"] or 0,
                start_time=datetime.fromisoformat(exp_row["start_time"])
                if exp_row["start_time"]
                else None,
                end_time=datetime.fromisoformat(exp_row["end_time"])
                if exp_row["end_time"]
                else None,
            )

    def update_trial_metrics(self, trial_id: str, metrics: AutomatedMetrics):
        """Update metrics for a trial."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE trials SET metrics_json = ? WHERE trial_id = ?",
                (metrics.model_dump_json(), trial_id),
            )

    def update_trial_ratings(self, trial_id: str, ratings: JudgeRatings):
        """Update judge ratings for a trial."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE trials SET judge_ratings_json = ? WHERE trial_id = ?",
                (ratings.model_dump_json(), trial_id),
            )

    def _row_to_trial(self, row: sqlite3.Row) -> Trial:
        """Convert a database row to a Trial object."""
        metrics = None
        if row["metrics_json"]:
            metrics = AutomatedMetrics.model_validate_json(row["metrics_json"])

        ratings = None
        if row["judge_ratings_json"]:
            ratings = JudgeRatings.model_validate_json(row["judge_ratings_json"])

        return Trial(
            trial_id=row["trial_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            model=row["model"],
            prompt_id=row["prompt_id"],
            prompt_text=row["prompt_text"],
            prompt_type=PromptType(row["prompt_type"]),
            prefix_id=row["prefix_id"],
            prefix_text=row["prefix_text"],
            prefix_category=PrefixCategory(row["prefix_category"]),
            replication=row["replication"],
            temperature=row["temperature"],
            full_response=row["full_response"],
            continuation_only=row["continuation_only"],
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            generation_time_ms=row["generation_time_ms"],
            course_corrected=bool(row["course_corrected"]),
            metrics=metrics,
            judge_ratings=ratings,
        )

    def export_to_json(self, experiment_id: str, output_path: str | Path):
        """Export experiment data to JSON."""
        output_path = Path(output_path)
        trials = list(self.get_trials(experiment_id=experiment_id))
        summary = self.get_experiment_summary(experiment_id)

        data = {
            "summary": summary.model_dump() if summary else None,
            "trials": [t.model_dump() for t in trials],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
