from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from smolagents import OpenAIModel, ToolCallingAgent

from data_quality_tools_agent.tools import (
    apply_cleaning_plan,
    compare_before_after,
    compute_correlations,
    detect_all_issues,
    plot_distributions,
    plot_quality_dashboard,
    prepare_run_dir,
    profile_table,
    render_quality_notebook,
    save_cleaned_table,
    save_report,
    select_best_strategy,
    suggest_dtypes,
    validate_and_load_table,
)


AGENT_INSTRUCTIONS = """You are ToolBasedDataQualityAgent, a compact tool-only orchestrator for tabular data quality work.

Rules:
- Use only the provided tools.
- Work only with CSV and Parquet inputs.
- Prefer the smallest number of tool calls that still gives a complete answer.
- Avoid creating unnecessary intermediate files.
- Save final artifacts under the prepared run directory.
- Compare at least two strategies in full-audit mode.
- Final answers must be short dictionaries with artifact paths.
"""


DETECT_TASK = """Run a quality-detection workflow for input_path.

Required behavior:
1. Validate and inspect the table.
2. Prepare a run directory.
3. Save profile, dtype-suggestion, and quality-report artifacts.
4. Save one correlation artifact when numeric columns allow it.
5. Generate dashboard plots and distribution plots.
6. Render notebook/data_quality_audit.ipynb from saved artifacts.
7. Return final_answer with report_path, notebook_path, figure_paths, and run_dir.
"""


FIX_TASK = """Run a cleaning workflow for input_path using strategy_json.

Required behavior:
1. Validate and inspect the table.
2. Prepare a run directory.
3. Apply the cleaning plan once.
4. Save summary/decision.md describing the strategy.
5. Return final_answer with cleaned_path, summary_path, and run_dir.
"""


COMPARE_TASK = """Compare quality before and after cleaning using before_path and after_path.

Required behavior:
1. Validate both tables.
2. Prepare a run directory.
3. Compare the two tables with one compare_before_after call.
4. Save CSV and JSON comparison artifacts.
5. Return final_answer with comparison_csv_path, comparison_json_path, score, and run_dir.
"""


FULL_AUDIT_TASK = """Run a full quality audit for input_path.

Required behavior:
1. Validate and inspect the table.
2. Prepare a run directory.
3. Save profile, dtype-suggestion, correlation, and baseline quality-report artifacts.
4. Use candidate_strategies_json if provided, otherwise compare at least two valid strategies.
5. For each strategy:
   - apply the cleaning plan
   - compare before vs after with compare_before_after
6. Select the best strategy with select_best_strategy.
7. Save the best cleaned dataset under cleaned/.
8. Save summary/decision.md explaining the chosen strategy and score.
9. Generate dashboard plots, distribution plots, and notebook/data_quality_audit.ipynb.
10. Return final_answer with report_path, comparison_paths, cleaned_path, notebook_path, decision_path, chosen_strategy, and run_dir.
"""


DETECT_TOOLS = [
    validate_and_load_table,
    prepare_run_dir,
    profile_table,
    suggest_dtypes,
    compute_correlations,
    detect_all_issues,
    plot_quality_dashboard,
    plot_distributions,
    render_quality_notebook,
]

FIX_TOOLS = [
    validate_and_load_table,
    prepare_run_dir,
    apply_cleaning_plan,
    save_report,
]

COMPARE_TOOLS = [
    validate_and_load_table,
    prepare_run_dir,
    compare_before_after,
]

FULL_AUDIT_TOOLS = [
    validate_and_load_table,
    prepare_run_dir,
    profile_table,
    suggest_dtypes,
    compute_correlations,
    detect_all_issues,
    apply_cleaning_plan,
    compare_before_after,
    select_best_strategy,
    save_cleaned_table,
    save_report,
    plot_quality_dashboard,
    plot_distributions,
    render_quality_notebook,
]


class ToolBasedDataQualityAgent:
    def __init__(
        self,
        model_id: str = "gpt-5-mini",
        artifacts_dir: str = "quality_tools_artifacts",
        task_description: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.artifacts_dir = Path(artifacts_dir)
        self.task_description = task_description
        self.last_result: Any | None = None

    def detect_issues(self, input_path: str, target_column: str | None = None, outlier_method: str = "iqr") -> dict:
        result = self._run_agent(
            task_name="detect",
            task=DETECT_TASK,
            additional_args={
                "input_path": input_path,
                "target_column": target_column or "",
                "outlier_method": outlier_method or "iqr",
                "artifacts_dir": str(self.artifacts_dir),
                "task_description": self.task_description or "",
            },
        )
        self._require_paths(result, ["report_path", "notebook_path"])
        return result

    def fix(
        self,
        input_path: str,
        strategy: dict,
        output_path: str | None = None,
        target_column: str | None = None,
        outlier_method: str = "iqr",
    ) -> dict:
        result = self._run_agent(
            task_name="fix",
            task=FIX_TASK,
            additional_args={
                "input_path": input_path,
                "strategy_json": json.dumps(strategy, ensure_ascii=False),
                "output_path": output_path or "",
                "target_column": target_column or "",
                "outlier_method": outlier_method or "iqr",
                "artifacts_dir": str(self.artifacts_dir),
                "task_description": self.task_description or "",
            },
        )
        self._require_paths(result, ["cleaned_path"])
        return result

    def compare(
        self,
        before_path: str,
        after_path: str,
        target_column: str | None = None,
        outlier_method: str = "iqr",
    ) -> dict:
        result = self._run_agent(
            task_name="compare",
            task=COMPARE_TASK,
            additional_args={
                "before_path": before_path,
                "after_path": after_path,
                "target_column": target_column or "",
                "outlier_method": outlier_method or "iqr",
                "artifacts_dir": str(self.artifacts_dir),
            },
        )
        self._require_paths(result, ["comparison_csv_path", "comparison_json_path"])
        return result

    def run_full_audit(
        self,
        input_path: str,
        target_column: str | None = None,
        candidate_strategies: list[dict] | None = None,
        input_format: str | None = None,
    ) -> dict:
        result = self._run_agent(
            task_name="full_audit",
            task=FULL_AUDIT_TASK,
            additional_args={
                "input_path": input_path,
                "target_column": target_column or "",
                "input_format": input_format or "",
                "candidate_strategies_json": json.dumps(candidate_strategies, ensure_ascii=False)
                if candidate_strategies is not None
                else "",
                "artifacts_dir": str(self.artifacts_dir),
                "task_description": self.task_description or "",
            },
            max_steps=18,
        )
        self._require_paths(result, ["report_path", "notebook_path", "decision_path"])
        return result

    @staticmethod
    def _tools_for_task(task_name: str) -> list:
        tool_map = {
            "detect": DETECT_TOOLS,
            "fix": FIX_TOOLS,
            "compare": COMPARE_TOOLS,
            "full_audit": FULL_AUDIT_TOOLS,
        }
        return tool_map[task_name]

    def _create_agent(self, task_name: str, max_steps: int) -> ToolCallingAgent:
        model = OpenAIModel(model_id=self.model_id, api_key=os.getenv("OPENAI_API_KEY"))
        return ToolCallingAgent(
            tools=self._tools_for_task(task_name),
            model=model,
            name="tool_based_data_quality_agent",
            description="Compact tool-only tabular data quality agent for CSV and Parquet datasets.",
            instructions=AGENT_INSTRUCTIONS,
            planning_interval=2,
            max_steps=max_steps,
            verbosity_level=2,
        )

    def _run_agent(self, *, task_name: str, task: str, additional_args: dict[str, Any], max_steps: int = 12) -> dict:
        agent = self._create_agent(task_name=task_name, max_steps=max_steps)
        result = agent.run(task=task, additional_args=additional_args)
        self.last_result = result
        return self._normalize_result(result)

    @staticmethod
    def _normalize_result(result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                return {"message": result}
        return {"result": result}

    @staticmethod
    def _require_paths(result: dict[str, Any], keys: list[str]) -> None:
        for key in keys:
            if key not in result:
                raise RuntimeError(f"Agent result is missing expected key '{key}': {result}")
            path = Path(result[key])
            if not path.exists():
                raise RuntimeError(f"Expected artifact for '{key}' does not exist: {path}")
