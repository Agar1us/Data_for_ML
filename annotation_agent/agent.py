from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from smolagents import OpenAIModel, ToolCallingAgent
except ImportError:  # pragma: no cover - import fallback for static analysis environments
    class OpenAIModel:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("smolagents is required to run AnnotationAgent LLM spec generation.")

    class ToolCallingAgent:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("smolagents is required to run AnnotationAgent LLM spec generation.")

from annotation_agent.config import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_MODEL_PATH,
    DEFAULT_REVIEW_LINK_MODE,
    ToolResult,
)
from annotation_agent.models import RunContext
from annotation_agent.tools import (
    compute_annotation_quality_impl,
    export_labelstudio_predictions_impl,
    generate_annotation_spec,
    generate_annotation_spec_impl,
    inspect_image_dataset_impl,
    prepare_run_dir_impl,
    run_yoloe_labeling_impl,
    save_segmentation_masks_impl,
    split_low_confidence_examples_impl,
    summarize_annotation_examples_impl,
)


SPEC_AGENT_INSTRUCTIONS = """You write concise, useful image annotation specifications.

Rules:
- Use the provided structured summary only.
- Produce a concrete Markdown spec that a human annotator can follow.
- Make class definitions operational and distinguish classes clearly.
- Use exactly one tool call to save the final Markdown.
- Final answer must be the tool result only.
"""


SPEC_GENERATION_TASK_TEMPLATE = """Generate the annotation spec for task "{task}".

Structured summary:
{summary_json}

Write Markdown with these sections:
- Title
- Task Description
- Classes
- Edge Cases
- Instructions

Important:
- Improve vague class definitions when possible from the summary context.
- Keep example file paths from the summary.
- When uncertain cases exist, explain how a human annotator should resolve them.
- Save the result by calling generate_annotation_spec with:
  - summary_path="{summary_path}"
  - task="{task}"
  - output_path="{output_path}"
  - spec_markdown=<your full markdown>
"""


class AnnotationAgent:
    def __init__(
        self,
        modality: str = "image",
        classes: list[str] | None = None,
        class_definitions: dict[str, str] | None = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_path: str = DEFAULT_MODEL_PATH,
        model_id: str = DEFAULT_LLM_MODEL_ID,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
        review_link_mode: str = DEFAULT_REVIEW_LINK_MODE,
    ) -> None:
        if modality != "image":
            raise ValueError("AnnotationAgent v1 supports only modality='image'.")
        self.modality = modality
        self.classes = classes
        self.class_definitions = class_definitions or {}
        self.confidence_threshold = float(confidence_threshold)
        self.model_path = model_path
        self.model_id = model_id
        self.artifacts_dir = Path(artifacts_dir)
        self.review_link_mode = review_link_mode
        self.last_result: dict[str, Any] | None = None
        self.last_run_dir: Path | None = None
        self.last_labeled_csv: Path | None = None
        self.last_manual_review_manifest: Path | None = None
        self.last_context: RunContext | None = None

    def auto_label(self, dataset_dir: str) -> pd.DataFrame:
        dataset_name = Path(dataset_dir).resolve().name
        run_context = prepare_run_dir_impl(str(self.artifacts_dir), dataset_name=dataset_name)
        run_dir = Path(run_context["run_dir"])
        reports_dir = Path(run_context["reports_dir"])
        labeled_dir = Path(run_context["labeled_dir"])
        manual_review_dir = Path(run_context["manual_review_dir"])
        masks_dir = Path(run_context["masks_dir"])

        inspected = inspect_image_dataset_impl(
            dataset_dir=dataset_dir,
            dataset_output_path=str(labeled_dir / "dataset_index.csv"),
            validation_output_path=str(reports_dir / "dataset_validation.json"),
        )
        validation = inspected["validation"]
        scan = inspected["scan"]

        inferred_classes = sorted(scan["class_counts"].keys())
        self.classes = list(self.classes or inferred_classes)

        labeling = run_yoloe_labeling_impl(
            dataset_csv_path=str(scan["output_path"]),
            classes=self.classes,
            output_path=str(labeled_dir / "labeled_raw.csv"),
            model_path=self.model_path,
            masks_dir=str(masks_dir),
            report_output_path=str(reports_dir / "annotation_report.json"),
        )
        finalized = save_segmentation_masks_impl(
            input_path=str(labeling["output_path"]),
            output_path=str(labeled_dir / "labeled.csv"),
            masks_dir=str(masks_dir),
        )
        split = split_low_confidence_examples_impl(
            input_path=str(finalized["output_path"]),
            review_dir=str(manual_review_dir),
            threshold=self.confidence_threshold,
            output_confident_path=str(labeled_dir / "confident.csv"),
            output_uncertain_path=str(labeled_dir / "uncertain.csv"),
            manifest_output_path=str(manual_review_dir / "manifest.csv"),
            link_mode=self.review_link_mode,
        )
        quality = compute_annotation_quality_impl(
            input_path=str(finalized["output_path"]),
            output_path=str(reports_dir / "quality_metrics.json"),
            confidence_threshold=self.confidence_threshold,
        )

        result = {
            "run_dir": str(run_dir),
            "labeled_csv": str(finalized["output_path"]),
            "confident_csv": str(split["confident_output_path"]),
            "uncertain_csv": str(split["uncertain_output_path"]),
            "manual_review_manifest": str(split["manifest_output_path"]),
            "quality_metrics_json": str(quality["output_path"]),
            "dataset_validation_json": str(validation["output_path"]),
            "annotation_report_json": str(labeling["report_output_path"]),
            "classes": self.classes,
            "model_version": labeling["model_version"],
        }
        self._require_paths(
            result,
            [
                "labeled_csv",
                "quality_metrics_json",
                "manual_review_manifest",
                "dataset_validation_json",
                "annotation_report_json",
            ],
        )
        self._remember_run(result)
        self.last_result = result
        return pd.read_csv(Path(result["labeled_csv"]))

    def generate_spec(self, task: str, context: RunContext | None = None) -> ToolResult:
        resolved = self._resolve_context(context)
        labeled_csv = Path(resolved.labeled_csv)
        run_dir = Path(resolved.run_dir)
        summary_path = run_dir / "reports" / "spec_summary.json"
        summary = summarize_annotation_examples_impl(
            input_path=str(labeled_csv),
            task=task,
            output_path=str(summary_path),
            class_definitions=self.class_definitions or None,
        )
        spec_path = run_dir / "summary" / "annotation_spec.md"
        if self._llm_available():
            result = self._generate_spec_with_agent(task=task, summary_path=summary_path, summary=summary, output_path=spec_path)
        else:
            result = generate_annotation_spec_impl(
                summary_path=str(summary_path),
                task=task,
                output_path=str(spec_path),
            )
        final_result = {
            "run_dir": str(run_dir),
            "annotation_spec_md": str(result["output_path"]),
            "spec_summary_json": str(summary["output_path"]),
        }
        self._require_paths(final_result, ["annotation_spec_md", "spec_summary_json"])
        self._remember_run(final_result)
        self.last_result = final_result
        return final_result

    def check_quality(self, human_labels_path: str | None = None, context: RunContext | None = None) -> ToolResult:
        resolved = self._resolve_context(context)
        labeled_csv = Path(resolved.labeled_csv)
        run_dir = Path(resolved.run_dir)
        quality = compute_annotation_quality_impl(
            input_path=str(labeled_csv),
            output_path=str(run_dir / "reports" / "quality_metrics.json"),
            human_labels_path=human_labels_path or "",
            confidence_threshold=self.confidence_threshold,
        )
        result = {
            "run_dir": str(run_dir),
            "quality_metrics_json": str(quality["output_path"]),
        }
        self._require_paths(result, ["quality_metrics_json"])
        self._remember_run(result)
        self.last_result = result
        return result

    def export_to_labelstudio(self, context: RunContext | None = None) -> ToolResult:
        resolved = self._resolve_context(context)
        labeled_csv = Path(resolved.labeled_csv)
        run_dir = Path(resolved.run_dir)
        export = export_labelstudio_predictions_impl(
            input_path=str(labeled_csv),
            output_path=str(run_dir / "reports" / "labelstudio_import.json"),
            review_output_path=str(run_dir / "reports" / "labelstudio_review.json"),
            review_manifest_path=resolved.manual_review_manifest,
            model_version=Path(self.model_path).name,
        )
        result = {
            "run_dir": str(run_dir),
            "labelstudio_import_json": str(export["output_path"]),
            "labelstudio_review_json": str(export["review_output_path"]),
        }
        self._require_paths(result, ["labelstudio_import_json", "labelstudio_review_json"])
        self._remember_run(result)
        self.last_result = result
        return result

    def run_pipeline(self, dataset_dir: str, task: str, human_labels_path: str | None = None) -> ToolResult:
        self.auto_label(dataset_dir)
        context = self.get_run_context()
        spec = self.generate_spec(task, context=context)
        quality = self.check_quality(human_labels_path=human_labels_path, context=context)
        exports = self.export_to_labelstudio(context=context)
        result = {
            "run_dir": context.run_dir,
            "labeled_csv": context.labeled_csv,
            "manual_review_manifest": context.manual_review_manifest,
            "annotation_spec_md": spec["annotation_spec_md"],
            "quality_metrics_json": quality["quality_metrics_json"],
            "labelstudio_import_json": exports["labelstudio_import_json"],
            "labelstudio_review_json": exports["labelstudio_review_json"],
            "classes": self.classes or [],
        }
        self._require_paths(
            result,
            [
                "labeled_csv",
                "annotation_spec_md",
                "quality_metrics_json",
                "labelstudio_import_json",
                "labelstudio_review_json",
                "manual_review_manifest",
            ],
        )
        self._remember_run(result)
        self.last_result = result
        return result

    def get_run_context(self) -> RunContext:
        if self.last_context is None:
            raise RuntimeError("Run context is missing. Run auto_label() first.")
        return self.last_context

    def _create_spec_agent(self) -> ToolCallingAgent:
        model = OpenAIModel(model_id=self.model_id, api_key=os.getenv("OPENAI_API_KEY"))
        return ToolCallingAgent(
            tools=[generate_annotation_spec],
            model=model,
            name="annotation_spec_writer",
            description="Writes annotation specification markdown from structured summaries.",
            instructions=SPEC_AGENT_INSTRUCTIONS,
            planning_interval=1,
            max_steps=4,
            verbosity_level=1,
        )

    def _generate_spec_with_agent(self, task: str, summary_path: Path, summary: ToolResult, output_path: Path) -> ToolResult:
        agent = self._create_spec_agent()
        task_prompt = SPEC_GENERATION_TASK_TEMPLATE.format(
            task=task,
            summary_json=json.dumps(summary["summary"], ensure_ascii=False, indent=2),
            summary_path=str(summary_path),
            output_path=str(output_path),
        )
        result = self._normalize_result(agent.run(task=task_prompt))
        if "output_path" not in result or not Path(result["output_path"]).exists():
            return generate_annotation_spec_impl(summary_path=str(summary_path), task=task, output_path=str(output_path))
        return result

    def _llm_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY")) and ToolCallingAgent.__module__ != __name__

    def _remember_run(self, result: ToolResult) -> None:
        run_dir = result.get("run_dir")
        labeled_csv = result.get("labeled_csv")
        manual_review_manifest = result.get("manual_review_manifest")
        if run_dir:
            self.last_run_dir = Path(run_dir)
        if labeled_csv:
            self.last_labeled_csv = Path(labeled_csv)
        if manual_review_manifest:
            self.last_manual_review_manifest = Path(manual_review_manifest)
        if self.last_run_dir and self.last_labeled_csv and self.last_manual_review_manifest:
            self.last_context = RunContext(
                run_dir=str(self.last_run_dir),
                labeled_csv=str(self.last_labeled_csv),
                manual_review_manifest=str(self.last_manual_review_manifest),
                quality_metrics_json=str(result.get("quality_metrics_json") or ""),
                annotation_spec_md=str(result.get("annotation_spec_md") or ""),
                labelstudio_import_json=str(result.get("labelstudio_import_json") or ""),
                labelstudio_review_json=str(result.get("labelstudio_review_json") or ""),
            )

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
    def _require_paths(result: ToolResult, keys: list[str]) -> None:
        for key in keys:
            if key not in result:
                raise RuntimeError(f"Agent result is missing expected key '{key}': {result}")
            path = Path(result[key])
            if not path.exists():
                raise RuntimeError(f"Expected artifact for '{key}' does not exist: {path}")
            if path.is_file() and path.stat().st_size == 0:
                raise RuntimeError(f"Artifact for '{key}' is empty: {path}")

    def _resolve_context(self, context: RunContext | None) -> RunContext:
        return context if context is not None else self.get_run_context()

    def _require_state(self, attribute: str) -> Path:
        value = getattr(self, attribute)
        if value is None:
            raise RuntimeError("Required pipeline state is missing. Run auto_label() first or use run_pipeline().")
        return value
