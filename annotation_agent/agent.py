from __future__ import annotations

import json
import os
from html import escape
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
    AL_HUMAN_LABELS_FILE_NAME,
    AL_LABELS_FILE_NAME,
    COLUMN_FOLDER_LABEL,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_LABEL_ASSIGNMENT_MODE,
    DEFAULT_LLM_MODEL_ID,
    DEFAULT_MODEL_PATH,
    DEFAULT_TASK_MODE,
    ToolResult,
)
from annotation_agent.models import RunContext
from annotation_agent.tools import (
    build_object_labels_impl,
    compute_annotation_quality_impl,
    convert_labelstudio_export_to_object_labels_impl,
    export_labelstudio_predictions_impl,
    generate_annotation_spec,
    generate_annotation_spec_impl,
    inspect_image_dataset_impl,
    prepare_run_dir_impl,
    run_yoloe_labeling_impl,
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
        object_prompts: list[str] | None = None,
        task_mode: str = DEFAULT_TASK_MODE,
        label_assignment_mode: str = DEFAULT_LABEL_ASSIGNMENT_MODE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        model_path: str = DEFAULT_MODEL_PATH,
        model_id: str = DEFAULT_LLM_MODEL_ID,
        artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
        labelstudio_document_root: str | None = None,
    ) -> None:
        if modality != "image":
            raise ValueError("AnnotationAgent v1 supports only modality='image'.")
        if task_mode != DEFAULT_TASK_MODE:
            raise ValueError(f"AnnotationAgent v1 supports only task_mode='{DEFAULT_TASK_MODE}'.")
        if label_assignment_mode != DEFAULT_LABEL_ASSIGNMENT_MODE:
            raise ValueError(
                f"AnnotationAgent v1 supports only label_assignment_mode='{DEFAULT_LABEL_ASSIGNMENT_MODE}'."
            )
        self.modality = modality
        self.classes = classes
        self.class_definitions = class_definitions or {}
        self.object_prompts = [str(item).strip() for item in (object_prompts or []) if str(item).strip()]
        self.task_mode = task_mode
        self.label_assignment_mode = label_assignment_mode
        self.confidence_threshold = float(confidence_threshold)
        self.model_path = model_path
        self.model_id = model_id
        self.artifacts_dir = Path(artifacts_dir)
        self.labelstudio_document_root = str(Path(labelstudio_document_root).resolve()) if labelstudio_document_root else str(Path.cwd().resolve())
        self.last_result: dict[str, Any] | None = None
        self.last_run_dir: Path | None = None
        self.last_labels_csv: Path | None = None
        self.last_labeled_df: pd.DataFrame | None = None
        self.last_context: RunContext | None = None

    def auto_label(self, dataset_dir: str, task_description: str = "") -> pd.DataFrame:
        run_context = prepare_run_dir_impl(str(self.artifacts_dir))
        run_dir = Path(run_context["run_dir"])
        reports_dir = Path(run_context["reports_dir"])
        dataset_index_path = run_dir / "reports" / ".dataset_index.tmp.csv"
        runtime_labeled_path = run_dir / "reports" / ".labeled_runtime.tmp.csv"

        try:
            inspected = inspect_image_dataset_impl(
                dataset_dir=dataset_dir,
                dataset_output_path=str(dataset_index_path),
                validation_output_path="",
            )
            scan = inspected["scan"]

            inferred_classes = sorted(scan["class_counts"].keys())
            self.classes = list(self.classes or inferred_classes)
            self.object_prompts = self._resolve_object_prompts(Path(dataset_dir), task_description=task_description)

            labeling = run_yoloe_labeling_impl(
                dataset_csv_path=str(scan["output_path"]),
                object_prompts=self.object_prompts,
                output_path=str(runtime_labeled_path),
                model_path=self.model_path,
                masks_dir="",
                report_output_path=str(reports_dir / "annotation_report.json"),
                task_mode=self.task_mode,
                label_assignment_mode=self.label_assignment_mode,
            )
            quality = compute_annotation_quality_impl(
                input_path=str(labeling["output_path"]),
                output_path=str(reports_dir / "quality_metrics.json"),
                confidence_threshold=self.confidence_threshold,
            )
            labels = build_object_labels_impl(
                input_path=str(labeling["output_path"]),
                output_path=str(reports_dir / AL_LABELS_FILE_NAME),
                confidence_threshold=self.confidence_threshold,
            )
            self.last_labeled_df = pd.read_csv(Path(labeling["output_path"]))
        finally:
            if dataset_index_path.exists():
                dataset_index_path.unlink()
            if runtime_labeled_path.exists():
                runtime_labeled_path.unlink()

        result = {
            "run_dir": str(run_dir),
            "labels_csv": str(labels["output_path"]),
            "quality_metrics_json": str(quality["output_path"]),
            "annotation_report_json": str(labeling["report_output_path"]),
            "classes": self.classes,
            "object_prompts": self.object_prompts,
            "label_assignment_mode": self.label_assignment_mode,
            "task_mode": self.task_mode,
            "model_version": labeling["model_version"],
        }
        self._require_paths(
            result,
            [
                "labels_csv",
                "quality_metrics_json",
                "annotation_report_json",
            ],
        )
        self._remember_run(result)
        self.last_result = result
        return pd.read_csv(Path(result["labels_csv"]))

    def generate_spec(self, task: str, context: RunContext | None = None) -> ToolResult:
        resolved = self._resolve_context(context)
        run_dir = Path(resolved.run_dir)
        summary_path = run_dir / "reports" / ".spec_summary.tmp.json"
        spec_path = run_dir / "summary" / "annotation_spec.md"
        labeled_csv = self._materialize_runtime_labeled_csv(run_dir)
        try:
            summary = summarize_annotation_examples_impl(
                input_path=str(labeled_csv),
                task=task,
                output_path=str(summary_path),
                class_definitions=self.class_definitions or None,
                object_prompts=resolved.object_prompts or self.object_prompts,
                label_assignment_mode=resolved.label_assignment_mode or self.label_assignment_mode,
            )
            if self._llm_available():
                result = self._generate_spec_with_agent(task=task, summary_path=summary_path, summary=summary, output_path=spec_path)
            else:
                result = generate_annotation_spec_impl(
                    summary_path=str(summary_path),
                    task=task,
                    output_path=str(spec_path),
                )
        finally:
            if labeled_csv.exists():
                labeled_csv.unlink()
            if summary_path.exists():
                summary_path.unlink()
        final_result = {
            "run_dir": str(run_dir),
            "annotation_spec_md": str(result["output_path"]),
            "object_prompts": resolved.object_prompts or self.object_prompts,
            "label_assignment_mode": resolved.label_assignment_mode or self.label_assignment_mode,
        }
        self._require_paths(final_result, ["annotation_spec_md"])
        self._remember_run(final_result)
        self.last_result = final_result
        return final_result

    def check_quality(self, human_labels_path: str | None = None, context: RunContext | None = None) -> ToolResult:
        resolved = self._resolve_context(context)
        run_dir = Path(resolved.run_dir)
        labeled_csv = self._materialize_runtime_labeled_csv(run_dir)
        try:
            quality = compute_annotation_quality_impl(
                input_path=str(labeled_csv),
                output_path=str(run_dir / "reports" / "quality_metrics.json"),
                human_labels_path=human_labels_path or "",
                confidence_threshold=self.confidence_threshold,
            )
        finally:
            if labeled_csv.exists():
                labeled_csv.unlink()
        result = {
            "run_dir": str(run_dir),
            "quality_metrics_json": str(quality["output_path"]),
            "object_prompts": resolved.object_prompts or self.object_prompts,
            "label_assignment_mode": resolved.label_assignment_mode or self.label_assignment_mode,
        }
        self._require_paths(result, ["quality_metrics_json"])
        self._remember_run(result)
        self.last_result = result
        return result

    def export_to_labelstudio(self, context: RunContext | None = None) -> ToolResult:
        resolved = self._resolve_context(context)
        run_dir = Path(resolved.run_dir)
        labeled_csv = self._materialize_runtime_labeled_csv(run_dir)
        try:
            export = export_labelstudio_predictions_impl(
                input_path=str(labeled_csv),
                output_path=str(run_dir / "reports" / "labelstudio_import.json"),
                review_output_path=str(run_dir / "reports" / "labelstudio_review.json"),
                model_version=Path(self.model_path).name,
                local_files_document_root=self.labelstudio_document_root,
                confidence_threshold=self.confidence_threshold,
            )
        finally:
            if labeled_csv.exists():
                labeled_csv.unlink()
        labelstudio_config_path = run_dir / "reports" / "labelstudio_config.xml"
        labelstudio_config_path.write_text(
            self._build_labelstudio_config(self.classes or self._infer_classes_from_runtime()),
            encoding="utf-8",
        )
        result = {
            "run_dir": str(run_dir),
            "labelstudio_import_json": str(export["output_path"]),
            "labelstudio_review_json": str(export["review_output_path"]),
            "labelstudio_config_xml": str(labelstudio_config_path),
            "labelstudio_document_root": str(export["local_files_document_root"]),
            "object_prompts": resolved.object_prompts or self.object_prompts,
            "label_assignment_mode": resolved.label_assignment_mode or self.label_assignment_mode,
        }
        self._require_paths(result, ["labelstudio_import_json", "labelstudio_review_json", "labelstudio_config_xml"])
        self._remember_run(result)
        self.last_result = result
        return result

    def convert_labelstudio_export(
        self,
        export_path: str,
        context: RunContext | None = None,
        output_path: str | None = None,
    ) -> ToolResult:
        resolved = self._resolve_context(context)
        run_dir = Path(resolved.run_dir)
        object_labels = convert_labelstudio_export_to_object_labels_impl(
            export_path=export_path,
            output_path=output_path or str(run_dir / "reports" / AL_HUMAN_LABELS_FILE_NAME),
            local_files_document_root=self.labelstudio_document_root,
        )
        result = {
            "run_dir": str(run_dir),
            "human_labels_csv": str(object_labels["output_path"]),
            "labelstudio_export_path": str(object_labels["export_path"]),
            "labelstudio_document_root": str(object_labels["local_files_document_root"]),
        }
        self._require_paths(result, ["human_labels_csv"])
        self.last_result = result
        return result

    def run_pipeline(self, dataset_dir: str, task: str, human_labels_path: str | None = None) -> ToolResult:
        self.auto_label(dataset_dir, task_description=task)
        context = self.get_run_context()
        spec = self.generate_spec(task, context=context)
        quality = self.check_quality(human_labels_path=human_labels_path, context=context)
        exports = self.export_to_labelstudio(context=context)
        result = {
            "run_dir": context.run_dir,
            "labels_csv": context.labels_csv,
            "annotation_spec_md": spec["annotation_spec_md"],
            "quality_metrics_json": quality["quality_metrics_json"],
            "labelstudio_import_json": exports["labelstudio_import_json"],
            "labelstudio_review_json": exports["labelstudio_review_json"],
            "labelstudio_config_xml": exports["labelstudio_config_xml"],
            "classes": self.classes or [],
            "object_prompts": context.object_prompts or self.object_prompts,
            "label_assignment_mode": context.label_assignment_mode or self.label_assignment_mode,
            "task_mode": context.task_mode or self.task_mode,
        }
        self._require_paths(
            result,
            [
                "labels_csv",
                "annotation_spec_md",
                "quality_metrics_json",
                "labelstudio_import_json",
                "labelstudio_review_json",
                "labelstudio_config_xml",
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

    def _infer_classes_from_runtime(self) -> list[str]:
        if self.last_labeled_df is None or COLUMN_FOLDER_LABEL not in self.last_labeled_df.columns:
            return []
        return sorted({str(value).strip() for value in self.last_labeled_df[COLUMN_FOLDER_LABEL].dropna().tolist() if str(value).strip()})

    @staticmethod
    def _build_labelstudio_config(classes: list[str]) -> str:
        labels = "\n".join(f'    <Label value="{escape(class_name, quote=True)}"/>' for class_name in classes if class_name)
        return (
            "<View>\n"
            '  <Image name="image" value="$image" zoom="true"/>\n'
            '  <RectangleLabels name="label" toName="image">\n'
            f"{labels}\n"
            "  </RectangleLabels>\n"
            "</View>\n"
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
        labels_csv = result.get("labels_csv")
        if run_dir:
            self.last_run_dir = Path(run_dir)
        if labels_csv:
            self.last_labels_csv = Path(labels_csv)
        if self.last_run_dir:
            self.last_context = RunContext(
                run_dir=str(self.last_run_dir),
                labels_csv=str(self.last_labels_csv) if self.last_labels_csv else "",
                quality_metrics_json=str(result.get("quality_metrics_json") or ""),
                annotation_spec_md=str(result.get("annotation_spec_md") or ""),
                labelstudio_import_json=str(result.get("labelstudio_import_json") or ""),
                labelstudio_review_json=str(result.get("labelstudio_review_json") or ""),
                object_prompts=[str(item) for item in list(result.get("object_prompts") or self.object_prompts or [])],
                label_assignment_mode=str(result.get("label_assignment_mode") or self.label_assignment_mode),
                task_mode=str(result.get("task_mode") or self.task_mode),
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

    def _materialize_runtime_labeled_csv(self, run_dir: Path) -> Path:
        if self.last_labeled_df is None:
            raise RuntimeError(
                "Image-level annotation state is not available in memory. "
                "Run auto_label() in the current process before calling downstream annotation steps."
            )
        runtime_path = run_dir / "reports" / ".labeled_runtime.tmp.csv"
        self.last_labeled_df.to_csv(runtime_path, index=False)
        return runtime_path

    def _resolve_object_prompts(self, dataset_dir: Path, task_description: str = "") -> list[str]:
        if self.object_prompts:
            return self.object_prompts
        config_path = dataset_dir / "annotation_config.json"
        if config_path.exists():
            try:
                payload = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid annotation_config.json: {config_path}") from exc
            prompts = [str(item).strip() for item in list(payload.get("object_prompts", [])) if str(item).strip()]
            if prompts:
                return prompts
        raise RuntimeError(
            "Object prompts are required for image classification annotation. "
            "Pass object_prompts to AnnotationAgent or provide dataset_dir/annotation_config.json with object_prompts."
        )
