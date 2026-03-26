from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_AGENT_ROOT = PROJECT_ROOT / "dataset-agent"
if str(DATASET_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_AGENT_ROOT))

from dotenv import load_dotenv  # noqa: E402

from annotation_agent import AnnotationAgent  # noqa: E402
from data_quality_tools_agent import ToolBasedDataQualityAgent  # noqa: E402
# from dataset-agent?  # noqa
