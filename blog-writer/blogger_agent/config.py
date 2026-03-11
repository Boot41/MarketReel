# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass

import google.auth
from dotenv import load_dotenv

# Load local environment variables (for example from .env).
load_dotenv()

# Prefer Gemini API key from .env when present.
if os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")

# Configure Vertex AI defaults only when Vertex mode is enabled.
if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true":
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        _, project_id = google.auth.default()
        if project_id:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)


@dataclass
class ResearchConfiguration:
    """Configuration for research-related models and parameters.

    Attributes:
        critic_model (str): Model for evaluation tasks.
        worker_model (str): Model for working/generation tasks.
        max_search_iterations (int): Maximum search iterations allowed.
    """

    critic_model: str = "gemini-2.5-pro"
    worker_model: str = "gemini-2.5-flash"
    max_search_iterations: int = 5


config = ResearchConfiguration()
