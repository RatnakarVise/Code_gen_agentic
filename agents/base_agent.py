from pathlib import Path
import logging

class BaseAgent:
    def __init__(self, llm, job_dir: Path):
        self.llm = llm
        self.job_dir = job_dir
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, section_text: str, metadata=None):
        raise NotImplementedError("Each agent must implement its own run()")
