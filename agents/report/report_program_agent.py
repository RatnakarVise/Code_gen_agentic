import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent

load_dotenv()

class ReportProgramAgent(BaseAgent):
    """
    Generates a refined ABAP report program using prior agent outputs.
    """

    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("REPORT_MODEL_NAME", "gpt-5"),
            temperature=float(os.getenv("REPORT_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def run(self, section_text: str, purposes: dict | None = None, metadata=None) -> Path:
        if not section_text:
            self.logger.warning("No text provided to ReportProgramAgent.")
            return Path()

        full_context = section_text
        if purposes:
            self.logger.info(f"🔍 Received purposes from previous agents: {list(purposes.keys())}")
            for key, purpose_text in purposes.items():
                if purpose_text and purpose_text.strip():
                    preview = purpose_text.strip().replace("\n", " ")
                    self.logger.info(f"✅ Adding context from '{key}' → Preview: {preview}...")
                    full_context += f"\n\nAdditional Context (from {key} purpose):\n{purpose_text.strip()}"
                else:
                    self.logger.warning(f"⚠️ '{key}' purpose is empty or invalid.")

        self.logger.info("Running ReportProgramAgent...")
        print("\n📝 [ReportProgramAgent] Input text preview:\n", full_context[:500], "\n--- END ---")

        # --- Step 1: Generate Draft ---
        draft_prompt = f"""
        You are a senior SAP ABAP developer.
        Generate a complete ABAP report program from the following requirement.
        Follow SAP best practices, modularization, and comments.
        Only return ABAP code (no markdown, no explanations).

        Requirement Context:
        {full_context}
        """

        resp_draft = self.llm.invoke([HumanMessage(content=draft_prompt)])
        draft_code = getattr(resp_draft, "content", str(resp_draft))
        draft_code = re.sub(r"```(?:abap)?|```", "", draft_code).strip()

        # --- Step 2: Refine Code ---
        refine_prompt = f"""
        You are an ABAP reviewer.
        Refine this report:
        - Fix syntax and indentation
        - Use naming lv_/lt_/ls_
        - Add comments before logic
        - Ensure SAP best practices
        Return only final ABAP code.

        ABAP Code:
        {draft_code}
        """

        resp_refine = self.llm.invoke([HumanMessage(content=refine_prompt)])
        final_code = getattr(resp_refine, "content", str(resp_refine))
        final_code = re.sub(r"```(?:abap)?|```", "", final_code).strip()

        out_path = self.job_dir / "ReportProgram.txt"
        out_path.write_text(final_code, encoding="utf-8")

        self.logger.info(f"✅ Final ABAP report generated: {out_path}")
        return out_path
