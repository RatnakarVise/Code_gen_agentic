from pathlib import Path
import re
import logging
from langchain_core.messages import HumanMessage
# from langchain.schema import HumanMessage
from agents.base_agent import BaseAgent

class ReportProgramAgent(BaseAgent):
    """
    Generates a final refined ABAP report program from the given section text.
    Performs:
    1. Initial ABAP generation
    2. Internal refinement
    3. Outputs only final refined ABAP code file
    """

    def run(self, section_text: str, metadata=None) -> Path:
        if not section_text:
            self.logger.warning("No text provided to ReportProgramAgent.")
            return Path()

        self.logger.info("Running ReportProgramAgent...")
        print("\n📝 [ReportProgramAgent] Input text preview:\n", section_text[:500], "\n--- END ---")

        # -----------------------
        # Step 1: Generate Draft
        # -----------------------
        draft_prompt = f"""
        You are a senior SAP ABAP developer.
        Generate a complete ABAP report program from the following requirement.
        Follow SAP best practices and include meaningful comments and proper modularization.
        Only return valid ABAP code (no markdown, no explanations).

        Requirement Context:
        {section_text}
        """

        resp_draft = self.llm.invoke([HumanMessage(content=draft_prompt)])
        draft_code = getattr(resp_draft, "content", str(resp_draft))
        draft_code = re.sub(r"```(?:abap)?|```", "", draft_code).strip()

        # -----------------------
        # Step 2: Refine Code
        # -----------------------
        refine_prompt = f"""
        You are an ABAP code reviewer.
        Refine the following ABAP report:
        - Fix syntax and indentation.
        - Use consistent variable naming (lv_, lt_, ls_).
        - Add comments before logic blocks.
        - Improve readability and align with SAP standards.
        Return only the final ABAP code.

        ABAP Code:
        {draft_code}
        """

        resp_refine = self.llm([HumanMessage(content=refine_prompt)])
        final_code = getattr(resp_refine, "content", str(resp_refine))
        final_code = re.sub(r"```(?:abap)?|```", "", final_code).strip()

        # -----------------------
        # Step 3: Save Final Output
        # -----------------------
        out_path = self.job_dir / "ReportProgram.txt"
        out_path.write_text(final_code, encoding="utf-8")

        self.logger.info(f"✅ Final ABAP report generated: {out_path}")
        return out_path
