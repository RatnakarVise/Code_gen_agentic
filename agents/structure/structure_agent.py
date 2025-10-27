from pathlib import Path
from langchain.schema import HumanMessage
from agents.base_agent import BaseAgent

class StructureAgent(BaseAgent):
    def run(self, section_text: str, metadata=None) -> Path:
        if not section_text:
            self.logger.warning("No text provided to StructureAgent.")
            return Path()

        self.logger.info("Running StructureAgent with provided section text...")
        print("\n🧩 [StructureAgent] Received text:\n", section_text[:500], "\n--- END ---")

        prompt = f"""
        You are an expert SAP ABAP developer. Using the following requirement/context, produce:
        1) An ABAP DDIC structure DDL Code which we can directly paste in eclipse and generate structure with field names/types and short comments
        2) A 'PURPOSE' section explaining the overall use of this structure.

        Context:
        {section_text}
        """
        resp = self.llm([HumanMessage(content=prompt)])
        text = getattr(resp, "content", str(resp)).strip()

        out_path = self.job_dir / "Structure.txt"
        out_path.write_text(text, encoding="utf-8")

        self.logger.info(f"StructureAgent output written to: {out_path}")
        return out_path
