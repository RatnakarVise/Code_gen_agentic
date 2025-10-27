from pathlib import Path
from langchain_core.messages import HumanMessage
# from langchain.schema import HumanMessage
from agents.base_agent import BaseAgent

class TableAgent(BaseAgent):
    def run(self, section_text: str, metadata=None) -> Path:
        if not section_text:
            self.logger.warning("No text provided to TableAgent.")
            return Path()

        self.logger.info("Running TableAgent with provided section text...")
        print("\n📊 [TableAgent] Received text:\n", section_text[:500], "\n--- END ---")

        prompt = f"""
        You are an expert SAP ABAP data modeler. Using the following requirement/context, produce:
        1) An ABAP transparent table definition (DDIC) DDL Code which we can directly paste in eclipse and generate table with field names/types and comments.
        2) A 'PURPOSE' section explaining what this table is used for and key relationships.

        Context:
        {section_text}
        """
        resp = self.llm([HumanMessage(content=prompt)])
        text = getattr(resp, "content", str(resp)).strip()

        out_path = self.job_dir / "Table.txt"
        out_path.write_text(text, encoding="utf-8")

        self.logger.info(f"TableAgent output written to: {out_path}")
        return out_path
