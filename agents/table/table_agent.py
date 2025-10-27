import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent

load_dotenv()

class TableAgent(BaseAgent):
    """
    Generates ABAP transparent table DDL code and extracts its purpose.
    """

    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("TABLE_MODEL_NAME", "gpt-5"),
            temperature=float(os.getenv("TABLE_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def run(self, section_text: str, metadata=None) -> dict:
        if not section_text:
            self.logger.warning("No text provided to TableAgent.")
            return {"type": "table", "purpose": "", "path": Path()}

        self.logger.info("Running TableAgent with provided section text...")
        print("\n📊 [TableAgent] Received text:\n", section_text[:500], "\n--- END ---")

        prompt = f"""
        You are an expert SAP ABAP data modeler. Using the following requirement/context, 
        produce and return JSON with two keys:
        1) "table_code": ABAP transparent table DDL code that can be pasted directly into Eclipse 
           to generate the table with field names, data types, and short comments.
        2) "table_purpose": A short explanation of what this table stores and its key relationships.

        Context:
        {section_text}

        Output format (strict JSON only, no markdown, no commentary):
        {{
            "table_code": "...",
            "table_purpose": "..."
        }}
        """

        resp = self.llm.invoke([HumanMessage(content=prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")
            data = json.loads(match.group())
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from LLM output: {e}")
            self.logger.debug(f"Raw LLM output:\n{raw}")
            raise

        table_code = data.get("table_code", "").strip()
        table_purpose = data.get("table_purpose", "").strip()

        out_path = self.job_dir / "Table.txt"
        out_path.write_text(table_code, encoding="utf-8")

        self.logger.info(f"✅ TableAgent outputs written to: {out_path}")
        self.logger.debug(f"Extracted Table Purpose: {table_purpose[:200]}...")

        return {
            "type": "table",
            "purpose": table_purpose,
            "path": out_path
        }
