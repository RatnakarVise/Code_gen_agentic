import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.base_agent import BaseAgent

load_dotenv()

class StructureAgent(BaseAgent):
    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("STRUCTURE_MODEL_NAME", "gpt-5"),
            temperature=float(os.getenv("STRUCTURE_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    def run(self, section_text: str, metadata=None) -> dict:
        if not section_text:
            self.logger.warning("No text provided to StructureAgent.")
            return {"purpose": "", "path": Path(), "type": "structure"}

        self.logger.info("Running StructureAgent with provided section text...")
        print("\n🧩 [StructureAgent] Received text:\n", section_text[:500], "\n--- END ---")

        prompt = f"""
        You are an expert SAP ABAP developer. Using the following requirement/context,
        produce and return JSON with two keys:
        1) "structure_code":  An ABAP DDIC structure DDL code which we can directly paste
           in Eclipse to generate the structure with field names, data types, and short comments.
        2) "structure_purpose": A short 'PURPOSE' explaining its use.

        Context:
        {section_text}

        Output format (strict JSON only, no markdown, no commentary):
        {{
            "structure_code": "...",
            "structure_purpose": "..."
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

        structure_code = data.get("structure_code", "").strip()
        structure_purpose = data.get("structure_purpose", "").strip()

        out_path = self.job_dir / "Structure.txt"
        out_path.write_text(structure_code, encoding="utf-8")

        self.logger.info(f"✅ StructureAgent outputs written to: {out_path}")

        return {
            "purpose": structure_purpose,
            "path": out_path,
            "type": "structure"
        }
