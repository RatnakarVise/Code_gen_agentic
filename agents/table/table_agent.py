import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class TableAgent(BaseAgent):
    """
    Generates ABAP transparent table DDL code using dynamic RAG context from vector database.
    """

    # ------------------ LLM INIT ------------------
    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("TABLE_MODEL_NAME", "gpt-4.1"),
            temperature=float(os.getenv("TABLE_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # ------------------ VECTORSTORE INIT ------------------
    def _init_vectorstore(self):
        """
        Loads or builds FAISS vector DB from Table_RAG_KB.txt file.
        """
        kb_path = Path(os.path.dirname(__file__)) / "Table_RAG_KB.txt"
        vs_path = Path(os.path.dirname(__file__)) / "table_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        if vs_path.exists():
            self.logger.info("📚 Loading existing FAISS vector DB for TableAgent...")
            return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

        if not kb_path.exists():
            self.logger.warning(f"⚠️ No KB file found at {kb_path}. Proceeding without RAG context.")
            return None

        kb_text = kb_path.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(kb_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created for TableAgent.")
        return vectorstore

    def _get_relevant_context(self, query: str, k: int = 4) -> str:
        """
        Retrieves top-k relevant KB chunks from vector DB for given query.
        """
        if not hasattr(self, "vectorstore") or self.vectorstore is None:
            self.vectorstore = self._init_vectorstore()
        if not self.vectorstore:
            return ""

        results = self.vectorstore.similarity_search(query, k=k)
        if not results:
            return ""
        combined = "\n\n".join([r.page_content for r in results])
        self.logger.info(f"📖 Retrieved {len(results)} RAG context chunks.")
        return combined

    # ------------------ MAIN RUN METHOD ------------------
    def run(self, section_text: str, metadata=None) -> dict:
        if not section_text:
            self.logger.warning("No text provided to TableAgent.")
            return {"type": "table", "purpose": "", "path": Path()}

        self.logger.info("Running TableAgent with provided section text...")
        print("\n📊 [TableAgent] Received text:\n", section_text[:500], "\n--- END ---")

        # --- Retrieve relevant RAG context ---
        rag_context = self._get_relevant_context(section_text)
        full_context = section_text.strip()
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        # --- System message ---
        system_message = SystemMessage(
            content=(
                "You are a senior SAP ABAP Data Modeler. "
                "Follow SAP best practices for transparent table creation, including naming conventions, "
                "key fields, technical settings, and field documentation. "
                "Always generate clean, valid, and well-commented ABAP DDL code for tables."
            )
        )

        # --- Prompt to LLM ---
        prompt = f"""
        Using the following requirement/context, produce JSON with two keys:
        1) "table_code": ABAP transparent table DDL code that can be pasted directly into Eclipse 
           to generate the table with field names, data types, keys, and short comments.
        2) "table_purpose": A short explanation of what this table stores and its key relationships 
           with table name and its all fields.

        Requirement Context:
        {full_context}

        Output format (strict JSON only, no markdown, no commentary):
        {{
            "table_code": "...",
            "table_purpose": "..."
        }}
        """

        # --- Get response ---
        resp = self.llm.invoke([system_message, HumanMessage(content=prompt)])
        raw = getattr(resp, "content", str(resp)).strip()

        # --- Parse JSON safely ---
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError("No JSON object found in LLM response.")
            data = json.loads(match.group())
        except Exception as e:
            self.logger.error(f"Failed to parse JSON from LLM output: {e}")
            self.logger.debug(f"Raw LLM output:\n{raw}")
            raise

        # --- Extract values ---
        table_code = data.get("table_code", "").strip()
        table_purpose = data.get("table_purpose", "").strip()

        # --- Save result ---
        out_path = self.job_dir / "Table.txt"
        out_path.write_text(table_code, encoding="utf-8")

        self.logger.info(f"✅ TableAgent outputs written to: {out_path}")
        self.logger.debug(f"Extracted Table Purpose: {table_purpose[:200]}...")

        return {
            "type": "table",
            "purpose": table_purpose,
            "path": out_path
        }
