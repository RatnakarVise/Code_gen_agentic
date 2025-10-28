import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from agents.base_agent import BaseAgent

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


class ReportProgramAgent(BaseAgent):
    """
    Generates and refines ABAP report programs using dynamic RAG context from a vector database.
    """

    def _init_llm(self):
        return ChatOpenAI(
            model_name=os.getenv("REPORT_MODEL_NAME", "gpt-5"),
            temperature=float(os.getenv("REPORT_TEMPERATURE", "0.1")),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )

    # --- Vector DB Setup ---
    def _init_vectorstore(self):
        """
        Loads or builds FAISS vector DB from the Report_RAG_KB.txt file.
        """
        kb_path = Path(os.path.dirname(__file__)) / "Report_RAG_KB.txt"
        vs_path = Path(os.path.dirname(__file__)) / "rag_vector_store"

        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # Load existing vector DB if available
        if vs_path.exists():
            self.logger.info("📚 Loading existing FAISS vector DB...")
            return FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

        # Otherwise build it fresh
        if not kb_path.exists():
            self.logger.warning(f"⚠️ No KB file found at {kb_path}. Proceeding without RAG context.")
            return None

        kb_text = kb_path.read_text(encoding="utf-8").strip()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(kb_text)]

        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(vs_path)
        self.logger.info("✅ New FAISS vector DB created and saved.")
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

    # --- Main Execution ---
    def run(self, section_text: str, purposes: dict | None = None, metadata=None) -> Path:
        if not section_text:
            self.logger.warning("No text provided to ReportProgramAgent.")
            return Path()

        full_context = section_text.strip()

        # Combine purposes text if provided
        if purposes:
            self.logger.info(f"🔍 Received purposes from previous agents: {list(purposes.keys())}")
            seen_texts = set()

            for key, text in purposes.items():
                if not text or not text.strip():
                    continue
                clean = text.strip()
                if clean in seen_texts:
                    continue
                seen_texts.add(clean)
                full_context += f"\n\nAdditional Context ({key}):\n{clean}"

        # --- Retrieve relevant RAG context ---
        rag_context = self._get_relevant_context(full_context)
        if rag_context:
            full_context += f"\n\n--- Retrieved Knowledge Base Context ---\n{rag_context}"

        self.logger.info("Running ReportProgramAgent...")

        # --- System role ---
        system_message = SystemMessage(
            content=(
                "You are a senior SAP ABAP developer assistant. "
                "Follow SAP best practices, naming conventions, and modularization principles. "
                "Always ensure correctness, readability, and maintainability in the generated ABAP code. "
                "Always use includes for data declaration and selection screen first and then main program logic."
                "Always use local classes and methods for modularization."
            )
        )

        # --- Step 1: Generate Draft ---
        draft_prompt = f"""
        Generate a complete ABAP report program based on the following requirements.
        Use modularization, meaningful comments, and correct syntax.
        Only return ABAP code (no markdown).
        If ALV Output - you must not miss any field mentioned.

        Requirement Context:
        {full_context}
        """

        resp_draft = self.llm.invoke([
            system_message,
            HumanMessage(content=draft_prompt)
        ])
        draft_code = getattr(resp_draft, "content", str(resp_draft))
        draft_code = re.sub(r"```(?:abap)?|```", "", draft_code).strip()

        # --- Step 2: Refine Code ---
        refine_prompt = f"""
        Review and improve the following ABAP program:
        - Ensure indentation and spacing are consistent.
        - Use lv_, lt_, ls_ prefixes.
        - Add comments for each logic block.
        - Optimize redundant logic and fix syntax issues.
        Return only the final refined ABAP code.

        ABAP Code:
        {draft_code}
        """

        resp_refine = self.llm.invoke([
            system_message,
            HumanMessage(content=refine_prompt)
        ])
        final_code = getattr(resp_refine, "content", str(resp_refine))
        final_code = re.sub(r"```(?:abap)?|```", "", final_code).strip()

        # --- Save Output ---
        # out_path = self.job_dir / "ReportProgram.txt"
        # out_path.write_text(final_code, encoding="utf-8")

        # self.logger.info(f"✅ Final ABAP report generated: {final_code}")
        return final_code
