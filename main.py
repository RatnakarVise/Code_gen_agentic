"""
main.py
FastAPI app + background job controller for modular AI agents.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# LangChain LLM
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI

# Local modules
from utils.file_utils import get_job_dir, zip_outputs
from utils.job_utils import split_sections
from agents.structure.structure_agent import StructureAgent
from agents.table.table_agent import TableAgent
from agents.report.report_program_agent import ReportProgramAgent

# ------------------------------ CONFIG ------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

app = FastAPI(title="SAP ABAP Code Generator (AI Agents)")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ai_agents")

# In-memory job store
jobs = {}

# ------------------------------ REQUEST MODEL ------------------------------
class RequirementPayload(BaseModel):
    REQUIREMENT: str


# ------------------------------ BACKGROUND JOB ------------------------------
def run_job(job_id: str, requirement_text: str):
    logger.info(f"Job {job_id} started")

    # Create job folder
    job_dir = get_job_dir()
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    # Initialize LLM
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )

    try:
        # --- Split Sections ---
        sections = split_sections(requirement_text)
        logger.info(f"[{job_id}] Parsed sections: {list(sections.keys())}")

        # -------------------- Section Mapping --------------------
        def get_section_text(prefix: str):
            """Combine parent and child sections by number prefix."""
            matched = [v for k, v in sections.items() if k.startswith(prefix)]
            return "\n\n".join(matched).strip()

        structure_text = get_section_text("5")  # Section 5 (Structure)
        table_text = get_section_text("4")      # Section 4 (Table)
        report_text = get_section_text("6")     # Section 6 (Report)

        logger.info(f"[{job_id}] Section 4 length: {len(table_text)}")
        logger.info(f"[{job_id}] Section 5 length: {len(structure_text)}")
        logger.info(f"[{job_id}] Section 6 length: {len(report_text)}")

        # -------------------- Run Agents --------------------
        structure_agent = StructureAgent(llm=llm, job_dir=job_dir)
        table_agent = TableAgent(llm=llm, job_dir=job_dir)
        report_agent = ReportProgramAgent(llm=llm, job_dir=job_dir)

        logger.info(f"[{job_id}] Running StructureAgent...")
        path_structure = structure_agent.run(structure_text)
        structure_result = path_structure.read_text(encoding="utf-8") if path_structure.exists() else ""

        logger.info(f"[{job_id}] Running TableAgent...")
        path_table = table_agent.run(table_text)
        table_result = path_table.read_text(encoding="utf-8") if path_table.exists() else ""

        logger.info(f"[{job_id}] Running ReportProgramAgent...")
        path_report = report_agent.run(report_text, metadata={
            "structure_text": structure_result,
            "table_text": table_result
        })

        # -------------------- Zip Results --------------------
        zip_path = zip_outputs(job_dir, [path_structure, path_table, path_report], job_id)
        logger.info(f"[{job_id}] Finished successfully. ZIP: {zip_path}")

        # Update job record
        jobs[job_id].update({
            "status": "finished",
            "finished_at": datetime.utcnow().isoformat(),
            "zip_path": str(zip_path),
            "outputs": {
                "structure": str(path_structure.name),
                "table": str(path_table.name),
                "report": str(path_report.name),
            },
        })

        # Log success
        logger.info(f"✅ Job {job_id} completed. File ready at: {zip_path}")

    except Exception as e:
        logger.exception(f"❌ Job {job_id} failed: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# ------------------------------ ENDPOINTS ------------------------------
@app.post("/generate")
def create_job(payload: RequirementPayload, background_tasks: BackgroundTasks):
    """Start job with unified document text (in REQUIREMENT key)."""
    requirement_text = payload.REQUIREMENT.strip()
    if not requirement_text:
        raise HTTPException(status_code=400, detail="REQUIREMENT text is missing or empty")

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "queued", "created_at": datetime.utcnow().isoformat()}

    background_tasks.add_task(run_job, job_id, requirement_text)
    logger.info(f"Job {job_id} queued")

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    """Check current job status and download if ready."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") == "finished":
        zip_path = Path(job["zip_path"])
        if not zip_path.exists():
            raise HTTPException(status_code=500, detail="ZIP file missing")

        # Automatically return ZIP once job is done
        return FileResponse(
            path=str(zip_path),
            filename=zip_path.name,
            media_type="application/zip",
            headers={
                "X-Job-ID": job_id,
                "X-Status": "finished",
            }
        )

    return JSONResponse(job)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
