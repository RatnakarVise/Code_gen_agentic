"""
main.py
FastAPI app + background job controller for modular AI agents.
"""

import os,io,zipfile
import uuid
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
# Local modules
from utils.file_utils import get_job_dir, zip_outputs
from utils.job_utils import split_sections
from agents.structure.structure_agent import StructureAgent
from agents.table.table_agent import TableAgent
from agents.report.report_program_agent import ReportProgramAgent
from agents.global_class.class_agent import ClassAgent
# ------------------------------ CONFIG ------------------------------
load_dotenv()
from utils.logger_config import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

app = FastAPI(title="SAP ABAP Code Generator (AI Agents)")
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# In-memory job store
jobs = {}


# ------------------------------ REQUEST MODEL ------------------------------
class RequirementPayload(BaseModel):
    REQUIREMENT: str


def run_job(job_id: str, requirement_text: str):
    logger.info(f"Job {job_id} started")

    job_dir = get_job_dir()
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        sections = split_sections(requirement_text)
        logger.info(f"[{job_id}] Parsed sections: {list(sections.keys())}")

        def get_section_text(prefix: str):
            matched = [v for k, v in sections.items() if k.startswith(prefix)]
            return "\n\n".join(matched).strip()

        structure_text = get_section_text("5")
        table_text = get_section_text("4")
        report_text = get_section_text("6")
        class_text = get_section_text("7")


        logger.info(f"[{job_id}] Section 4 length: {len(table_text)}")
        logger.info(f"[{job_id}] Section 5 length: {len(structure_text)}")
        logger.info(f"[{job_id}] Section 6 length: {len(report_text)}")
        logger.info(f"[{job_id}] Section 7 length: {len(class_text)}")

        # -------------------- Initialize --------------------
        structure_result = ""
        table_result = ""
        class_result = ""
        purposes = {}
        files_to_zip = []

        # -------------------- Run Structure Agent --------------------
        if structure_text:
            logger.info(f"[{job_id}] Running StructureAgent...")
            structure_agent = StructureAgent(job_dir=job_dir)
            structure_output = structure_agent.run(structure_text)
            
            structure_code = structure_output.get("code", "")
            structure_purpose = structure_output["purpose"]
            # structure_result = path_structure.read_text(encoding="utf-8") if path_structure.exists() else ""
            purposes["structure"] = structure_purpose
            if structure_code:
                files_to_zip.append(("structure.txt", structure_code))
            else:
                logger.warning(f"[{job_id}] StructureAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No structure section found — skipping StructureAgent.")

        # -------------------- Run Table Agent --------------------
        if table_text:
            logger.info(f"[{job_id}] Running TableAgent...")
            table_agent = TableAgent(job_dir=job_dir)
            table_output = table_agent.run(table_text)
            table_code = table_output.get("code", "")
            table_purpose = table_output.get("purpose", "")
            purposes["table"] = table_purpose

            if table_code:
                files_to_zip.append(("table.txt", table_code))
            else:
                logger.warning(f"[{job_id}] TableAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No table section found — skipping TableAgent.")
        # -------------------- Run Class Agent --------------------
        if class_text:
            logger.info(f"[{job_id}] Running ClassAgent...")
            class_agent = ClassAgent(job_dir=job_dir)
            class_output = class_agent.run(
                class_text,
                purposes=purposes,
                metadata={
                    "structure_text": structure_result,
                    "table_text": table_result,
                    "report_text": report_text
                }
            )
            
            class_code = class_output.get("code", "")
            class_purpose = class_output.get("purpose", "")
            purposes["class"] = class_purpose

            if class_code:
                files_to_zip.append(("class.txt", class_code))
            else:
                logger.warning(f"[{job_id}] ClassAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No class section found — skipping ClassAgent.")
            
        # -------------------- Run Report Agent --------------------
        if report_text:
            logger.info(f"[{job_id}] Running ReportProgramAgent...")
            report_agent = ReportProgramAgent(job_dir=job_dir)
            report_output = report_agent.run(
            report_text,
            purposes=purposes,
            metadata={"structure_text": structure_code, "table_text": table_code},
                )
            # Handle both return styles
            if isinstance(report_output, dict):
                report_code = report_output.get("code", "")
            else:
                report_code = str(report_output)

            if report_code:
                files_to_zip.append(("report.txt", report_code))
            else:
                logger.warning(f"[{job_id}] ReportProgramAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No report section found — skipping ReportProgramAgent.")

        

        # -------------------- Create In-Memory ZIP --------------------
        if not files_to_zip:
            raise ValueError("No valid sections found — no output generated.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_to_zip:
                zf.writestr(filename, content)
        zip_buffer.seek(0)

        # -------------------- Save Job Result (In Memory) --------------------
        jobs[job_id].update({
            "status": "finished",
            "finished_at": datetime.utcnow().isoformat(),
            "zip_bytes": zip_buffer.getvalue(),
            "outputs": [f[0] for f in files_to_zip],
        })

        logger.info(f"✅ Job {job_id} completed successfully (in-memory ZIP).")

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
    """Check current job status or download ZIP if finished (in-memory)."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")

    if status == "finished":
        # If in-memory ZIP is available
        if "zip_bytes" in job:
            zip_buffer = io.BytesIO(job["zip_bytes"])
            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{job_id}_results.zip"',
                    "X-Job-ID": job_id,
                    "X-Status": "finished"
                }
            )
        else:
            raise HTTPException(status_code=500, detail="ZIP bytes not found in memory")

    # Otherwise return live job status
    return JSONResponse(job)

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
