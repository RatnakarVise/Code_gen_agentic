"""
main.py
FastAPI app + background job controller for modular AI agents.
"""

import os
import io,re
import zipfile
import uuid
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Local modules
from utils.file_utils import get_job_dir
from utils.job_utils import split_sections
from agents.structure.structure_agent import StructureAgent
from agents.table.table_agent import TableAgent
from agents.global_class.class_agent import ClassAgent
from agents.report.report_program_agent import ReportProgramAgent
from agents.CDS.cds_agent import CdsAgent
from agents.value_help.value_help_agent import ValueHelpAgent
from agents.FM.fm_agent import FmAgent

# ------------------------------ CONFIG ------------------------------
from utils.logger_config import setup_logger
load_dotenv()
setup_logger()
logger = logging.getLogger(__name__)

app = FastAPI(title="SAP ABAP Code Generator (AI Agents)")

# In-memory job store
jobs = {}

# ------------------------------ REQUEST MODEL ------------------------------
class RequirementPayload(BaseModel):
    REQUIREMENT: str

def is_na(text: str) -> bool:
    """
    Return True if the section text is:
    - Empty or whitespace,
    - 'N/A' or 'NA' (case-insensitive),
    - Or has 25 or fewer characters (too short to be meaningful).
    """
    cleaned = text.strip().lower()
    if not cleaned or cleaned in {"n/a", "na"}:
        return True
    return len(cleaned) <= 25

# ------------------------------ BACKGROUND JOB ------------------------------
def run_job(job_id: str, requirement_text: str):
    logger.info(f"Job {job_id} started")

    job_dir = get_job_dir()
    jobs[job_id]["status"] = "running"
    jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

    try:
        # --- Parse sections from document ---
        sections = split_sections(requirement_text)

        def get_section_text(prefix: str) -> str:
            """
            Returns text for the requested section number or title.
            - Matches either exact number (e.g., "7") or full title (e.g., "7 global class design").
            - Case-insensitive.
            """
            prefix = prefix.strip().lower()
            matched = [
                v for k, v in sorted(sections.items())
                if k.lower() == prefix or k.lower().startswith(prefix + " ")
            ]

            if not matched:
                return sections.get(prefix, "").strip()

            return "\n\n".join(matched).strip()

        # Extract relevant sections
        structure_text = get_section_text("3")   # Structure / types
        table_text = get_section_text("4")       # Data model / table design
        value_help_text = get_section_text("5")  # Value help CDS view
        cds_text = get_section_text("6")         # CDS view design
        fm_text = get_section_text("7")          # Function module design
        class_text = get_section_text("8")       # Global class design
        # report_text = get_section_text("9")      # Report logic
        report_text = "\n\n".join([
            get_section_text("1"),  # Purpose or Introduction
            get_section_text("2"),  # Scope or Requirements
            get_section_text("9"),  # Report logic
        ]).strip()

        
        logger.info(f"[{job_id}] Section 3 length: {len(structure_text)}")
        logger.info(f"[{job_id}] Section 4 length: {len(table_text)}")
        logger.info(f"[{job_id}] Section 5 length: {len(value_help_text)}")
        logger.info(f"[{job_id}] Section 6 length: {len(cds_text)}")
        logger.info(f"[{job_id}] Section 7 length: {len(fm_text)}")
        logger.info(f"[{job_id}] Section 8 length: {len(class_text)}")
        logger.info(f"[{job_id}] Section 9 length: {len(report_text)}")

        # --- Run AI agents ---
        structure_result = ""
        table_result = ""
        class_result = ""
        cds_result = ""
        purposes = {}
        files_to_zip = []

        # -------------------- Structure Agent --------------------
        if structure_text and not is_na(structure_text):
            logger.info(f"[{job_id}] Running StructureAgent...")
            structure_agent = StructureAgent(job_dir=job_dir)
            structure_output = structure_agent.run(structure_text)
            structure_code = structure_output.get("code", "")
            purposes["structure"] = structure_output.get("purpose", "")

            if structure_code:
                files_to_zip.append(("structure.txt", structure_code))
            else:
                logger.warning(f"[{job_id}] StructureAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No structure section found — skipping StructureAgent.")

        # -------------------- Table Agent --------------------
        if table_text and not is_na(table_text):
            logger.info(f"[{job_id}] Running TableAgent...")
            table_agent = TableAgent(job_dir=job_dir)
            table_output = table_agent.run(table_text)
            table_code = table_output.get("code", "")
            purposes["table"] = table_output.get("purpose", "")

            if table_code:
                files_to_zip.append(("table.txt", table_code))
            else:
                logger.warning(f"[{job_id}] TableAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No table section found — skipping TableAgent.")

        # -------------------- Value Help Agent --------------------
        if value_help_text and not is_na(value_help_text):
            logger.info(f"[{job_id}] Running ValueHelpAgent...")
            value_agent = ValueHelpAgent(job_dir=job_dir)
            value_output = value_agent.run(
                value_help_text
            )
            value_help_code = value_output.get("code", "")
            purposes["value_help"] = value_output.get("purpose", "")

            if value_help_code:
                files_to_zip.append(("value_help.txt", value_help_code))
            else:
                logger.warning(f"[{job_id}] ValueHelpAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No Value Help section found — skipping ValueHelpAgent.")
        
        value_help_entity = None

        if value_help_code:
            # Optional: Extract entity name from the code (e.g., "define view entity ZCDS_VENDOR_VH")
            match = re.search(r"define\s+view\s+entity\s+(\w+)", value_help_code, re.IGNORECASE)
            if match:
                value_help_entity = match.group(1)

        # -------------------- CDS Agent --------------------
        if cds_text and not is_na(cds_text):
            logger.info(f"[{job_id}] Running CDSAgent...")
            cds_agent = CdsAgent(job_dir=job_dir)
            metadata = {}
            if value_help_entity:
                metadata["value_help_entity"] = value_help_entity
            if purposes.get("value_help"):
                metadata["value_help_purpose"] = purposes["value_help"]
            cds_output = cds_agent.run(
                cds_text,
                metadata=metadata if metadata else None,
            )
            cds_code = cds_output.get("code", "")
            purposes["cds"] = cds_output.get("purpose", "")

            if cds_code:
                files_to_zip.append(("cds.txt", cds_code))
            else:
                logger.warning(f"[{job_id}] CdsAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No cds section found — skipping CdsAgent.")

        # -------------------- FM Agent --------------------
        if fm_text and not is_na(fm_text):
            logger.info(f"[{job_id}] Running ClassAgent...")
            fm_agent = FmAgent(job_dir=job_dir)
            fm_output = fm_agent.run(
                fm_text,
                purposes=purposes,
            )
            fm_code = fm_output.get("code", "")
            purposes["fm"] = fm_output.get("purpose", "")

            if fm_code:
                files_to_zip.append(("fm.txt", fm_code))
            else:
                logger.warning(f"[{job_id}] FmAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No fm section found — skipping FmAgent.")

        # -------------------- Class Agent --------------------
        if class_text and not is_na(class_text):
            logger.info(f"[{job_id}] Running ClassAgent...")
            class_agent = ClassAgent(job_dir=job_dir)
            class_output = class_agent.run(
                class_text,
                purposes=purposes,
            )
            class_code = class_output.get("code", "")
            purposes["class"] = class_output.get("purpose", "")

            if class_code:
                files_to_zip.append(("class.txt", class_code))
            else:
                logger.warning(f"[{job_id}] ClassAgent returned empty code.")
        else:
            logger.info(f"[{job_id}] No class section found — skipping ClassAgent.")

        # -------------------- Report Agent  --------------------
        if report_text and not is_na(report_text):
            logger.info(f"[{job_id}] Running ReportProgramAgent...")
            metadata = {}
            if "structure_code" in locals() and structure_code:
                metadata["structure_text"] = structure_code
            if "table_code" in locals() and table_code:
                metadata["table_text"] = table_code
            if "class_code" in locals() and class_code:
                metadata["class_text"] = class_code

            report_agent = ReportProgramAgent(job_dir=job_dir)
            report_output = report_agent.run(
                report_text,
                purposes=purposes,
                metadata=metadata if metadata else None,
            )
            report_code = report_output.get("code", "") if isinstance(report_output, dict) else str(report_output)
            if report_code:
                files_to_zip.append(("report.txt", report_code))
        else:
            logger.info(f"[{job_id}] No report section found — skipping ReportProgramAgent.")

        del structure_text, table_text, value_help_text, cds_text, class_text, report_text

        # -------------------- Finalize ZIP --------------------
        if not files_to_zip:
            raise ValueError("No valid sections found — no output generated.")

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_to_zip:
                zf.writestr(filename, content)
        zip_buffer.seek(0)

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


@app.get("/generate/{job_id}")
def job_status(job_id: str):
    """Check job status or download ZIP if finished."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") == "finished":
        if "zip_bytes" not in job:
            raise HTTPException(status_code=500, detail="ZIP bytes not found in memory")

        zip_buffer = io.BytesIO(job["zip_bytes"])
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{job_id}_results.zip"',
                "X-Job-ID": job_id,
                "X-Status": "finished",
            },
        )

    return JSONResponse(job)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}
