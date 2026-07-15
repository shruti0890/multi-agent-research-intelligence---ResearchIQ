import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

from schemas import ProjectReportState
from agents.agent_research import fetch_arxiv_papers
from agents.agent_gap import cluster_and_analyze_gaps
from agents.agent_patent import search_and_classify_patents
from agents.agent_compile import compile_final_report

# Load environment variables
load_dotenv()

app = FastAPI(title="ResearchIQ Multi-Agent API", version="1.0")

# Serve frontend folder containing index.html UI
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if not os.path.exists(frontend_dir):
    os.makedirs(frontend_dir)
app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/", response_class=HTMLResponse)
def read_root():
    frontend_file_path = os.path.join(frontend_dir, "index.html")
    if not os.path.exists(frontend_file_path):
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    with open(frontend_file_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

# Enable CORS for Next.js/Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResearchRequest(BaseModel):
    topic: str
    max_results: int = 5

@app.post("/api/research", response_model=ProjectReportState)
def trigger_agent_pipeline(request: ResearchRequest):
    """
    Triggers the multi-agent pipeline sequentially:
    Agent 1 (Research) -> Agent 2 (Gaps) -> Agent 3 (Patents) -> Agent 4 (PDF Compiler)
    """
    topic = request.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic cannot be empty")
        
    print(f"\n[HTTP POST] Triggering agent pipeline for topic: '{topic}'")
    
    # 1. Agent 1: Research Paper Discovery
    print("Executing Agent 1: Research Paper Discovery...")
    research_out = fetch_arxiv_papers(topic, max_results=request.max_results)
    
    # 2. Agent 2: Gap Analysis
    print("Executing Agent 2: Gap Analysis & Synthesis...")
    gap_out = cluster_and_analyze_gaps(research_out)
    
    # 3. Agent 3: Patent Discovery
    print("Executing Agent 3: Patent Landscape Check...")
    patent_out = search_and_classify_patents(gap_out, topic)
    
    # 4. Construct Unified State
    state = ProjectReportState(
        topic=topic,
        research=research_out,
        gaps=gap_out,
        patents=patent_out
    )
    
    # 5. Agent 4: Compile PDF Report
    print("Executing Agent 4: PDF Generation...")
    # Save PDF in a temp/static directory inside the scratch workspace
    output_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        pdf_path = compile_final_report(state, output_dir=output_dir)
        filename = os.path.basename(pdf_path)
        state.pdf_filename = filename
        print(f"Pipeline completed. PDF generated at: {pdf_path}")
    except Exception as e:
        print(f"Error compiling PDF: {e}")
        state.pdf_filename = None
        
    return state

@app.get("/api/download/{filename}")
def download_pdf_report(filename: str):
    """
    Allows downloading the generated PDF report from the backend.
    """
    output_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(output_dir, filename)
    
    # Clean file path to prevent directory traversal vulnerabilities
    file_path = os.path.abspath(file_path)
    if not file_path.startswith(output_dir):
        raise HTTPException(status_code=403, detail="Access denied")
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "127.0.0.1")
    uvicorn.run("main:app", host=host, port=port, reload=True)
