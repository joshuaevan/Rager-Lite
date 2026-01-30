#!/usr/bin/env python3
"""
Streamlit Web UI for RAG Document Processor

Run with: streamlit run web_ui.py
"""

import streamlit as st
import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import sys

from dotenv import load_dotenv, set_key

# Load environment variables from .env file
load_dotenv()

# Page config
st.set_page_config(
    page_title="Rager - Lite",
    page_icon="🎸",
    layout="wide",
)

# Initialize session state
if "processing" not in st.session_state:
    st.session_state.processing = False
if "last_run" not in st.session_state:
    st.session_state.last_run = None


def get_saved_api_key() -> str:
    """Get API key from environment or .env file."""
    return os.environ.get("OPENAI_API_KEY", "")


def save_api_key(api_key: str):
    """Save API key to .env file."""
    env_path = Path(__file__).parent / ".env"
    
    # Create .env if it doesn't exist
    if not env_path.exists():
        env_path.write_text("# Rager Environment Configuration\n")
    
    set_key(str(env_path), "OPENAI_API_KEY", api_key)
    os.environ["OPENAI_API_KEY"] = api_key


def get_runs(output_path: str) -> list:
    """Get list of previous runs."""
    output_dir = Path(output_path)
    if not output_dir.exists():
        return []
    
    runs = []
    for folder in sorted(output_dir.iterdir(), reverse=True):
        if folder.is_dir() and folder.name.startswith("run_"):
            manifest_path = folder / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                runs.append({
                    "name": folder.name,
                    "path": str(folder),
                    "timestamp": manifest.get("generated_at", ""),
                    "stats": manifest.get("stats", {}),
                    "files": manifest.get("files", []),
                    "openai_usage": manifest.get("openai_usage", {}),
                })
    return runs


def get_input_files(input_path: str) -> list:
    """Get list of files in input directory."""
    input_dir = Path(input_path)
    if not input_dir.exists():
        return []
    
    files = []
    for f in input_dir.rglob("*"):
        if f.is_file() and f.name != ".gitkeep":
            files.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "type": f.suffix.lower(),
            })
    return files


def run_processor(config: dict) -> str:
    """Run the document processor."""
    cmd = [sys.executable, "process_docs.py"]
    
    if config.get("input_path"):
        cmd.extend(["--input", config["input_path"]])
    if config.get("output_path"):
        cmd.extend(["--output", config["output_path"]])
    if config.get("output_format"):
        cmd.extend(["--format", config["output_format"]])
    if config.get("no_chunk"):
        cmd.append("--no-chunk")
    if config.get("use_openai"):
        cmd.append("--use-openai")
    if config.get("api_key"):
        cmd.extend(["--api-key", config["api_key"]])
    if config.get("ai_types"):
        cmd.extend(["--ai-types", ",".join(config["ai_types"])])
    if config.get("ai_batch_size"):
        cmd.extend(["--ai-batch-size", str(config["ai_batch_size"])])
    if config.get("clear_cache"):
        cmd.append("--clear-cache")
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))
    return result.stdout + result.stderr


def create_download_zip(run_path: str) -> bytes:
    """Create a zip file of the knowledge folder."""
    import io
    import zipfile
    
    knowledge_dir = Path(run_path) / "knowledge"
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in knowledge_dir.iterdir():
            if file.is_file():
                zip_file.write(file, file.name)
        
        # Also include instructions.md
        instructions_path = Path(run_path) / "instructions.md"
        if instructions_path.exists():
            zip_file.write(instructions_path, "instructions.md")
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# Main UI
st.title("Rager - Lite")
st.markdown("**RAG Document Processor** - It's a freakin Rager bro!")
st.caption("Created by Josh Holtzclaw")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    input_path = st.text_input("Input Path", value="./input", help="Folder containing source documents")
    output_path = st.text_input("Output Path", value="./output", help="Folder for processed output")
    
    st.subheader("Processing Options")
    
    output_format = st.selectbox(
        "Output Format",
        options=["txt", "jsonl", "both"],
        index=0,
        help="Format for output files"
    )
    
    chunk_enabled = st.checkbox("Enable Chunking", value=True, help="Split documents into smaller chunks")
    
    if chunk_enabled:
        chunk_size = st.slider("Chunk Size", min_value=256, max_value=4096, value=1024, step=128)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=512, value=200, step=50)
    
    st.subheader("File Types")
    col1, col2 = st.columns(2)
    with col1:
        ft_pdf = st.checkbox("PDF", value=True)
        ft_docx = st.checkbox("DOCX", value=True)
        ft_csv = st.checkbox("CSV", value=True)
    with col2:
        ft_txt = st.checkbox("TXT", value=True)
        ft_html = st.checkbox("HTML", value=True)
        ft_md = st.checkbox("MD", value=True)
    
    st.subheader("OpenAI Integration (Optional)")
    
    # Check for saved API key
    saved_key = get_saved_api_key()
    has_saved_key = bool(saved_key)
    
    use_openai = st.checkbox(
        "Use OpenAI for descriptions", 
        value=has_saved_key, 
        help="Generate AI-powered file descriptions"
    )
    api_key = ""
    ai_types = []
    ai_batch_size = 10
    clear_cache = False
    
    if use_openai:
        if has_saved_key:
            st.success("API key loaded from .env file")
            api_key = saved_key
            show_key = st.checkbox("Show/Edit API key", value=False)
            if show_key:
                new_key = st.text_input(
                    "OpenAI API Key", 
                    value=saved_key,
                    type="password",
                    help="Edit to update saved key"
                )
                if new_key != saved_key:
                    if st.button("Save New Key"):
                        save_api_key(new_key)
                        st.success("API key saved to .env file!")
                        st.rerun()
                    api_key = new_key
        else:
            st.info("Enter your API key below. It will be saved to .env for future use.")
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password", 
                help="Will be saved to .env file"
            )
            if api_key:
                if st.button("Save API Key"):
                    save_api_key(api_key)
                    st.success("API key saved to .env file!")
                    st.rerun()
        
        st.markdown("**AI-enabled file types:**")
        ai_col1, ai_col2 = st.columns(2)
        with ai_col1:
            ai_pdf = st.checkbox("PDF", value=True, key="ai_pdf")
            ai_docx = st.checkbox("DOCX", value=True, key="ai_docx")
        with ai_col2:
            ai_html = st.checkbox("HTML", value=True, key="ai_html")
            ai_md = st.checkbox("MD", value=True, key="ai_md")
        
        if ai_pdf: ai_types.append("pdf")
        if ai_docx: ai_types.append("docx")
        if ai_html: ai_types.append("html")
        if ai_md: ai_types.append("md")
        
        ai_batch_size = st.slider("Batch Size", min_value=1, max_value=20, value=10, 
                                   help="Files per API call (higher = fewer calls, lower cost)")
        
        clear_cache = st.checkbox("Clear description cache", value=False, 
                                   help="Force regenerate all descriptions")

# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Process", "Previous Runs", "Input Files"])

with tab1:
    st.header("Process Documents")
    
    # Show input summary
    input_files = get_input_files(input_path)
    if input_files:
        st.success(f"Found {len(input_files)} files in input folder")
        
        with st.expander("View input files"):
            for f in input_files:
                size_kb = f["size"] / 1024
                st.text(f"{f['name']} ({size_kb:.1f} KB) - {f['type']}")
    else:
        st.warning(f"No files found in {input_path}. Add documents to process.")
    
    st.markdown("---")
    
    # Process button
    if st.button("Process Documents", type="primary", disabled=len(input_files) == 0):
        with st.spinner("Processing documents..."):
            config = {
                "input_path": input_path,
                "output_path": output_path,
                "output_format": output_format,
                "no_chunk": not chunk_enabled,
                "use_openai": use_openai,
                "api_key": api_key if use_openai else "",
                "ai_types": ai_types if use_openai else [],
                "ai_batch_size": ai_batch_size if use_openai else 10,
                "clear_cache": clear_cache if use_openai else False,
            }
            
            output = run_processor(config)
            
            st.text_area("Processing Output", output, height=300)
            
            # Get latest run
            runs = get_runs(output_path)
            if runs:
                st.session_state.last_run = runs[0]
                st.success(f"Processing complete! Output saved to: {runs[0]['path']}")
                st.rerun()
    
    # Show last run results
    if st.session_state.last_run:
        st.markdown("---")
        st.subheader("Latest Run Results")
        
        run = st.session_state.last_run
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", run["stats"].get("documents_processed", 0))
        with col2:
            st.metric("Chunks", run["stats"].get("chunks_created", 0))
        with col3:
            st.metric("Output Files", run["stats"].get("output_files", 0))
        
        # Show OpenAI token usage if available
        openai_usage = run.get("openai_usage", {})
        if openai_usage and openai_usage.get("total_tokens"):
            st.markdown("**OpenAI Usage:**")
            token_col1, token_col2, token_col3, token_col4 = st.columns(4)
            with token_col1:
                st.metric("Total Tokens", f"{openai_usage.get('total_tokens', 0):,}")
            with token_col2:
                st.metric("Prompt", f"{openai_usage.get('prompt_tokens', 0):,}")
            with token_col3:
                st.metric("Completion", f"{openai_usage.get('completion_tokens', 0):,}")
            with token_col4:
                st.metric("Cached Files", openai_usage.get('cached_files', 0))
            
            # Estimate cost (gpt-5-mini pricing)
            prompt_cost = openai_usage.get('prompt_tokens', 0) * 0.25 / 1_000_000
            completion_cost = openai_usage.get('completion_tokens', 0) * 2.00 / 1_000_000
            total_cost = prompt_cost + completion_cost
            st.caption(f"Estimated cost: ${total_cost:.4f} (gpt-5-mini pricing)")
        
        # Download button
        if Path(run["path"]).exists():
            zip_data = create_download_zip(run["path"])
            st.download_button(
                label="Download Knowledge Files (ZIP)",
                data=zip_data,
                file_name=f"{run['name']}_knowledge.zip",
                mime="application/zip",
            )
        
        # Show instructions
        instructions_path = Path(run["path"]) / "instructions.md"
        if instructions_path.exists():
            with st.expander("View Generated Instructions"):
                st.markdown(instructions_path.read_text(encoding="utf-8"))

with tab2:
    st.header("Previous Runs")
    
    runs = get_runs(output_path)
    
    if not runs:
        st.info("No previous runs found.")
    else:
        for run in runs:
            # Show token count in header if available
            openai_usage = run.get("openai_usage", {})
            tokens_str = ""
            if openai_usage and openai_usage.get("total_tokens"):
                tokens_str = f" | {openai_usage['total_tokens']:,} tokens"
            
            with st.expander(f"{run['name']} - {run['stats'].get('documents_processed', 0)} docs{tokens_str}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Documents", run["stats"].get("documents_processed", 0))
                with col2:
                    st.metric("Chunks", run["stats"].get("chunks_created", 0))
                with col3:
                    st.metric("Files", run["stats"].get("output_files", 0))
                
                # Show OpenAI usage if available
                if openai_usage and openai_usage.get("total_tokens"):
                    st.markdown("**OpenAI Usage:**")
                    st.text(f"  Total tokens: {openai_usage.get('total_tokens', 0):,}")
                    st.text(f"  Prompt: {openai_usage.get('prompt_tokens', 0):,} | Completion: {openai_usage.get('completion_tokens', 0):,}")
                    st.text(f"  API calls: {openai_usage.get('api_calls', 0)} | Cached: {openai_usage.get('cached_files', 0)}")
                    # Estimate cost
                    prompt_cost = openai_usage.get('prompt_tokens', 0) * 0.25 / 1_000_000
                    completion_cost = openai_usage.get('completion_tokens', 0) * 2.00 / 1_000_000
                    st.text(f"  Est. cost: ${prompt_cost + completion_cost:.4f}")
                
                st.markdown(f"**Timestamp:** {run['timestamp']}")
                st.markdown(f"**Path:** `{run['path']}`")
                
                # File list
                st.markdown("**Files:**")
                for f in run["files"]:
                    st.text(f"  - {f.get('file', 'unknown')}")
                
                # Download button
                if Path(run["path"]).exists():
                    zip_data = create_download_zip(run["path"])
                    st.download_button(
                        label="Download",
                        data=zip_data,
                        file_name=f"{run['name']}_knowledge.zip",
                        mime="application/zip",
                        key=f"download_{run['name']}",
                    )
                
                # View instructions
                instructions_path = Path(run["path"]) / "instructions.md"
                if instructions_path.exists():
                    if st.button(f"View Instructions", key=f"instr_{run['name']}"):
                        st.markdown(instructions_path.read_text(encoding="utf-8"))

with tab3:
    st.header("Input Files")
    
    input_files = get_input_files(input_path)
    
    if not input_files:
        st.info(f"No files in {input_path}. Add documents to this folder.")
    else:
        # Summary by type
        type_counts = {}
        total_size = 0
        for f in input_files:
            ext = f["type"] or "unknown"
            type_counts[ext] = type_counts.get(ext, 0) + 1
            total_size += f["size"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Files", len(input_files))
        with col2:
            st.metric("Total Size", f"{total_size / 1024 / 1024:.2f} MB")
        
        st.markdown("**By Type:**")
        for ext, count in sorted(type_counts.items()):
            st.text(f"  {ext}: {count} files")
        
        st.markdown("---")
        st.markdown("**All Files:**")
        for f in input_files:
            size_kb = f["size"] / 1024
            st.text(f"{f['name']} ({size_kb:.1f} KB)")

# Footer
st.markdown("---")
st.markdown("**Rager - Lite** - RAG Document Processor | Created by Josh Holtzclaw | Leveraging LlamaIndex")
