#!/usr/bin/env python3
"""
Rager - RAG Document Processor
Created by Josh Holtzclaw - 12/28/2025

Processes documents (PDF, DOCX, CSV, TXT, HTML, MD) into clean text files
optimized for GPT knowledge bases. Auto-generates instructions for each run.

Usage:
    python process_docs.py
    python process_docs.py --input "./my_docs" --output "./my_output"
    python process_docs.py --format jsonl --no-chunk
    python process_docs.py --use-openai  # Use OpenAI for better descriptions

Requires Python 3.11 (3.13 has compatibility issues with LlamaIndex)
"""

import sys

# Check Python version
if sys.version_info >= (3, 13):
    print("WARNING: Python 3.13+ detected. LlamaIndex may have compatibility issues.")
    print("Recommended: Use Python 3.11 or 3.12")
    print("  Windows: py -3.11 -m venv venv")
    print("  Linux/Mac: python3.11 -m venv venv")
    print()

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document


# File type descriptions for auto-generated instructions
FILE_TYPE_HINTS = {
    ".pdf": "PDF document",
    ".docx": "Word document",
    ".doc": "Word document",
    ".csv": "CSV data file (comma-separated values)",
    ".txt": "Plain text file",
    ".html": "HTML web page",
    ".htm": "HTML web page",
    ".md": "Markdown document",
    ".markdown": "Markdown document",
}


# Cache file for OpenAI descriptions
DESCRIPTION_CACHE_FILE = ".description_cache.json"


def load_description_cache(cache_path: Path) -> dict:
    """Load cached descriptions from file."""
    cache_file = cache_path / DESCRIPTION_CACHE_FILE
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_description_cache(cache_path: Path, cache: dict):
    """Save descriptions cache to file."""
    cache_file = cache_path / DESCRIPTION_CACHE_FILE
    cache_path.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def get_content_hash(content: str) -> str:
    """Generate a hash for content to use as cache key."""
    import hashlib
    return hashlib.md5(content[:1000].encode()).hexdigest()


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    default_config = {
        "input_path": "./input",
        "output_path": "./output",
        "output_format": "txt",
        "chunk_documents": True,
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "max_file_size_mb": 19,
        "file_types": {
            "pdf": True,
            "docx": True,
            "csv": True,
            "txt": True,
            "html": True,
            "md": True,
        },
        "recursive": True,
        "include_metadata": True,
        "openai_api_key": "",
        "use_openai": False,
        "openai_model": "gpt-5-mini",
        "openai_batch_size": 10,
        "openai_file_types": ["pdf", "docx", "html", "md"],  # File types to use AI for
        "csv_rows_per_chunk": 50,  # Rows per chunk for CSV files (header preserved in each)
    }

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
            default_config.update(user_config)

    # Check environment variable for API key
    if not default_config["openai_api_key"]:
        default_config["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")

    return default_config


def get_enabled_extensions(file_types: dict) -> list:
    """Get list of enabled file extensions based on config."""
    extension_map = {
        "pdf": [".pdf"],
        "docx": [".docx", ".doc"],
        "csv": [".csv"],
        "txt": [".txt"],
        "html": [".html", ".htm"],
        "md": [".md", ".markdown"],
    }

    extensions = []
    for file_type, enabled in file_types.items():
        if enabled and file_type in extension_map:
            extensions.extend(extension_map[file_type])

    return extensions


def load_documents(input_path: str, recursive: bool, extensions: list) -> list:
    """Load documents using SimpleDirectoryReader."""
    input_dir = Path(input_path)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_path}' does not exist.")
        sys.exit(1)

    # Check if directory has any files (excluding .gitkeep)
    files = [f for f in input_dir.iterdir() if f.name != ".gitkeep"]
    if not files:
        print(f"Warning: Input directory '{input_path}' is empty.")
        return []

    print(f"Loading documents from: {input_path}")
    print(f"Recursive: {recursive}")
    print(f"File types: {', '.join(extensions)}")

    try:
        reader = SimpleDirectoryReader(
            input_dir=str(input_dir),
            recursive=recursive,
            required_exts=extensions,
        )
        documents = reader.load_data()
        print(f"Loaded {len(documents)} document(s)")
        return documents
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []


def chunk_csv_document(doc, rows_per_chunk: int = 50) -> list:
    """Chunk a CSV document by rows, preserving header in each chunk."""
    from llama_index.core.schema import TextNode
    
    text = getattr(doc, "text", "")
    metadata = getattr(doc, "metadata", {}) or {}
    
    lines = text.strip().split("\n")
    if len(lines) <= 1:
        return [doc]  # No data rows, return as-is
    
    header = lines[0]
    data_rows = lines[1:]
    
    chunks = []
    for i in range(0, len(data_rows), rows_per_chunk):
        chunk_rows = data_rows[i:i + rows_per_chunk]
        chunk_text = header + "\n" + "\n".join(chunk_rows)
        
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk_index"] = i // rows_per_chunk
        chunk_metadata["is_csv_chunk"] = True
        
        node = TextNode(
            text=chunk_text,
            metadata=chunk_metadata,
        )
        chunks.append(node)
    
    return chunks


def chunk_documents(documents: list, chunk_size: int, chunk_overlap: int, csv_rows_per_chunk: int = 50) -> list:
    """Split documents into smaller chunks. CSVs are chunked by rows with header preserved."""
    if not documents:
        return []

    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    print(f"CSV chunking: {csv_rows_per_chunk} rows per chunk (header preserved)")

    # Separate CSVs from other documents
    csv_docs = []
    other_docs = []
    
    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        file_name = metadata.get("file_name", "")
        if file_name.lower().endswith(".csv"):
            csv_docs.append(doc)
        else:
            other_docs.append(doc)
    
    nodes = []
    
    # Chunk CSVs by rows (preserving header)
    if csv_docs:
        print(f"  Processing {len(csv_docs)} CSV file(s) with row-based chunking...")
        for doc in csv_docs:
            csv_chunks = chunk_csv_document(doc, csv_rows_per_chunk)
            nodes.extend(csv_chunks)
    
    # Chunk other documents with SentenceSplitter
    if other_docs:
        print(f"  Processing {len(other_docs)} other document(s) with sentence chunking...")
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        other_nodes = splitter.get_nodes_from_documents(other_docs)
        nodes.extend(other_nodes)

    print(f"Created {len(nodes)} chunk(s)")
    return nodes


def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filename."""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, "_")
    return filename


def get_source_filename(item) -> str:
    """Extract source filename from document or node metadata."""
    metadata = getattr(item, "metadata", {}) or {}

    for key in ["file_name", "filename", "source"]:
        if key in metadata:
            return Path(metadata[key]).stem

    doc_id = getattr(item, "id_", None) or getattr(item, "doc_id", "unknown")
    return str(doc_id)[:50]


def get_source_extension(item) -> str:
    """Extract source file extension from document or node metadata."""
    metadata = getattr(item, "metadata", {}) or {}

    for key in ["file_name", "filename", "source"]:
        if key in metadata:
            return Path(metadata[key]).suffix.lower()

    return ""


def create_run_folder(base_output_path: str) -> Path:
    """Create a timestamped run folder with knowledge subfolder."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = Path(base_output_path) / f"run_{timestamp}"
    knowledge_folder = run_folder / "knowledge"
    knowledge_folder.mkdir(parents=True, exist_ok=True)
    return run_folder


def write_txt_output(items: list, output_path: Path, max_size_mb: float, is_chunked: bool) -> tuple:
    """Write items to text files in knowledge folder."""
    knowledge_dir = output_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    max_size_bytes = int(max_size_mb * 1024 * 1024)
    files_written = 0
    manifest = []

    # Group items by source file
    grouped = {}
    extensions = {}
    for item in items:
        source = get_source_filename(item)
        ext = get_source_extension(item)
        if source not in grouped:
            grouped[source] = []
            extensions[source] = ext
        grouped[source].append(item)

    for source_name, source_items in tqdm(grouped.items(), desc="Writing files"):
        safe_name = sanitize_filename(source_name)
        content_parts = []
        original_ext = extensions.get(source_name, "")

        for idx, item in enumerate(source_items):
            text = getattr(item, "text", "") or getattr(item, "get_content", lambda: "")()

            if is_chunked and len(source_items) > 1:
                header = f"--- Chunk {idx + 1}/{len(source_items)} ---\n"
                content_parts.append(header + text)
            else:
                content_parts.append(text)

        full_content = "\n\n".join(content_parts)

        # Split if exceeds max size
        if len(full_content.encode("utf-8")) > max_size_bytes:
            part_num = 1
            current_content = ""

            for part in content_parts:
                test_content = current_content + "\n\n" + part if current_content else part

                if len(test_content.encode("utf-8")) > max_size_bytes:
                    if current_content:
                        filename = f"{safe_name}_part{part_num}.txt"
                        filepath = knowledge_dir / filename
                        filepath.write_text(current_content, encoding="utf-8")
                        manifest.append({
                            "file": filename,
                            "source": source_name,
                            "original_type": original_ext,
                            "part": part_num,
                            "chunks": len(source_items),
                        })
                        files_written += 1
                        part_num += 1
                    current_content = part
                else:
                    current_content = test_content

            if current_content:
                filename = f"{safe_name}_part{part_num}.txt"
                filepath = knowledge_dir / filename
                filepath.write_text(current_content, encoding="utf-8")
                manifest.append({
                    "file": filename,
                    "source": source_name,
                    "original_type": original_ext,
                    "part": part_num,
                    "chunks": len(source_items),
                })
                files_written += 1
        else:
            filename = f"{safe_name}.txt"
            filepath = knowledge_dir / filename
            filepath.write_text(full_content, encoding="utf-8")
            manifest.append({
                "file": filename,
                "source": source_name,
                "original_type": original_ext,
                "chunks": len(source_items),
            })
            files_written += 1

    return files_written, manifest


def write_jsonl_output(items: list, output_path: Path, max_size_mb: float, is_chunked: bool) -> tuple:
    """Write items to JSONL file in knowledge folder."""
    knowledge_dir = output_path / "knowledge"
    knowledge_dir.mkdir(parents=True, exist_ok=True)

    max_size_bytes = int(max_size_mb * 1024 * 1024)

    records = []
    for idx, item in enumerate(items):
        text = getattr(item, "text", "") or getattr(item, "get_content", lambda: "")()
        metadata = getattr(item, "metadata", {}) or {}

        record = {
            "id": idx,
            "text": text,
            "source_file": metadata.get("file_name", "unknown"),
            "file_path": metadata.get("file_path", ""),
            "file_type": metadata.get("file_type", ""),
        }

        if is_chunked:
            record["chunk_index"] = idx

        records.append(record)

    # Write JSONL, splitting if needed
    files_written = 0
    current_lines = []
    current_size = 0
    part_num = 1

    for record in tqdm(records, desc="Writing JSONL"):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        line_size = len(line.encode("utf-8"))

        if current_size + line_size > max_size_bytes and current_lines:
            filename = f"documents_part{part_num}.jsonl"
            filepath = knowledge_dir / filename
            filepath.write_text("".join(current_lines), encoding="utf-8")
            files_written += 1
            part_num += 1
            current_lines = []
            current_size = 0

        current_lines.append(line)
        current_size += line_size

    if current_lines:
        filename = f"documents_part{part_num}.jsonl" if part_num > 1 else "documents.jsonl"
        filepath = knowledge_dir / filename
        filepath.write_text("".join(current_lines), encoding="utf-8")
        files_written += 1

    return files_written, records


def generate_file_description(file_info: dict, use_openai: bool = False, api_key: str = "") -> str:
    """Generate a description for a file based on its metadata."""
    source = file_info.get("source", "unknown")
    original_type = file_info.get("original_type", "")
    chunks = file_info.get("chunks", 1)

    type_desc = FILE_TYPE_HINTS.get(original_type, "document")

    # Basic description
    desc = f"{source} - {type_desc}"
    if chunks > 1:
        desc += f" ({chunks} chunks)"

    return desc


def generate_openai_descriptions(
    manifest: list, 
    documents: list, 
    api_key: str,
    config: dict
) -> tuple:
    """Use OpenAI to generate better file descriptions with batching and caching.
    
    Returns:
        tuple: (descriptions dict, token_usage dict)
    """
    if not api_key:
        return {}, {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Track token usage
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_files": 0,
            "api_calls": 0,
        }

        # Load cache
        cache_path = Path(config.get("output_path", "./output"))
        cache = load_description_cache(cache_path)
        
        # Get config options
        model = config.get("openai_model", "gpt-4o-mini")
        batch_size = config.get("openai_batch_size", 10)
        ai_file_types = config.get("openai_file_types", ["pdf", "docx", "html", "md"])
        
        # Convert to extensions
        ai_extensions = []
        ext_map = {"pdf": [".pdf"], "docx": [".docx", ".doc"], "csv": [".csv"], 
                   "txt": [".txt"], "html": [".html", ".htm"], "md": [".md", ".markdown"]}
        for ft in ai_file_types:
            ai_extensions.extend(ext_map.get(ft, []))

        descriptions = {}
        files_to_process = []
        
        print(f"\nPreparing AI descriptions (model: {model}, batch size: {batch_size})...")
        print(f"AI-enabled file types: {', '.join(ai_file_types)}")

        # Collect files that need AI descriptions
        for file_info in manifest:
            source = file_info.get("source", "unknown")
            original_type = file_info.get("original_type", "")
            
            # Skip file types not selected for AI
            if original_type not in ai_extensions:
                descriptions[source] = generate_file_description(file_info)
                continue
            
            # Find document content
            content_preview = ""
            for doc in documents:
                metadata = getattr(doc, "metadata", {}) or {}
                if source in metadata.get("file_name", ""):
                    text = getattr(doc, "text", "")
                    
                    # For CSVs, only use header + first few rows (saves tokens)
                    if original_type == ".csv":
                        lines = text.strip().split("\n")
                        if len(lines) > 1:
                            # Header + first 3 data rows max
                            preview_lines = lines[:4]
                            content_preview = "\n".join(preview_lines)
                            content_preview += f"\n... ({len(lines) - 1} total data rows)"
                        else:
                            content_preview = text[:300]
                    else:
                        content_preview = text[:500]
                    break
            
            if not content_preview:
                descriptions[source] = generate_file_description(file_info)
                continue
            
            # Check cache
            content_hash = get_content_hash(content_preview)
            cache_key = f"{source}_{content_hash}"
            
            if cache_key in cache:
                descriptions[source] = cache[cache_key]
                token_usage["cached_files"] += 1
                continue
            
            files_to_process.append({
                "source": source,
                "content": content_preview,
                "cache_key": cache_key,
                "file_info": file_info,
            })

        if not files_to_process:
            print("  All descriptions cached or skipped.")
            return descriptions

        print(f"  Processing {len(files_to_process)} files in {(len(files_to_process) + batch_size - 1) // batch_size} batch(es)...")

        # Process in batches
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(files_to_process) + batch_size - 1) // batch_size
            
            print(f"  Batch {batch_num}/{total_batches} ({len(batch)} files)...")
            
            # Build batch prompt
            files_text = ""
            for idx, item in enumerate(batch):
                files_text += f"\n[FILE {idx + 1}] {item['source']}\n{item['content'][:300]}\n"
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": """Generate brief 1-sentence descriptions for each document. 
Be specific and factual. Format your response as:
[FILE 1] Description here
[FILE 2] Description here
etc."""
                        },
                        {
                            "role": "user",
                            "content": f"Generate descriptions for these {len(batch)} documents:{files_text}"
                        }
                    ],
                    max_tokens=50 * len(batch),
                    temperature=0.3,
                )
                
                # Track token usage
                if response.usage:
                    token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    token_usage["completion_tokens"] += response.usage.completion_tokens
                    token_usage["total_tokens"] += response.usage.total_tokens
                token_usage["api_calls"] += 1
                
                # Parse response
                response_text = response.choices[0].message.content.strip()
                lines = response_text.split("\n")
                
                for idx, item in enumerate(batch):
                    # Try to find matching line
                    desc = None
                    for line in lines:
                        if f"[FILE {idx + 1}]" in line:
                            desc = line.replace(f"[FILE {idx + 1}]", "").strip()
                            break
                    
                    if not desc:
                        # Fallback: try to get line by index
                        if idx < len(lines):
                            desc = lines[idx].strip()
                            # Remove any [FILE X] prefix
                            if desc.startswith("[FILE"):
                                desc = desc.split("]", 1)[-1].strip()
                    
                    if desc:
                        descriptions[item["source"]] = desc
                        cache[item["cache_key"]] = desc
                    else:
                        descriptions[item["source"]] = generate_file_description(item["file_info"])
                        
            except Exception as e:
                print(f"    Warning: Batch failed: {e}")
                # Fallback for failed batch
                for item in batch:
                    descriptions[item["source"]] = generate_file_description(item["file_info"])

        # Save updated cache
        save_description_cache(cache_path, cache)
        print(f"  Cache updated ({len(cache)} entries)")
        
        # Print token usage summary
        if token_usage["api_calls"] > 0:
            print(f"  Token usage: {token_usage['total_tokens']} total "
                  f"({token_usage['prompt_tokens']} prompt, {token_usage['completion_tokens']} completion)")
            print(f"  API calls: {token_usage['api_calls']}, Cached: {token_usage['cached_files']}")

        return descriptions, token_usage

    except ImportError:
        print("Warning: openai package not installed. Run: pip install openai")
        return {}, {}
    except Exception as e:
        print(f"Warning: OpenAI API error: {e}")
        return {}, {}


def generate_instructions(manifest: list, stats: dict, documents: list, config: dict) -> tuple:
    """Generate GPT instructions based on processed files.
    
    Returns:
        tuple: (instructions string, token_usage dict)
    """

    # Get AI descriptions if enabled
    ai_descriptions = {}
    token_usage = {}
    use_openai = config.get("use_openai", False)
    api_key = config.get("openai_api_key", "")
    
    if use_openai and api_key:
        ai_descriptions, token_usage = generate_openai_descriptions(manifest, documents, api_key, config)

    # Build file list with descriptions
    file_lines = []
    for file_info in manifest:
        source = file_info.get("source", "unknown")
        filename = file_info.get("file", "")

        if source in ai_descriptions:
            desc = ai_descriptions[source]
        else:
            desc = generate_file_description(file_info)

        file_lines.append(f"- **{filename}**: {desc}")

    files_section = "\n".join(file_lines)

    # Determine content types
    content_types = set()
    for file_info in manifest:
        ext = file_info.get("original_type", "")
        if ext in [".csv"]:
            content_types.add("structured data (CSV)")
        elif ext in [".pdf", ".docx", ".doc"]:
            content_types.add("documents")
        elif ext in [".html", ".htm"]:
            content_types.add("web content")
        elif ext in [".md", ".markdown"]:
            content_types.add("markdown documentation")
        elif ext in [".txt"]:
            content_types.add("text files")

    content_desc = ", ".join(sorted(content_types)) if content_types else "various documents"

    instructions = f"""# GPT Knowledge Base Instructions

## Overview

This knowledge base contains {stats['documents_processed']} processed documents with {stats['chunks_created']} text chunks, ready for use with a Custom GPT.

## Knowledge Files

Upload all files from the `knowledge/` folder to your GPT's Knowledge section:

{files_section}

## Content Types

This knowledge base includes: {content_desc}

## Suggested GPT Instructions

Copy and customize the following instructions for your GPT's "Instructions" field:

---

### System Instructions (copy below this line)

You have access to a knowledge base containing the following files:

{files_section}

**How to use this knowledge:**

1. When answering questions, search the relevant knowledge files first
2. For structured data (CSV-derived files), the data is comma-separated with the format shown in the first line
3. Quote specific information from the knowledge base when relevant
4. If information is not found in the knowledge base, clearly state that

**Response style:**
- Be factual and reference specific documents when possible
- For data queries, provide specific values from the knowledge files
- Acknowledge when information might be incomplete or when the knowledge base doesn't cover a topic

---

## Upload Instructions

1. Go to [ChatGPT](https://chat.openai.com)
2. Click your profile icon > "My GPTs" > Select or create your GPT
3. Click "Configure"
4. Scroll to "Knowledge" section
5. Click "Upload files" and select all files from the `knowledge/` folder
6. Paste the suggested instructions above into the "Instructions" field
7. Save your GPT

## Processing Details

- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Documents processed: {stats['documents_processed']}
- Chunks created: {stats['chunks_created']}
- Output files: {stats['output_files']}
- Chunking enabled: {stats['chunking_enabled']}
"""

    if use_openai and api_key:
        instructions += "\n- AI-enhanced descriptions: Yes\n"
        if token_usage:
            instructions += f"- Tokens used: {token_usage.get('total_tokens', 0)}\n"

    return instructions, token_usage


def write_manifest(output_path: Path, manifest_data: list, stats: dict, token_usage: dict = None):
    """Write manifest.json with processing summary."""
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "stats": stats,
        "files": manifest_data,
    }
    
    if token_usage:
        manifest["openai_usage"] = token_usage

    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Process documents for GPT knowledge base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", "-i", help="Input directory path")
    parser.add_argument("--output", "-o", help="Output directory path")
    parser.add_argument("--format", "-f", choices=["txt", "jsonl", "both"], help="Output format")
    parser.add_argument("--no-chunk", action="store_true", help="Disable document chunking")
    parser.add_argument("--config", "-c", default="config.yaml", help="Config file path")
    parser.add_argument("--use-openai", action="store_true", help="Use OpenAI API for better descriptions")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--ai-types", help="File types for AI descriptions (comma-separated: pdf,docx,html,md)")
    parser.add_argument("--ai-batch-size", type=int, help="Batch size for OpenAI API calls (default: 10)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear the description cache before processing")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.input:
        config["input_path"] = args.input
    if args.output:
        config["output_path"] = args.output
    if args.format:
        config["output_format"] = args.format
    if args.no_chunk:
        config["chunk_documents"] = False
    if args.use_openai:
        config["use_openai"] = True
    if args.api_key:
        config["openai_api_key"] = args.api_key
    if args.ai_types:
        config["openai_file_types"] = [t.strip() for t in args.ai_types.split(",")]
    if args.ai_batch_size:
        config["openai_batch_size"] = args.ai_batch_size
    
    # Clear cache if requested
    if args.clear_cache:
        cache_path = Path(config["output_path"]) / DESCRIPTION_CACHE_FILE
        if cache_path.exists():
            cache_path.unlink()
            print("Description cache cleared.")

    print("=" * 50)
    print("Rager - RAG Document Processor")
    print("Created by Josh Holtzclaw")
    print("=" * 50)

    # Get enabled extensions
    extensions = get_enabled_extensions(config["file_types"])

    # Load documents
    documents = load_documents(
        config["input_path"],
        config["recursive"],
        extensions,
    )

    if not documents:
        print("No documents found. Exiting.")
        sys.exit(0)

    # Create timestamped run folder
    run_folder = create_run_folder(config["output_path"])
    print(f"\nOutput folder: {run_folder}")

    # Chunk if enabled
    if config["chunk_documents"]:
        items = chunk_documents(
            documents,
            config["chunk_size"],
            config["chunk_overlap"],
            config.get("csv_rows_per_chunk", 50),
        )
        is_chunked = True
    else:
        items = documents
        is_chunked = False

    # Write output
    output_format = config["output_format"]
    manifest_data = []
    total_files = 0

    if output_format in ["txt", "both"]:
        print("\nWriting TXT output...")
        files, manifest = write_txt_output(
            items,
            run_folder,
            config["max_file_size_mb"],
            is_chunked,
        )
        total_files += files
        manifest_data.extend(manifest)

    if output_format in ["jsonl", "both"]:
        print("\nWriting JSONL output...")
        files, _ = write_jsonl_output(
            items,
            run_folder,
            config["max_file_size_mb"],
            is_chunked,
        )
        total_files += files

    # Build stats
    stats = {
        "documents_processed": len(documents),
        "chunks_created": len(items) if is_chunked else len(documents),
        "output_files": total_files,
        "chunking_enabled": is_chunked,
    }

    # Generate and write instructions
    print("\nGenerating instructions...")
    instructions, token_usage = generate_instructions(
        manifest_data,
        stats,
        documents,
        config,
    )
    instructions_path = run_folder / "instructions.md"
    instructions_path.write_text(instructions, encoding="utf-8")

    # Write manifest (with token usage if available)
    write_manifest(run_folder, manifest_data, stats, token_usage)

    print("\n" + "=" * 50)
    print("Processing Complete!")
    print(f"  Documents processed: {len(documents)}")
    print(f"  Chunks created: {len(items)}")
    print(f"  Output files: {total_files}")
    if token_usage and token_usage.get("total_tokens"):
        print(f"  OpenAI tokens used: {token_usage['total_tokens']}")
    print(f"  Run folder: {run_folder}")
    print(f"  Knowledge files: {run_folder / 'knowledge'}")
    print(f"  Instructions: {run_folder / 'instructions.md'}")
    print("=" * 50)

    return str(run_folder)


if __name__ == "__main__":
    main()
