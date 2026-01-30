# Rager - Lite

**RAG Document Processor**

*Created by Josh Holtzclaw - 12/28/2025*

A Python CLI tool that processes documents (PDF, DOCX, CSV, TXT, HTML, MD) into clean text files optimized for GPT knowledge bases. This is the lightweight version of Rager. The name "Rager" is a play on RAG (Retrieval-Augmented Generation) - because processing documents should be a party!

**No API key required** - all processing is done locally (with optional OpenAI enhancement).

## Features

- Recursively processes folders of documents
- Supports PDF, DOCX, CSV, TXT, HTML, and Markdown files
- Optional text chunking for better GPT retrieval performance
- Output as plain text or JSONL with metadata
- Automatically splits large files (GPT has a 20MB limit)
- **Timestamped run folders** - each run creates organized output
- **Auto-generated instructions** - creates `instructions.md` with GPT setup guide
- **Web UI** - Streamlit interface for easy configuration and downloads
- **Optional OpenAI integration** - AI-powered file descriptions with:
  - Batched API calls (10 files per call = 90% fewer API calls)
  - Local caching (don't re-process unchanged files)
  - Selective file types (only use AI for PDFs, DOCX - skip simple TXT/CSV)

## Prerequisites

- **Python 3.11** (recommended - Python 3.13 has compatibility issues with LlamaIndex)
- pip (Python package manager)

## Installation

1. Clone or download this repository

2. Open a terminal in the project folder

3. Create a virtual environment with Python 3.11:

```bash
# Linux/Mac
python3.11 -m venv venv
source venv/bin/activate

# Windows (PowerShell) - use py launcher to specify version
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows Note:** If venv creation fails with ensurepip errors, use this workaround:

```powershell
py -3.11 -m venv venv --without-pip
.\venv\Scripts\Activate.ps1
Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile get-pip.py
python get-pip.py
Remove-Item get-pip.py
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

1. Place your documents in the `input/` folder

2. Run the processor:

```bash
python process_docs.py
```

3. Find the processed files in `output/run_YYYY-MM-DD_HH-MM-SS/`
   - `knowledge/` - Files to upload to GPT
   - `instructions.md` - Setup guide with GPT instructions
   - `manifest.json` - Processing summary

4. Follow the instructions in `instructions.md` to set up your GPT

## Web UI (Optional)

For a visual interface, run the Streamlit web UI:

```bash
streamlit run web_ui.py
```

This opens a browser with:
- Configuration options
- One-click processing
- Download buttons for knowledge files
- View previous runs and instructions

## Usage

### Basic Usage

```bash
# Process documents using config.yaml settings
python process_docs.py
```

### Command Line Options

```bash
# Specify input and output directories
# Linux/Mac:
python process_docs.py --input "/home/user/documents" --output "/home/user/output"
# Windows:
python process_docs.py --input "D:\Documents\MyFiles" --output "D:\Output"

# Output as JSONL format (includes metadata)
python process_docs.py --format jsonl

# Output both TXT and JSONL
python process_docs.py --format both

# Disable chunking (one output file per input document)
python process_docs.py --no-chunk

# Use a different config file
python process_docs.py --config my_config.yaml
```

### All Options

| Option | Short | Description |
|--------|-------|-------------|
| `--input` | `-i` | Input directory path |
| `--output` | `-o` | Output directory path |
| `--format` | `-f` | Output format: `txt`, `jsonl`, or `both` |
| `--no-chunk` | | Disable document chunking |
| `--config` | `-c` | Path to config file (default: config.yaml) |
| `--use-openai` | | Use OpenAI API for better file descriptions |
| `--api-key` | | OpenAI API key (or set OPENAI_API_KEY env var) |
| `--ai-types` | | File types for AI descriptions (comma-separated: pdf,docx,html,md) |
| `--ai-batch-size` | | Files per API call (default: 10, reduces API calls by 90%) |
| `--clear-cache` | | Clear the description cache before processing |

## Configuration

Edit `config.yaml` to customize processing:

```yaml
# Input/Output Paths
input_path: "./input"
output_path: "./output"

# Output Format: txt, jsonl, or both
output_format: "txt"

# Chunking Settings
chunk_documents: true      # Enable/disable chunking
chunk_size: 1024           # Characters per chunk
chunk_overlap: 200         # Overlap between chunks

# File size limit (GPT max is 20MB)
max_file_size_mb: 19

# Enable/disable file types
file_types:
  pdf: true
  docx: true
  csv: true
  txt: true
  html: true
  md: true

# Search subdirectories
recursive: true
```

## Output Files

Each run creates a timestamped folder with organized output:

```
output/
└── run_2025-01-30_10-30-00/
    ├── knowledge/              # Upload these to GPT
    │   ├── report.txt
    │   ├── orders.txt
    │   └── documents.jsonl     # If JSONL format selected
    ├── instructions.md         # GPT setup instructions
    └── manifest.json           # Processing summary
```

### instructions.md

Auto-generated file containing:
- List of all knowledge files with descriptions
- Suggested GPT instructions (copy/paste ready)
- Upload steps for Custom GPT
- Processing statistics

### manifest.json

```json
{
  "generated_at": "2025-01-30T10:30:00",
  "stats": {
    "documents_processed": 15,
    "chunks_created": 142,
    "output_files": 15,
    "chunking_enabled": true
  },
  "files": [
    {"file": "report.txt", "source": "report", "original_type": ".pdf", "chunks": 12},
    {"file": "orders.txt", "source": "orders", "original_type": ".csv", "chunks": 3}
  ]
}
```

## Uploading to GPT Knowledge Base

1. Open the generated `instructions.md` file in your run folder

2. Follow the detailed steps, which include:
   - Uploading all files from the `knowledge/` folder
   - Copying the suggested GPT instructions
   - Configuring your Custom GPT

**Quick steps:**
1. Go to [ChatGPT](https://chat.openai.com) > My GPTs > Create/Edit GPT
2. Click "Configure" > scroll to "Knowledge"
3. Upload all files from `output/run_XXXX/knowledge/`
4. Paste instructions from `instructions.md` into the Instructions field
5. Save

**Note:** OpenAI Custom GPTs have a 20MB per-file limit. This tool automatically splits larger files.

## Supported File Types

| Type | Extensions |
|------|------------|
| PDF | .pdf |
| Word | .docx, .doc |
| CSV | .csv |
| Text | .txt |
| HTML | .html, .htm |
| Markdown | .md, .markdown |

## Troubleshooting

### "No documents found"

- Check that the `input/` folder contains supported file types
- Verify the file extensions are enabled in `config.yaml`

### Import errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Use Python 3.11** - check with `python --version`
- Python 3.13 has known compatibility issues with LlamaIndex (circular import errors)
- On Linux/Mac, you may need to use `python3.11` instead of `python`

### Python 3.13 errors (idna circular import)

If you see errors like `cannot import name 'idnadata' from partially initialized module 'idna'`:
- This is a Python 3.13 compatibility issue
- Solution: Create a new venv with Python 3.11:

```powershell
# Windows
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### PDF parsing issues

- Some PDFs with complex layouts or scanned images may not extract cleanly
- For scanned PDFs, you may need OCR preprocessing

## License

MIT License - feel free to use and modify as needed.
