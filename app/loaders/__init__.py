import re
import ftfy
from pathlib import Path


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    return text.strip()


def load_txt(file_path: str) -> list[dict]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return [{
        "content": clean_text(text),
        "metadata": {
            "source": file_path,
            "page": 1,
            "doc_type": "txt",
            "filename": Path(file_path).name,
        }
    }]


def load_md(file_path: str) -> list[dict]:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    return [{
        "content": clean_text(text),
        "metadata": {
            "source": file_path,
            "page": 1,
            "doc_type": "markdown",
            "filename": Path(file_path).name,
        }
    }]


def load_document(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".txt":
        return load_txt(file_path)
    elif ext in (".md", ".markdown"):
        return load_md(file_path)
    else:
        print(f"  Skipping unsupported file type: {ext}")
        return []


def load_directory(dir_path: str) -> list[dict]:
    all_docs = []
    files = list(Path(dir_path).rglob("*.*"))
    print(f"\nFound {len(files)} files in {dir_path}")
    for f in files:
        print(f"Loading: {f.name}")
        docs = load_document(str(f))
        all_docs.extend(docs)
    print(f"\nTotal sections loaded: {len(all_docs)}")
    return all_docs