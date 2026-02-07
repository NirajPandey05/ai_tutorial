# Document Loading Strategies

Learn how to load, parse, and prepare documents from various sources for RAG pipelines.

## Why Document Loading Matters

```yaml
the_challenge:
  - "Documents come in many formats (PDF, HTML, Word, etc.)"
  - "Each format has unique parsing requirements"
  - "Metadata extraction varies by source"
  - "Quality of loading affects retrieval quality"

loading_pipeline:
  1. "Identify document type"
  2. "Extract raw text content"
  3. "Preserve structure where possible"
  4. "Capture relevant metadata"
  5. "Handle errors gracefully"
```

## Common Document Formats

### PDF Documents

```python
import fitz  # PyMuPDF - fast and reliable

def load_pdf(path: str) -> dict:
    """Load PDF with text and metadata."""
    doc = fitz.open(path)
    
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        pages.append({
            "page_number": page_num + 1,
            "content": text,
            "char_count": len(text)
        })
    
    return {
        "content": "\n\n".join(p["content"] for p in pages),
        "pages": pages,
        "metadata": {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "page_count": len(doc),
            "format": "pdf"
        }
    }

# Usage
doc = load_pdf("company_handbook.pdf")
print(f"Loaded {doc['metadata']['page_count']} pages")
```

### PDF with Tables and Images

```python
import fitz
from PIL import Image
import io


def load_pdf_rich(path: str) -> dict:
    """Load PDF with tables, images, and text."""
    doc = fitz.open(path)
    
    content_blocks = []
    images = []
    
    for page_num, page in enumerate(doc):
        # Extract text blocks with positions
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text += span["text"]
                    text += "\n"
                
                content_blocks.append({
                    "type": "text",
                    "page": page_num + 1,
                    "content": text.strip()
                })
            
            elif block["type"] == 1:  # Image block
                # Extract image
                xref = block["xref"]
                if xref:
                    img_data = doc.extract_image(xref)
                    images.append({
                        "page": page_num + 1,
                        "data": img_data["image"],
                        "ext": img_data["ext"]
                    })
        
        # Try to extract tables
        tables = page.find_tables()
        for table in tables:
            content_blocks.append({
                "type": "table",
                "page": page_num + 1,
                "content": table.extract()  # Returns list of rows
            })
    
    return {
        "blocks": content_blocks,
        "images": images,
        "metadata": {"format": "pdf", "pages": len(doc)}
    }
```

### HTML Documents

```python
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse


def load_html(source: str) -> dict:
    """Load HTML from file or URL."""
    
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        html_content = response.text
        base_url = source
    else:
        with open(source, "r", encoding="utf-8") as f:
            html_content = f.read()
        base_url = None
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    # Extract title
    title = soup.title.string if soup.title else ""
    
    # Extract main content (try common patterns)
    main_content = (
        soup.find("main") or 
        soup.find("article") or 
        soup.find(class_="content") or
        soup.body or
        soup
    )
    
    # Extract text with some structure
    text = main_content.get_text(separator="\n", strip=True)
    
    # Extract links
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if base_url:
            href = urljoin(base_url, href)
        links.append({"text": a.get_text(strip=True), "url": href})
    
    return {
        "content": text,
        "metadata": {
            "title": title,
            "format": "html",
            "url": source if source.startswith("http") else None,
            "link_count": len(links)
        },
        "links": links
    }


def load_html_structured(source: str) -> dict:
    """Load HTML preserving more structure."""
    
    if source.startswith(("http://", "https://")):
        response = requests.get(source, timeout=30)
        html_content = response.text
    else:
        with open(source, "r", encoding="utf-8") as f:
            html_content = f.read()
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    sections = []
    current_section = {"heading": None, "content": []}
    
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "ul", "ol", "pre"]):
        if element.name in ["h1", "h2", "h3", "h4"]:
            if current_section["content"]:
                sections.append(current_section)
            current_section = {
                "heading": element.get_text(strip=True),
                "level": int(element.name[1]),
                "content": []
            }
        else:
            text = element.get_text(strip=True)
            if text:
                current_section["content"].append(text)
    
    if current_section["content"]:
        sections.append(current_section)
    
    return {
        "sections": sections,
        "metadata": {"format": "html_structured"}
    }
```

### Markdown Documents

```python
import re
from pathlib import Path


def load_markdown(path: str) -> dict:
    """Load Markdown with structure."""
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Extract frontmatter if present
    frontmatter = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            import yaml
            frontmatter = yaml.safe_load(parts[1])
            content = parts[2].strip()
    
    # Parse sections by headers
    sections = []
    current_section = {"heading": None, "level": 0, "content": ""}
    
    for line in content.split("\n"):
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if header_match:
            if current_section["content"]:
                sections.append(current_section)
            current_section = {
                "heading": header_match.group(2),
                "level": len(header_match.group(1)),
                "content": ""
            }
        else:
            current_section["content"] += line + "\n"
    
    if current_section["content"]:
        sections.append(current_section)
    
    return {
        "content": content,
        "sections": sections,
        "frontmatter": frontmatter,
        "metadata": {
            "format": "markdown",
            "filename": Path(path).name
        }
    }
```

### Code Files

```python
import ast
from pathlib import Path


def load_code_file(path: str) -> dict:
    """Load source code with structure."""
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    suffix = Path(path).suffix.lower()
    language = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".rs": "rust",
        ".go": "go"
    }.get(suffix, "unknown")
    
    result = {
        "content": content,
        "metadata": {
            "format": "code",
            "language": language,
            "filename": Path(path).name,
            "lines": content.count("\n") + 1
        }
    }
    
    # Parse Python structure
    if language == "python":
        try:
            tree = ast.parse(content)
            
            classes = []
            functions = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "line": node.lineno,
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
            
            result["structure"] = {
                "classes": classes,
                "functions": functions,
                "imports": imports
            }
        except SyntaxError:
            pass
    
    return result
```

### Word Documents

```python
from docx import Document as DocxDocument


def load_docx(path: str) -> dict:
    """Load Word document."""
    
    doc = DocxDocument(path)
    
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append({
                "text": para.text,
                "style": para.style.name if para.style else None
            })
    
    # Extract tables
    tables = []
    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text for cell in row.cells]
            table_data.append(row_data)
        tables.append(table_data)
    
    # Combine content
    content = "\n\n".join(p["text"] for p in paragraphs)
    
    return {
        "content": content,
        "paragraphs": paragraphs,
        "tables": tables,
        "metadata": {
            "format": "docx",
            "paragraph_count": len(paragraphs),
            "table_count": len(tables)
        }
    }
```

## Universal Document Loader

```python
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass


@dataclass
class LoadedDocument:
    """Standardized document format."""
    content: str
    source: str
    format: str
    metadata: dict
    sections: Optional[list] = None


class UniversalLoader:
    """Load any supported document type."""
    
    SUPPORTED_FORMATS = {
        ".pdf": "pdf",
        ".html": "html",
        ".htm": "html",
        ".md": "markdown",
        ".txt": "text",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".docx": "docx",
        ".json": "json",
    }
    
    def __init__(self):
        self.loaders = {
            "pdf": self._load_pdf,
            "html": self._load_html,
            "markdown": self._load_markdown,
            "text": self._load_text,
            "code": self._load_code,
            "docx": self._load_docx,
            "json": self._load_json,
        }
    
    def load(self, source: str) -> LoadedDocument:
        """Load a document from path or URL."""
        
        # Handle URLs
        if source.startswith(("http://", "https://")):
            return self._load_url(source)
        
        # Handle files
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source}")
        
        format_type = self.SUPPORTED_FORMATS.get(path.suffix.lower())
        if not format_type:
            raise ValueError(f"Unsupported format: {path.suffix}")
        
        loader = self.loaders[format_type]
        return loader(source)
    
    def load_directory(
        self,
        directory: str,
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> Iterator[LoadedDocument]:
        """Load all documents from a directory."""
        
        path = Path(directory)
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            
            suffix = file_path.suffix.lower()
            
            # Filter by extensions if specified
            if extensions and suffix not in extensions:
                continue
            
            # Skip unsupported formats
            if suffix not in self.SUPPORTED_FORMATS:
                continue
            
            try:
                yield self.load(str(file_path))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def _load_pdf(self, path: str) -> LoadedDocument:
        import fitz
        doc = fitz.open(path)
        content = "\n\n".join(page.get_text() for page in doc)
        return LoadedDocument(
            content=content,
            source=path,
            format="pdf",
            metadata={"pages": len(doc), **doc.metadata}
        )
    
    def _load_html(self, path: str) -> LoadedDocument:
        from bs4 import BeautifulSoup
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return LoadedDocument(
            content=soup.get_text(separator="\n"),
            source=path,
            format="html",
            metadata={"title": soup.title.string if soup.title else ""}
        )
    
    def _load_markdown(self, path: str) -> LoadedDocument:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return LoadedDocument(
            content=content,
            source=path,
            format="markdown",
            metadata={}
        )
    
    def _load_text(self, path: str) -> LoadedDocument:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return LoadedDocument(
            content=content,
            source=path,
            format="text",
            metadata={}
        )
    
    def _load_code(self, path: str) -> LoadedDocument:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return LoadedDocument(
            content=content,
            source=path,
            format="code",
            metadata={"language": Path(path).suffix[1:]}
        )
    
    def _load_docx(self, path: str) -> LoadedDocument:
        from docx import Document as DocxDocument
        doc = DocxDocument(path)
        content = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return LoadedDocument(
            content=content,
            source=path,
            format="docx",
            metadata={}
        )
    
    def _load_json(self, path: str) -> LoadedDocument:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return LoadedDocument(
            content=json.dumps(data, indent=2),
            source=path,
            format="json",
            metadata={}
        )
    
    def _load_url(self, url: str) -> LoadedDocument:
        import requests
        from bs4 import BeautifulSoup
        
        response = requests.get(url, timeout=30)
        content_type = response.headers.get("content-type", "")
        
        if "html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            content = soup.get_text(separator="\n")
            title = soup.title.string if soup.title else ""
        else:
            content = response.text
            title = ""
        
        return LoadedDocument(
            content=content,
            source=url,
            format="html",
            metadata={"title": title, "url": url}
        )


# Usage
loader = UniversalLoader()

# Load single document
doc = loader.load("./docs/handbook.pdf")
print(f"Loaded: {doc.source} ({len(doc.content)} chars)")

# Load directory
for doc in loader.load_directory("./documents", recursive=True):
    print(f"Loaded: {doc.source} - {doc.format}")
```

## Loading from APIs and Databases

```python
import requests
from typing import Iterator


class APIDocumentLoader:
    """Load documents from various APIs."""
    
    def load_notion_page(self, page_id: str, api_key: str) -> LoadedDocument:
        """Load content from Notion page."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Notion-Version": "2022-06-28"
        }
        
        # Get page blocks
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        response = requests.get(url, headers=headers)
        blocks = response.json().get("results", [])
        
        content = []
        for block in blocks:
            block_type = block["type"]
            if block_type in ["paragraph", "heading_1", "heading_2", "heading_3"]:
                rich_text = block[block_type].get("rich_text", [])
                text = "".join(t["plain_text"] for t in rich_text)
                content.append(text)
        
        return LoadedDocument(
            content="\n\n".join(content),
            source=f"notion:{page_id}",
            format="notion",
            metadata={"page_id": page_id}
        )
    
    def load_confluence_page(
        self,
        page_id: str,
        base_url: str,
        username: str,
        api_token: str
    ) -> LoadedDocument:
        """Load content from Confluence page."""
        from bs4 import BeautifulSoup
        
        url = f"{base_url}/wiki/rest/api/content/{page_id}?expand=body.storage"
        response = requests.get(url, auth=(username, api_token))
        data = response.json()
        
        html_content = data["body"]["storage"]["value"]
        soup = BeautifulSoup(html_content, "html.parser")
        
        return LoadedDocument(
            content=soup.get_text(separator="\n"),
            source=f"confluence:{page_id}",
            format="confluence",
            metadata={
                "title": data["title"],
                "page_id": page_id
            }
        )
    
    def load_github_repo(
        self,
        owner: str,
        repo: str,
        path: str = "",
        token: Optional[str] = None
    ) -> Iterator[LoadedDocument]:
        """Load files from GitHub repository."""
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            headers["Authorization"] = f"token {token}"
        
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        
        for item in response.json():
            if item["type"] == "file":
                # Get file content
                file_response = requests.get(item["download_url"])
                
                yield LoadedDocument(
                    content=file_response.text,
                    source=f"github:{owner}/{repo}/{item['path']}",
                    format=Path(item["name"]).suffix[1:] or "text",
                    metadata={
                        "repo": f"{owner}/{repo}",
                        "path": item["path"],
                        "sha": item["sha"]
                    }
                )
            elif item["type"] == "dir":
                # Recursively load directory
                yield from self.load_github_repo(
                    owner, repo, item["path"], token
                )
```

## Best Practices

```yaml
loading_best_practices:
  error_handling:
    - "Catch and log parsing errors"
    - "Continue processing on individual failures"
    - "Return partial results when possible"
  
  metadata_extraction:
    - "Capture source information"
    - "Extract creation/modification dates"
    - "Preserve document structure metadata"
  
  text_cleaning:
    - "Remove excessive whitespace"
    - "Handle encoding issues"
    - "Strip irrelevant content (nav, footer)"
  
  performance:
    - "Use streaming for large files"
    - "Implement caching for repeated loads"
    - "Process in batches for directories"
```

## Summary

```yaml
key_takeaways:
  - "Different formats need different parsing strategies"
  - "Preserve structure and metadata when possible"
  - "Use a universal loader for flexibility"
  - "Handle errors gracefully at scale"

supported_formats:
  documents: "PDF, DOCX, HTML, Markdown, Text"
  code: "Python, JavaScript, TypeScript, etc."
  apis: "Notion, Confluence, GitHub"
```

## Next Steps

1. **Chunking Strategies** - Split documents effectively
2. **Document Parsing** - Advanced parsing techniques
3. **Building Processor Lab** - Hands-on implementation
