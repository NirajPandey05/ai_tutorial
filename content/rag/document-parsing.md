# Document Parsing: Extracting Content from Complex Formats

Master the art of extracting clean, structured text from PDFs, HTML, and other challenging formats.

## The Parsing Challenge

```yaml
real_world_documents:
  pdfs:
    - "Multi-column layouts"
    - "Headers and footers"
    - "Tables and figures"
    - "Scanned images (OCR needed)"
  
  html:
    - "Navigation and ads"
    - "JavaScript-rendered content"
    - "Nested structures"
    - "Boilerplate text"
  
  office_docs:
    - "Track changes and comments"
    - "Embedded objects"
    - "Complex formatting"

goal: "Extract clean, semantic content while preserving structure"
```

## PDF Parsing Deep Dive

### Basic PDF Extraction

```python
import fitz  # PyMuPDF


def extract_pdf_basic(path: str) -> str:
    """Basic PDF text extraction."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

### Advanced PDF Extraction

```python
from dataclasses import dataclass
from typing import Optional
import fitz
import re


@dataclass
class PDFPage:
    number: int
    text: str
    tables: list[list[list[str]]]
    images: list[dict]
    headers: list[str]
    footers: list[str]


@dataclass
class PDFDocument:
    title: str
    author: str
    pages: list[PDFPage]
    toc: list[dict]  # Table of contents


class AdvancedPDFParser:
    """Extract structured content from PDFs."""
    
    def __init__(
        self,
        extract_tables: bool = True,
        extract_images: bool = False,
        remove_headers_footers: bool = True
    ):
        self.extract_tables = extract_tables
        self.extract_images = extract_images
        self.remove_headers_footers = remove_headers_footers
    
    def parse(self, path: str) -> PDFDocument:
        """Parse PDF with full structure."""
        
        doc = fitz.open(path)
        
        # Get metadata
        metadata = doc.metadata
        
        # Get table of contents
        toc = [
            {"title": item[1], "page": item[2], "level": item[0]}
            for item in doc.get_toc()
        ]
        
        # Parse each page
        pages = []
        for page_num, page in enumerate(doc):
            parsed_page = self._parse_page(page, page_num + 1)
            pages.append(parsed_page)
        
        # Remove headers/footers if requested
        if self.remove_headers_footers:
            self._detect_and_remove_headers_footers(pages)
        
        return PDFDocument(
            title=metadata.get("title", ""),
            author=metadata.get("author", ""),
            pages=pages,
            toc=toc
        )
    
    def _parse_page(self, page: fitz.Page, page_num: int) -> PDFPage:
        """Parse a single page."""
        
        # Extract text blocks with positions
        blocks = page.get_text("dict")["blocks"]
        
        text_content = []
        for block in blocks:
            if block["type"] == 0:  # Text block
                block_text = ""
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        block_text += span["text"]
                    block_text += "\n"
                text_content.append(block_text.strip())
        
        # Extract tables
        tables = []
        if self.extract_tables:
            for table in page.find_tables():
                tables.append(table.extract())
        
        # Extract images
        images = []
        if self.extract_images:
            for img_index, img in enumerate(page.get_images()):
                xref = img[0]
                img_data = doc.extract_image(xref)
                images.append({
                    "index": img_index,
                    "width": img_data.get("width"),
                    "height": img_data.get("height"),
                    "format": img_data.get("ext")
                })
        
        return PDFPage(
            number=page_num,
            text="\n\n".join(text_content),
            tables=tables,
            images=images,
            headers=[],
            footers=[]
        )
    
    def _detect_and_remove_headers_footers(self, pages: list[PDFPage]) -> None:
        """Detect and remove repeated headers/footers."""
        
        if len(pages) < 3:
            return
        
        # Get first lines from each page
        first_lines = []
        last_lines = []
        
        for page in pages:
            lines = page.text.split("\n")
            lines = [l.strip() for l in lines if l.strip()]
            
            if lines:
                first_lines.append(lines[0] if lines else "")
                last_lines.append(lines[-1] if lines else "")
        
        # Find repeated patterns (headers/footers)
        header_pattern = self._find_repeated_pattern(first_lines)
        footer_pattern = self._find_repeated_pattern(last_lines)
        
        # Remove detected headers/footers
        for page in pages:
            if header_pattern:
                page.headers.append(header_pattern)
                page.text = self._remove_pattern(page.text, header_pattern, "start")
            if footer_pattern:
                page.footers.append(footer_pattern)
                page.text = self._remove_pattern(page.text, footer_pattern, "end")
    
    def _find_repeated_pattern(self, lines: list[str]) -> Optional[str]:
        """Find patterns that repeat across pages."""
        
        # Count occurrences (ignoring page numbers)
        patterns = {}
        for line in lines:
            # Normalize by removing numbers
            normalized = re.sub(r'\d+', '#', line)
            patterns[normalized] = patterns.get(normalized, 0) + 1
        
        # Pattern present in > 50% of pages is likely header/footer
        threshold = len(lines) * 0.5
        for pattern, count in patterns.items():
            if count >= threshold and len(pattern) > 5:
                return pattern
        
        return None
    
    def _remove_pattern(self, text: str, pattern: str, position: str) -> str:
        """Remove pattern from start or end of text."""
        lines = text.split("\n")
        
        if position == "start" and lines:
            normalized_first = re.sub(r'\d+', '#', lines[0])
            if normalized_first == pattern:
                lines = lines[1:]
        
        if position == "end" and lines:
            normalized_last = re.sub(r'\d+', '#', lines[-1])
            if normalized_last == pattern:
                lines = lines[:-1]
        
        return "\n".join(lines)
    
    def to_text(self, document: PDFDocument) -> str:
        """Convert parsed document to clean text."""
        
        sections = []
        
        for page in document.pages:
            page_content = f"[Page {page.number}]\n{page.text}"
            
            # Add tables as text
            for table in page.tables:
                table_text = self._table_to_text(table)
                page_content += f"\n\n[Table]\n{table_text}"
            
            sections.append(page_content)
        
        return "\n\n---\n\n".join(sections)
    
    def _table_to_text(self, table: list[list[str]]) -> str:
        """Convert table to readable text."""
        if not table:
            return ""
        
        # Simple markdown-style table
        lines = []
        for row in table:
            lines.append(" | ".join(str(cell) for cell in row))
        
        return "\n".join(lines)


# Usage
parser = AdvancedPDFParser(
    extract_tables=True,
    extract_images=False,
    remove_headers_footers=True
)

document = parser.parse("annual_report.pdf")
clean_text = parser.to_text(document)
print(f"Extracted {len(document.pages)} pages")
```

### OCR for Scanned PDFs

```python
import fitz
from PIL import Image
import pytesseract
import io


def extract_pdf_with_ocr(path: str) -> str:
    """Extract text from PDF, using OCR for scanned pages."""
    
    doc = fitz.open(path)
    full_text = []
    
    for page_num, page in enumerate(doc):
        # Try regular text extraction first
        text = page.get_text()
        
        # If little text found, try OCR
        if len(text.strip()) < 100:
            # Render page as image
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR the image
            text = pytesseract.image_to_string(img)
        
        full_text.append(f"[Page {page_num + 1}]\n{text}")
    
    return "\n\n".join(full_text)
```

## HTML Parsing Deep Dive

### Basic HTML Extraction

```python
from bs4 import BeautifulSoup
import requests


def extract_html_basic(source: str) -> str:
    """Basic HTML text extraction."""
    
    if source.startswith("http"):
        html = requests.get(source, timeout=30).text
    else:
        with open(source, "r", encoding="utf-8") as f:
            html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove unwanted elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    
    return soup.get_text(separator="\n", strip=True)
```

### Advanced HTML Extraction

```python
from bs4 import BeautifulSoup, NavigableString
from dataclasses import dataclass
from typing import Optional
import requests
from urllib.parse import urljoin


@dataclass
class HTMLContent:
    title: str
    main_content: str
    structured_sections: list[dict]
    links: list[dict]
    metadata: dict


class AdvancedHTMLParser:
    """Extract clean, structured content from HTML."""
    
    BOILERPLATE_CLASSES = [
        "nav", "navigation", "menu", "sidebar", "footer", "header",
        "advertisement", "ad", "social", "share", "comment", "related"
    ]
    
    CONTENT_SELECTORS = [
        "article",
        "main",
        '[role="main"]',
        ".content",
        ".post-content",
        ".article-content",
        ".entry-content",
        "#content"
    ]
    
    def __init__(
        self,
        remove_boilerplate: bool = True,
        extract_links: bool = True,
        preserve_structure: bool = True
    ):
        self.remove_boilerplate = remove_boilerplate
        self.extract_links = extract_links
        self.preserve_structure = preserve_structure
    
    def parse(self, source: str) -> HTMLContent:
        """Parse HTML from URL or file."""
        
        if source.startswith("http"):
            html = requests.get(source, timeout=30).text
            base_url = source
        else:
            with open(source, "r", encoding="utf-8") as f:
                html = f.read()
            base_url = None
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Extract metadata
        metadata = self._extract_metadata(soup)
        
        # Remove boilerplate
        if self.remove_boilerplate:
            self._remove_boilerplate(soup)
        
        # Find main content
        main_element = self._find_main_content(soup)
        
        # Extract structured sections
        sections = []
        if self.preserve_structure:
            sections = self._extract_sections(main_element)
        
        # Extract links
        links = []
        if self.extract_links:
            links = self._extract_links(main_element, base_url)
        
        return HTMLContent(
            title=self._extract_title(soup),
            main_content=main_element.get_text(separator="\n", strip=True),
            structured_sections=sections,
            links=links,
            metadata=metadata
        )
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try OpenGraph title first
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return og_title["content"]
        
        # Fall back to title tag
        if soup.title:
            return soup.title.string or ""
        
        # Fall back to h1
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        
        return ""
    
    def _extract_metadata(self, soup: BeautifulSoup) -> dict:
        """Extract page metadata."""
        metadata = {}
        
        # Description
        desc = soup.find("meta", attrs={"name": "description"})
        if desc:
            metadata["description"] = desc.get("content", "")
        
        # Keywords
        keywords = soup.find("meta", attrs={"name": "keywords"})
        if keywords:
            metadata["keywords"] = keywords.get("content", "")
        
        # Author
        author = soup.find("meta", attrs={"name": "author"})
        if author:
            metadata["author"] = author.get("content", "")
        
        # Published date
        date = soup.find("meta", attrs={"property": "article:published_time"})
        if date:
            metadata["published"] = date.get("content", "")
        
        return metadata
    
    def _remove_boilerplate(self, soup: BeautifulSoup) -> None:
        """Remove boilerplate elements."""
        
        # Remove by tag name
        for tag in ["script", "style", "nav", "footer", "header", "aside", "noscript"]:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove by class name
        for class_name in self.BOILERPLATE_CLASSES:
            for element in soup.find_all(class_=lambda x: x and class_name in x.lower()):
                element.decompose()
        
        # Remove by ID
        for element in soup.find_all(id=lambda x: x and any(
            bp in x.lower() for bp in self.BOILERPLATE_CLASSES
        )):
            element.decompose()
        
        # Remove hidden elements
        for element in soup.find_all(style=lambda x: x and "display:none" in x.replace(" ", "")):
            element.decompose()
    
    def _find_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Find the main content area."""
        
        # Try known content selectors
        for selector in self.CONTENT_SELECTORS:
            element = soup.select_one(selector)
            if element and len(element.get_text(strip=True)) > 100:
                return element
        
        # Fall back to body or entire soup
        return soup.body or soup
    
    def _extract_sections(self, element: BeautifulSoup) -> list[dict]:
        """Extract content organized by headers."""
        
        sections = []
        current_section = {"heading": None, "level": 0, "content": []}
        
        for child in element.descendants:
            if isinstance(child, NavigableString):
                continue
            
            if child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                # Save previous section
                if current_section["content"]:
                    sections.append(current_section.copy())
                
                level = int(child.name[1])
                current_section = {
                    "heading": child.get_text(strip=True),
                    "level": level,
                    "content": []
                }
            
            elif child.name in ["p", "li", "td", "pre", "blockquote"]:
                text = child.get_text(strip=True)
                if text and text not in current_section["content"]:
                    current_section["content"].append(text)
        
        # Don't forget last section
        if current_section["content"]:
            sections.append(current_section)
        
        return sections
    
    def _extract_links(
        self,
        element: BeautifulSoup,
        base_url: Optional[str]
    ) -> list[dict]:
        """Extract links with context."""
        
        links = []
        for a in element.find_all("a", href=True):
            href = a["href"]
            
            # Make absolute URL
            if base_url and not href.startswith(("http://", "https://", "mailto:", "#")):
                href = urljoin(base_url, href)
            
            links.append({
                "text": a.get_text(strip=True),
                "url": href,
                "title": a.get("title", "")
            })
        
        return links


# Usage
parser = AdvancedHTMLParser(
    remove_boilerplate=True,
    extract_links=True,
    preserve_structure=True
)

content = parser.parse("https://example.com/article")
print(f"Title: {content.title}")
print(f"Sections: {len(content.structured_sections)}")
print(f"Links: {len(content.links)}")
```

### JavaScript-Rendered Content

```python
from playwright.sync_api import sync_playwright


def extract_js_rendered(url: str, wait_selector: str = None) -> str:
    """Extract content from JavaScript-rendered pages."""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to page
        page.goto(url)
        
        # Wait for content to load
        if wait_selector:
            page.wait_for_selector(wait_selector)
        else:
            page.wait_for_load_state("networkidle")
        
        # Get rendered HTML
        html = page.content()
        
        browser.close()
    
    # Parse with BeautifulSoup
    parser = AdvancedHTMLParser()
    return parser.parse_html_string(html)


# Alternative using Selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


def extract_js_selenium(url: str) -> str:
    """Extract using Selenium."""
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    
    # Wait for dynamic content
    import time
    time.sleep(3)
    
    html = driver.page_source
    driver.quit()
    
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)
```

## Universal Parser

```python
from pathlib import Path
from typing import Union


class UniversalParser:
    """Parse any supported document format."""
    
    def __init__(self):
        self.pdf_parser = AdvancedPDFParser()
        self.html_parser = AdvancedHTMLParser()
    
    def parse(self, source: str) -> dict:
        """Parse document and return standardized format."""
        
        # Detect format
        if source.startswith(("http://", "https://")):
            return self._parse_url(source)
        
        path = Path(source)
        suffix = path.suffix.lower()
        
        parsers = {
            ".pdf": self._parse_pdf,
            ".html": self._parse_html,
            ".htm": self._parse_html,
            ".md": self._parse_markdown,
            ".txt": self._parse_text,
            ".docx": self._parse_docx,
        }
        
        parser = parsers.get(suffix)
        if not parser:
            raise ValueError(f"Unsupported format: {suffix}")
        
        return parser(source)
    
    def _parse_pdf(self, path: str) -> dict:
        doc = self.pdf_parser.parse(path)
        return {
            "content": self.pdf_parser.to_text(doc),
            "format": "pdf",
            "title": doc.title,
            "metadata": {"author": doc.author, "pages": len(doc.pages)}
        }
    
    def _parse_html(self, path: str) -> dict:
        content = self.html_parser.parse(path)
        return {
            "content": content.main_content,
            "format": "html",
            "title": content.title,
            "metadata": content.metadata
        }
    
    def _parse_url(self, url: str) -> dict:
        content = self.html_parser.parse(url)
        return {
            "content": content.main_content,
            "format": "html",
            "title": content.title,
            "metadata": {**content.metadata, "url": url}
        }
    
    def _parse_markdown(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "content": content,
            "format": "markdown",
            "title": Path(path).stem,
            "metadata": {}
        }
    
    def _parse_text(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {
            "content": content,
            "format": "text",
            "title": Path(path).stem,
            "metadata": {}
        }
    
    def _parse_docx(self, path: str) -> dict:
        from docx import Document
        doc = Document(path)
        content = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return {
            "content": content,
            "format": "docx",
            "title": Path(path).stem,
            "metadata": {}
        }
```

## Best Practices

```yaml
pdf_parsing:
  - "Check if PDF is text-based or scanned"
  - "Handle multi-column layouts carefully"
  - "Preserve table structure when possible"
  - "Remove headers/footers for cleaner chunks"

html_parsing:
  - "Remove boilerplate (nav, footer, ads)"
  - "Find the main content area"
  - "Handle JavaScript-rendered content"
  - "Extract metadata for filtering"

general:
  - "Validate encoding (UTF-8 preferred)"
  - "Handle errors gracefully"
  - "Log problematic documents"
  - "Preserve source information"
```

## Summary

```yaml
key_takeaways:
  - "Different formats need specialized parsers"
  - "Clean extraction improves RAG quality"
  - "Preserve structure for better chunking"
  - "Handle edge cases (scanned PDFs, JS pages)"
```

## Next Steps

1. **Document Processor Lab** - Build a complete pipeline
2. **Retrieval Strategies** - Improve search quality
3. **RAG Evaluation** - Measure your pipeline
