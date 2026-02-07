"""
Markdown Renderer - Render markdown content with syntax highlighting.
"""

import re
from typing import Optional
import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension
from pygments import highlight
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.formatters import HtmlFormatter


class MarkdownRenderer:
    """Render markdown content to HTML with syntax highlighting."""
    
    def __init__(self):
        self.md = markdown.Markdown(
            extensions=[
                'fenced_code',
                'codehilite',
                'tables',
                'toc',
                'nl2br',
                'sane_lists',
                'smarty',
                CodeHiliteExtension(css_class='highlight', linenums=False),
                TocExtension(permalink=True),
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'linenums': False,
                    'guess_lang': True,
                }
            }
        )
    
    def render(self, content: str) -> str:
        """Render markdown to HTML."""
        self.md.reset()
        html = self.md.convert(content)
        return self._post_process(html)
    
    def _post_process(self, html: str) -> str:
        """Post-process HTML for additional features."""
        # Add target="_blank" to external links
        html = re.sub(
            r'<a href="(https?://[^"]+)"',
            r'<a href="\1" target="_blank" rel="noopener noreferrer"',
            html
        )
        
        # Wrap code blocks with copy button container
        html = re.sub(
            r'<div class="highlight"><pre>',
            r'<div class="code-block-wrapper"><button class="copy-btn" onclick="copyCode(this)">📋 Copy</button><div class="highlight"><pre>',
            html
        )
        html = re.sub(
            r'</pre></div>',
            r'</pre></div></div>',
            html
        )
        
        return html
    
    def get_toc(self) -> str:
        """Get table of contents HTML."""
        return self.md.toc
    
    def get_toc_tokens(self) -> list:
        """Get table of contents as tokens."""
        return self.md.toc_tokens


def highlight_code(code: str, language: str = "python") -> str:
    """Highlight code snippet with Pygments."""
    try:
        lexer = get_lexer_by_name(language)
    except:
        try:
            lexer = guess_lexer(code)
        except:
            lexer = get_lexer_by_name("text")
    
    formatter = HtmlFormatter(
        cssclass='highlight',
        linenos=False,
        nowrap=False,
    )
    
    return highlight(code, lexer, formatter)


def render_content(content: str) -> str:
    """Convenience function to render markdown content."""
    renderer = MarkdownRenderer()
    return renderer.render(content)


# CSS for syntax highlighting (Monokai-inspired dark theme)
HIGHLIGHT_CSS = """
.code-block-wrapper {
    position: relative;
    margin: 1rem 0;
}

.copy-btn {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    padding: 0.25rem 0.5rem;
    font-size: 0.75rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 0.25rem;
    color: #a0a0a0;
    cursor: pointer;
    transition: all 0.2s;
    z-index: 10;
}

.copy-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    color: white;
}

.highlight {
    background: #1e1e2e;
    border-radius: 0.5rem;
    padding: 1rem;
    overflow-x: auto;
}

.highlight pre {
    margin: 0;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.875rem;
    line-height: 1.5;
}

/* Syntax colors (Monokai-inspired) */
.highlight .hll { background-color: #3d3d3d }
.highlight .c { color: #6a737d } /* Comment */
.highlight .k { color: #ff79c6 } /* Keyword */
.highlight .n { color: #f8f8f2 } /* Name */
.highlight .o { color: #ff79c6 } /* Operator */
.highlight .p { color: #f8f8f2 } /* Punctuation */
.highlight .cm { color: #6a737d } /* Comment.Multiline */
.highlight .cp { color: #ff79c6 } /* Comment.Preproc */
.highlight .c1 { color: #6a737d } /* Comment.Single */
.highlight .cs { color: #6a737d } /* Comment.Special */
.highlight .gd { color: #ff5555 } /* Generic.Deleted */
.highlight .gi { color: #50fa7b } /* Generic.Inserted */
.highlight .ge { font-style: italic } /* Generic.Emph */
.highlight .gs { font-weight: bold } /* Generic.Strong */
.highlight .gu { color: #6a737d } /* Generic.Subheading */
.highlight .kc { color: #ff79c6 } /* Keyword.Constant */
.highlight .kd { color: #8be9fd } /* Keyword.Declaration */
.highlight .kn { color: #ff79c6 } /* Keyword.Namespace */
.highlight .kp { color: #ff79c6 } /* Keyword.Pseudo */
.highlight .kr { color: #ff79c6 } /* Keyword.Reserved */
.highlight .kt { color: #8be9fd } /* Keyword.Type */
.highlight .ld { color: #f1fa8c } /* Literal.Date */
.highlight .m { color: #bd93f9 } /* Literal.Number */
.highlight .s { color: #f1fa8c } /* Literal.String */
.highlight .na { color: #50fa7b } /* Name.Attribute */
.highlight .nb { color: #8be9fd } /* Name.Builtin */
.highlight .nc { color: #50fa7b } /* Name.Class */
.highlight .no { color: #bd93f9 } /* Name.Constant */
.highlight .nd { color: #50fa7b } /* Name.Decorator */
.highlight .ni { color: #f8f8f2 } /* Name.Entity */
.highlight .ne { color: #ff5555 } /* Name.Exception */
.highlight .nf { color: #50fa7b } /* Name.Function */
.highlight .nl { color: #f8f8f2 } /* Name.Label */
.highlight .nn { color: #f8f8f2 } /* Name.Namespace */
.highlight .nx { color: #50fa7b } /* Name.Other */
.highlight .py { color: #f8f8f2 } /* Name.Property */
.highlight .nt { color: #ff79c6 } /* Name.Tag */
.highlight .nv { color: #f8f8f2 } /* Name.Variable */
.highlight .ow { color: #ff79c6 } /* Operator.Word */
.highlight .w { color: #f8f8f2 } /* Text.Whitespace */
.highlight .mf { color: #bd93f9 } /* Literal.Number.Float */
.highlight .mh { color: #bd93f9 } /* Literal.Number.Hex */
.highlight .mi { color: #bd93f9 } /* Literal.Number.Integer */
.highlight .mo { color: #bd93f9 } /* Literal.Number.Oct */
.highlight .sb { color: #f1fa8c } /* Literal.String.Backtick */
.highlight .sc { color: #f1fa8c } /* Literal.String.Char */
.highlight .sd { color: #f1fa8c } /* Literal.String.Doc */
.highlight .s2 { color: #f1fa8c } /* Literal.String.Double */
.highlight .se { color: #ff79c6 } /* Literal.String.Escape */
.highlight .sh { color: #f1fa8c } /* Literal.String.Heredoc */
.highlight .si { color: #f1fa8c } /* Literal.String.Interpol */
.highlight .sx { color: #f1fa8c } /* Literal.String.Other */
.highlight .sr { color: #f1fa8c } /* Literal.String.Regex */
.highlight .s1 { color: #f1fa8c } /* Literal.String.Single */
.highlight .ss { color: #f1fa8c } /* Literal.String.Symbol */
.highlight .bp { color: #f8f8f2 } /* Name.Builtin.Pseudo */
.highlight .vc { color: #f8f8f2 } /* Name.Variable.Class */
.highlight .vg { color: #f8f8f2 } /* Name.Variable.Global */
.highlight .vi { color: #f8f8f2 } /* Name.Variable.Instance */
.highlight .il { color: #bd93f9 } /* Literal.Number.Integer.Long */
"""
