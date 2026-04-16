#!/usr/bin/env python3
"""将仓库内 Markdown 文档导出为带图片的 PDF。

该脚本面向本仓库的实验报告场景，重点解决两个问题：
1. Markdown 中的相对图片路径需要在导出时解析为绝对本地 URI。
2. 中文报告需要显式指定可用字体，否则默认字体常会丢失汉字。
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown
from weasyprint import CSS, HTML


ROOT_DIR = Path(__file__).resolve().parents[1]
IMAGE_PATTERN = re.compile(r"!\[(.*?)\]\((.*?)\)")
LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
WEB_PREFIXES = ("http://", "https://", "file://", "mailto:", "#")


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的命令行参数。
    """
    parser = argparse.ArgumentParser(description="将 Markdown 报告导出为 PDF。")
    parser.add_argument("input", type=Path, help="Markdown 文档路径。")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 PDF 路径；默认与输入同名，仅扩展名改为 .pdf。",
    )
    return parser.parse_args()


def resolve_repo_relative(path_text: str, markdown_path: Path) -> Path:
    """把 Markdown 中的相对路径解析为实际文件路径。

    Args:
        path_text: Markdown 链接里的原始路径文本。
        markdown_path: 当前 Markdown 文件路径。

    Returns:
        解析后的绝对路径。
    """
    if path_text.startswith("./work_dir/"):
        return (ROOT_DIR / path_text[2:]).resolve()
    return (markdown_path.parent / path_text).resolve()


def rewrite_image_links(markdown_text: str, markdown_path: Path) -> tuple[str, int]:
    """将图片路径重写为本地 file URI。

    Args:
        markdown_text: 原始 Markdown 文本。
        markdown_path: 当前 Markdown 文件路径。

    Returns:
        重写后的 Markdown 文本，以及成功重写的图片数量。
    """
    rewritten = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal rewritten
        alt_text, raw_target = match.groups()
        target = resolve_repo_relative(raw_target.strip(), markdown_path)
        rewritten += 1
        return f"![{alt_text}]({target.as_uri()})"

    return IMAGE_PATTERN.sub(replace, markdown_text), rewritten


def rewrite_document_links(markdown_text: str, markdown_path: Path) -> str:
    """将本地文档链接重写为 file URI，避免 PDF 中链接失效。

    Args:
        markdown_text: 已处理过图片路径的 Markdown 文本。
        markdown_path: 当前 Markdown 文件路径。

    Returns:
        重写后的 Markdown 文本。
    """

    def replace(match: re.Match[str]) -> str:
        label, raw_target = match.groups()
        target_text = raw_target.strip()
        if target_text.startswith(WEB_PREFIXES):
            return match.group(0)
        target = resolve_repo_relative(target_text, markdown_path)
        return f"[{label}]({target.as_uri()})"

    return LINK_PATTERN.sub(replace, markdown_text)


def build_html_document(markdown_text: str, title: str) -> str:
    """把 Markdown 转成带样式的完整 HTML 文档。

    Args:
        markdown_text: 已重写链接后的 Markdown 文本。
        title: HTML 文档标题。

    Returns:
        完整 HTML 字符串。
    """
    body = markdown.markdown(
        markdown_text,
        extensions=["extra", "fenced_code", "tables", "toc"],
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
@page {{
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}}
body {{
  font-family: "Noto Sans CJK SC", "Source Han Sans SC",
      "Droid Sans Fallback", "DejaVu Sans", sans-serif;
  font-size: 11pt;
  line-height: 1.6;
  color: #222;
}}
h1, h2, h3, h4 {{
  font-weight: 700;
  line-height: 1.3;
  margin-top: 1.2em;
  margin-bottom: 0.45em;
}}
h1 {{
  font-size: 22pt;
  border-bottom: 1px solid #ddd;
  padding-bottom: 0.25em;
}}
h2 {{ font-size: 16pt; }}
h3 {{ font-size: 13pt; }}
p, ul, ol, table, pre, blockquote {{
  margin-top: 0.5em;
  margin-bottom: 0.7em;
}}
img {{
  display: block;
  max-width: 100%;
  margin: 0.8em auto 1em auto;
  break-inside: avoid;
}}
table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 10pt;
}}
th, td {{
  border: 1px solid #cfcfcf;
  padding: 6px 8px;
  vertical-align: top;
}}
th {{ background: #f4f4f4; }}
code {{
  font-family: "DejaVu Sans Mono", monospace;
  font-size: 0.92em;
  background: #f6f6f6;
  padding: 0.08em 0.25em;
}}
pre code {{
  display: block;
  padding: 0.8em;
  overflow-wrap: anywhere;
  white-space: pre-wrap;
}}
a {{
  color: #1b5e9a;
  text-decoration: none;
}}
blockquote {{
  border-left: 3px solid #d0d0d0;
  padding-left: 0.8em;
  color: #555;
}}
</style>
</head>
<body>
{body}
</body>
</html>"""


def export_pdf(input_path: Path, output_path: Path) -> int:
    """执行 Markdown -> PDF 导出。

    Args:
        input_path: 输入 Markdown 路径。
        output_path: 输出 PDF 路径。

    Returns:
        报告中成功重写的图片数量。
    """
    markdown_text = input_path.read_text(encoding="utf-8")
    markdown_text, image_count = rewrite_image_links(markdown_text, input_path)
    markdown_text = rewrite_document_links(markdown_text, input_path)
    html_document = build_html_document(markdown_text, input_path.stem)
    HTML(string=html_document, base_url=str(ROOT_DIR)).write_pdf(
        output_path,
        stylesheets=[CSS(string="img { image-rendering: auto; }")],
    )
    return image_count


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve() if args.output else input_path.with_suffix(".pdf")
    image_count = export_pdf(input_path, output_path)
    print(f"exported: {output_path}")
    print(f"images_embedded: {image_count}")


if __name__ == "__main__":
    main()
