"""Regenerate the README's "Tool Catalog Overview" table from the actual
registered tools, so it can't silently drift out of sync with the server
again. Run with: python scripts/gen_tool_catalog.py

Prints a markdown table to stdout; paste it into README.md between the
"## Tool Catalog Overview" heading and the next "---".
"""
import asyncio
import re

from ypotheto_compchem_mcp.config import settings
from ypotheto_compchem_mcp.server import create_server


def _first_line(description: str | None) -> str:
    if not description:
        return ""
    for line in description.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _param_names(input_schema: dict) -> str:
    props = (input_schema or {}).get("properties", {})
    names = list(props.keys())
    if not names:
        return "None"
    preview = ", ".join(f"`{n}`" for n in names[:3])
    if len(names) > 3:
        preview += ", ..."
    return preview


async def _main() -> None:
    bundle = create_server(settings)
    tools = await bundle.mcp.list_tools()
    tools = sorted(tools, key=lambda t: t.name)

    lines = ["| Tool Name | Parameters | Description |", "| :--- | :--- | :--- |"]
    for tool in tools:
        desc = _first_line(tool.description)
        # Collapse internal whitespace/newlines from multi-line docstrings.
        desc = re.sub(r"\s+", " ", desc)
        lines.append(f"| `{tool.name}` | {_param_names(tool.inputSchema)} | {desc} |")

    print("\n".join(lines))
    print(f"\n<!-- {len(tools)} tools -->")


if __name__ == "__main__":
    asyncio.run(_main())
