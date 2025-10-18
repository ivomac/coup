def to_markdown(data: list[dict]) -> str:
    """Converts a list of dictionaries into a formatted Markdown table string."""

    if not data:
        return ""

    headers = list(data[0].keys())

    str_rows = []
    for record in data:
        str_rows.append([str(record.get(key, "")) for key in headers])

    column_widths = [max(len(header), 3) for header in headers]

    for row in str_rows:
        for i, cell in enumerate(row):
            column_widths[i] = max(column_widths[i], len(cell))

    header_cells = [
        f" {headers[i].ljust(column_widths[i])} " for i in range(len(headers))
    ]
    markdown_header = "| " + " | ".join(header_cells) + " |"

    separator_cells = ["-" * column_widths[i] for i in range(len(headers))]
    markdown_separator = "|--" + "--|--".join(separator_cells) + "--|"

    markdown_rows = []
    for row in str_rows:
        data_cells = [
            f" {row[i].ljust(column_widths[i])} " for i in range(len(headers))
        ]
        markdown_rows.append("| " + " | ".join(data_cells) + " |")

    return markdown_header + "\n" + markdown_separator + "\n" + "\n".join(markdown_rows)
