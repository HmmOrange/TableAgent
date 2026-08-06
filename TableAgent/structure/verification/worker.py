from __future__ import annotations

import json
import sys

from TableAgent.structure.verification.checks import verify_structure


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except AttributeError:
        pass
    workbook_path, sheet_name, structure_path = sys.argv[1:4]
    data_only = len(sys.argv) > 4 and sys.argv[4].strip().lower() == "true"
    print(
        json.dumps(
            verify_structure(
                workbook_path,
                sheet_name,
                structure_path,
                data_only=data_only,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
