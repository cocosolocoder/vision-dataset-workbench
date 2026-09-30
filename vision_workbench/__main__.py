from __future__ import annotations

import argparse
import json
from pathlib import Path

from .store import DatasetStore


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="vision-workbench")
    commands = value.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="initialize a dataset workspace")
    initialize.add_argument("workspace", type=Path)

    add = commands.add_parser("add", help="add an image record")
    add.add_argument("workspace", type=Path)
    add.add_argument("source", type=Path)
    add.add_argument("--label")

    summary = commands.add_parser("summary", help="show dataset statistics")
    summary.add_argument("workspace", type=Path)

    demo = commands.add_parser("demo", help="show a read-only product demonstration")
    demo.add_argument("--workspace", type=Path, default=Path("."))
    return value


def main() -> int:
    args = parser().parse_args()
    store = DatasetStore(args.workspace)
    if args.command == "init":
        store.initialize()
        print(f"Initialized dataset workspace: {store.root}")
    elif args.command == "add":
        result = store.add(args.source, args.label)
        status = "added" if result.added else "already present"
        print(f"{status}: {result.digest}")
    elif args.command == "summary":
        print(json.dumps(store.summary(), ensure_ascii=False, sort_keys=True))
    else:
        print("视觉数据集工作台")
        print("本地 CPU 模式：可用")
        print("当前数据集摘要：")
        print(json.dumps(store.summary(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
