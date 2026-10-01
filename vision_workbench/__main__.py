from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .batches import BatchError
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

    split = commands.add_parser(
        "split", help="create and inspect train/validation/test plans"
    )
    split_commands = split.add_subparsers(dest="split_command", required=True)

    create = split_commands.add_parser("create", help="create a named split plan")
    create.add_argument("workspace", type=Path)
    create.add_argument("name", help="plan name (no '/', '\\', '.' or '..')")
    create.add_argument("--seed", type=int, required=True, help="integer random seed")
    create.add_argument("--train", required=True, help="training ratio in [0, 1]")
    create.add_argument("--validation", required=True, help="validation ratio in [0, 1]")
    create.add_argument("--test", required=True, help="test ratio in [0, 1]")

    show = split_commands.add_parser("show", help="show a saved split plan as JSON")
    show.add_argument("workspace", type=Path)
    show.add_argument("name", help="plan name")

    label = commands.add_parser(
        "label", help="inspect and batch-modify classification labels"
    )
    label_commands = label.add_subparsers(dest="label_command", required=True)

    apply_parser = label_commands.add_parser(
        "apply", help="apply a batch of label changes from a UTF-8 JSON file"
    )
    apply_parser.add_argument("workspace", type=Path)
    apply_parser.add_argument("batch_file", type=Path, help="UTF-8 JSON batch file")

    label_show = label_commands.add_parser(
        "show", help="show the current label of one sample"
    )
    label_show.add_argument("workspace", type=Path)
    label_show.add_argument("sha256", help="full SHA-256 digest of the sample")

    history = label_commands.add_parser(
        "history", help="list successfully applied label batches"
    )
    history.add_argument("workspace", type=Path)

    undo = label_commands.add_parser("undo", help="undo a successful label batch")
    undo.add_argument("workspace", type=Path)
    undo.add_argument("batch_id", help="id of the batch to undo")

    demo = commands.add_parser("demo", help="show a read-only product demonstration")
    demo.add_argument("--workspace", type=Path, default=Path("."))
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        if args.command == "split":
            return _run_split(args)
        if args.command == "label":
            return _run_label(args)
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
    except BatchError as error:
        _print_batch_error(error)
        return 1
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _print_batch_error(error: BatchError) -> None:
    payload = {"error": str(error), "code": error.code}
    if error.records:
        payload["records"] = error.records
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def _run_split(args: argparse.Namespace) -> int:
    store = DatasetStore(args.workspace)
    if args.split_command == "create":
        result = store.create_split(
            args.name,
            args.seed,
            [args.train, args.validation, args.test],
        )
        plan = result.plan
        output = {
            "name": result.name,
            "status": "created" if result.created else "already exists",
            "seed": plan["seed"],
            "ratios": plan["ratios"],
            "samples": plan["samples"],
            "sets": {
                set_name: {
                    "samples": plan["sets"][set_name]["samples"],
                    "distribution": plan["sets"][set_name]["distribution"],
                }
                for set_name in plan["sets"]
            },
        }
    else:
        output = store.get_split(args.name)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


def _run_label(args: argparse.Namespace) -> int:
    store = DatasetStore(args.workspace)
    if args.label_command == "apply":
        try:
            text = args.batch_file.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise BatchError(
                f"Batch file is not valid UTF-8: {error}", code="invalid_document"
            ) from None
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise BatchError(
                f"Batch file is not valid JSON: {error}", code="invalid_document"
            ) from None
        result = store.apply_batch(payload)
        output = {
            "batch_id": result.batch_id,
            "status": result.status,
            "changed": result.changed,
            "total": result.total,
            "changes": list(result.changes),
        }
    elif args.label_command == "show":
        digest = args.sha256
        output = {"sha256": digest, "label": store.get_label(digest)}
    elif args.label_command == "history":
        output = {"batches": store.batch_history()}
    else:
        result = store.undo_batch(args.batch_id)
        output = {
            "batch_id": result.batch_id,
            "status": result.status,
            "restored": result.restored,
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
