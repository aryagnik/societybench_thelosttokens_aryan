#!/usr/bin/env python3
"""Collect an event's complete "finalized" set into <event>/final/.

Timestamped folders are immutable run snapshots; final/ is the fixed finalized
location. Missing core files raise an error (a code gate against omissions).
Also generates PROVENANCE.md recording the sources.

Two modes:
  full —— full-pipeline events like event4/5 (with an anon/ subdir, questionbank, evals)
    python3 finalize_event.py --mode full --event-dir runs_new/event5_smci \
      --anon-from 20260620_0327 --master-from 20260618_0611
  gt  —— events like event1/2/3 that only reached the GT + anonymized-GT stage
    python3 finalize_event.py --mode gt --event-dir runs_new/event1_library \
      --time 20260620_0618 --anon-subdir predict_ws2
"""
import argparse, shutil
from pathlib import Path
from datetime import datetime


def collect(final: Path, required, optional, prov_lines):
    missing = [str(src) for src, _, _ in required if not src.exists()]
    if missing:
        raise SystemExit("❌ 缺核心件，终止：\n  " + "\n  ".join(missing))
    for src, dst, kind in required + optional:
        if not src.exists():
            prov_lines.append(f"- (跳过，源不存在) {dst}")
            continue
        d = final / dst
        if kind == "dir":
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(src, d)
        else:
            shutil.copy2(src, d)
        prov_lines.append(f"- `{dst}` ← `{src.relative_to(final.parent)}`")


# Marker terms of editorial / question-authoring meta-comments: must never appear in benchmark data.
# (High-precision terms that essentially never occur in real news/opinion content; terms like '与事实不符'/'统计中'/'官方统计' cause false positives and are excluded)
EDITORIAL_PATTERNS = [
    # ① Proofreading / revision markers
    "原稿", "误记", "校勘", "笔误", "订正", "编者注", "译者注",
    "勘误", "源材料中并无", "此处应为", "原文误", "应更正为", "现据", "据百度百科",
    # ② Data-construction / question-authoring meta-info markers (stats blocks, source counts, augmentation/rebuild records, etc.)
    "**统计**", "来源条数", "事件块", "折入既有块", "本轮新增", "本轮重构",
    "raw_backup", "[WM]", "原GT", "补丰富", "增补后", "据原始采集", "多家权威媒体可证",
]


def scan_editorial(final: Path):
    """Freeze gate: scan finalized data for stray editorial/proofreading annotations; any hit raises and blocks the freeze.
    Covers timeline/contexts/gt/questionbank and A4 variants; skips results/, PROVENANCE, and backups."""
    hits = []
    for p in sorted(final.rglob("*")):
        if not p.is_file() or p.suffix not in (".md", ".json"):
            continue
        if "results" in p.parts or p.name == "PROVENANCE.md":
            continue
        if any(seg.startswith("_") and seg.endswith("bak") for seg in p.parts):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        for i, line in enumerate(text.splitlines(), 1):
            hit = next((pat for pat in EDITORIAL_PATTERNS if pat in line), None)
            if hit:
                hits.append(f"{p.relative_to(final)}:{i}  [{hit}]  {line.strip()[:80]}")
    if hits:
        raise SystemExit(
            "❌ 冻结门禁:发现编辑/校勘批注混入定稿数据(删除后再冻结):\n  "
            + "\n  ".join(hits[:40])
            + (f"\n  …(共 {len(hits)} 处)" if len(hits) > 40 else ""))
    print(f"✅ 编辑批注门禁:{final.name}/ 无校勘/编辑注释残留")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "gt"], default="full")
    ap.add_argument("--event-dir", required=True)
    # full
    ap.add_argument("--anon-from", help="full mode: timestamped folder containing anon/")
    ap.add_argument("--master-from", help="full mode: timestamped folder containing the non-anonymized master")
    ap.add_argument("--master-file", default="timeline_merged_long_gt.md")
    # gt
    ap.add_argument("--time", help="gt mode: the single timestamped folder, e.g. 20260620_0618")
    ap.add_argument("--anon-subdir", default="predict_ws", help="gt mode: subdirectory holding the anonymized GT")
    ap.add_argument("--note", default="")
    ap.add_argument("--scan-only", action="store_true",
                    help="Only run the editorial-annotation gate on the existing final/; do not re-collect")
    args = ap.parse_args()

    ev = Path(args.event_dir)
    final = ev / "final"
    if args.scan_only:
        if not final.exists():
            raise SystemExit(f"❌ {final} 不存在")
        scan_editorial(final)
        return
    final.mkdir(exist_ok=True)
    prov = [f"# {ev.name} 定稿来源（{datetime.now().strftime('%Y-%m-%d %H:%M')} 汇集 · mode={args.mode}）", ""]

    if args.mode == "full":
        anon = ev / args.anon_from / "anon"
        required = [
            (ev / args.master_from / args.master_file, args.master_file, "file"),
            (anon / "timeline_anon.md", "timeline_anon.md", "file"),
            (anon / "gt", "gt", "dir"),
            (anon / "contexts", "contexts", "dir"),
            (anon / "questionbank", "questionbank", "dir"),
            (anon / "prediction_points.json", "prediction_points.json", "file"),
            (anon / "replacements_effective.json", "replacements_effective.json", "file"),
            (anon / "results", "results", "dir"),
        ]
        # GT core files aligned with gt mode (keeps the 5 final/ dirs consistent)
        optional = [
            (anon / "fullanon_reaudit_report.json", "fullanon_reaudit_report.json", "file"),
            (ev / args.master_from / "timeline_merged_short.md", "timeline_short.md", "file"),
        ]
    else:  # gt
        t = ev / args.time
        sub = t / args.anon_subdir
        required = [
            (t / "media_workspace" / "timeline_merged_long_gt.md", "timeline_merged_long_gt.md", "file"),
            (sub / "timeline_anon.md", "timeline_anon.md", "file"),
        ]
        optional = [
            (sub / "fullanon_reaudit_report.json", "fullanon_reaudit_report.json", "file"),
            (sub / "replacements_effective.json", "replacements_effective.json", "file"),
            (t / "gt_workspace" / "timeline_short.md", "timeline_short.md", "file"),
        ]

    collect(final, required, optional, prov)
    prov += ["", "> 时间文件夹是不可变快照；本 final/ 是固定定稿位置。",
             "> 重出题/重评测后，重跑 finalize_event.py 覆盖刷新即可。"]
    if args.note:
        prov += ["", f"**备注**：{args.note}"]
    (final / "PROVENANCE.md").write_text("\n".join(prov), encoding="utf-8")

    scan_editorial(final)  # freeze gate: raise and block if editorial/proofreading annotations slipped in
    print(f"✅ final/ 汇集完成：{final}")
    for p in sorted(final.iterdir()):
        print(f"   {p.name}{'/' if p.is_dir() else ''}")


if __name__ == "__main__":
    main()
