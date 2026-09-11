# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_PATH = DATA_DIR / "all_data_platforms.csv"
OUTPUT_PATH = DATA_DIR / "valid_data_step4.csv"

FIELDS = [
    "platform",
    "type",
    "id",
    "parent_id",
    "parent_comment_id",
    "user_id",
    "user_name",
    "title",
    "content",
    "create_time",
    "create_timestamp",
    "liked_count",
    "comment_count",
    "share_count",
    "ip_location",
    "source_keyword",
    "url",
    "validation_status",
    "validation_reason",
]

ENTITY_PAT = re.compile(r"tiktok|字节跳动|bytedance", re.I)
US_PAT = re.compile(
    r"美国|u\.s\.|united states|特朗普|trump|拜登|biden|白宫|white house|"
    r"国会|众议院|参议院|congress|senate|house|最高法院|supreme court|"
    r"上诉法院|appeals court|garland|app store|google|apple|oracle",
    re.I,
)
ACTION_PAT = re.compile(
    r"禁令|封禁|不卖就禁|下架|出售|剥离|延期|恢复|交易|法案|诉讼|起诉|裁决|"
    r"ban|banned|shutdown|sale|sell|divest|divestiture|deadline|extension|extend|"
    r"restore|restoring|deal|lawsuit|sue|sues|uphold|upholds|upheld|executive order",
    re.I,
)
NOISE_PAT = re.compile(
    r"tiktok shop|直播带货|带货|营销|教程|vpn|翻墙|广告投放|"
    r"screen time|青少年|未成年人|shop|monetization|creator fund",
    re.I,
)
TARGET_KEYWORDS = {
    "TikTok美国禁令",
    "TikTok不卖就禁",
    "TikTok美国下架",
    "TikTok美国恢复",
    "TikTok禁令延期",
    "特朗普封禁TikTok",
    "特朗普 TikTok 延期",
    "拜登 TikTok 法案",
    "字节跳动出售TikTok",
    "TikTok Supreme Court",
    "TikTok sale US",
    "TikTok 美国新实体",
}


def classify_post(row: dict) -> tuple[str, str]:
    title = (row.get("title") or "").strip()
    content = (row.get("content") or "").strip()
    source_keyword = (row.get("source_keyword") or "").strip()
    text = " ".join(part for part in [title, content, source_keyword] if part)

    if len(text) < 8:
        return "auto_invalid", "too_short"
    if not ENTITY_PAT.search(text):
        return "auto_invalid", "missing_entity"
    if source_keyword in TARGET_KEYWORDS and (US_PAT.search(text) or ACTION_PAT.search(text)):
        if NOISE_PAT.search(text) and not US_PAT.search(text):
            return "auto_invalid", "generic_noise"
        return "auto_valid", "target_keyword_plus_context"
    if ENTITY_PAT.search(text) and US_PAT.search(text) and ACTION_PAT.search(text):
        return "auto_valid", "entity_us_action"
    if NOISE_PAT.search(text):
        return "auto_invalid", "noise_pattern"
    return "uncertain", "needs_review"


def main() -> None:
    rows = list(csv.DictReader(INPUT_PATH.open("r", encoding="utf-8-sig", newline="")))
    out = []
    counts = {"auto_valid": 0, "auto_invalid": 0, "uncertain": 0}

    for row in rows:
        if row.get("type") != "post":
            continue
        status, reason = classify_post(row)
        counts[status] += 1
        new_row = {field: row.get(field, "") for field in FIELDS if field in row}
        new_row["validation_status"] = status
        new_row["validation_reason"] = reason
        out.append(new_row)

    with OUTPUT_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)

    print(f"Wrote {len(out)} posts to {OUTPUT_PATH}")
    for key, value in counts.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
