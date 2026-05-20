"""
Post-process graph_edges.jsonl: normalize character names in existing edges.
Much faster than re-running LLM extraction.

Run:
    uv run python experiments/ragm_novel/normalize_character_edges.py
    uv run python experiments/ragm_novel/normalize_character_edges.py --dry-run
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DB_PATH = "./chroma_novel"
GRAPH_EDGE_FILE = "graph_edges.jsonl"

ALIAS_MAP: dict[str, str] = {
    # 綾小路清隆
    "清隆":           "綾小路",
    "綾小路清隆":     "綾小路",
    "綾小路君":       "綾小路",
    "綾小路同學":     "綾小路",
    "小清":           "綾小路",
    "綾小路 (我)":    "綾小路",
    "我 (綾小路)":    "綾小路",
    "清隆 (綾小路)":  "綾小路",
    "綾小路 (主角)":  "綾小路",
    "綾小路 (自稱)":  "綾小路",
    # 堀北鈴音
    "鈴音":           "堀北",
    "堀北鈴音":       "堀北",
    "堀北同學":       "堀北",
    "鈴音 (堀北)":    "堀北",
    # 龍園翔也
    "龍園翔":         "龍園",
    "龍園君":         "龍園",
    "龍園同學":       "龍園",
    "龍園 (提及)":    "龍園",
    # 一之瀨帆波
    "帆波":           "一之瀨",
    "一之瀨帆波":     "一之瀨",
    "小帆波":         "一之瀨",
    "一之瀨同學":     "一之瀨",
    "一之瀬":         "一之瀨",
    # 平田洋介
    "洋介":           "平田",
    "平田洋介":       "平田",
    "平田君":         "平田",
    # 輕井澤惠
    "惠":             "輕井澤",
    "輕井澤惠":       "輕井澤",
    "輕井澤同學":     "輕井澤",
    "輕井沢":         "輕井澤",
    # 佐倉愛裡
    "愛裡":           "佐倉",
    "佐倉愛裡":       "佐倉",
    # 篠原波瑠加
    "筱原":           "篠原",
    "波瑠加":         "篠原",
    "波琉加":         "篠原",
    "筱原皐月":       "篠原",
    # 須藤健
    "須藤健":         "須藤",
    "須藤君":         "須藤",
    # 茶柱老師
    "茶柱老師":       "茶柱",
    # 阪柳有棲
    "阪柳有棲":       "阪柳",
    "阪柳理事長":     "阪柳",
    "阪柳同學":       "阪柳",
    # 幸村啟誠
    "啟誠":           "幸村",
    "幸村啟誠":       "幸村",
    "幸村輝彥":       "幸村",
    # 三宅明人
    "明人":           "三宅",
    "三宅明人":       "三宅",
    # 天澤一夏
    "一夏":           "天澤",
    "天澤一夏":       "天澤",
    # 真嶋老師
    "真嶋老師":       "真嶋",
    "真島老師":       "真嶋",
    # 月城
    "月城代理理事長": "月城",
    # 八神拓也
    "八神拓也":       "八神",
    # 山內春樹
    "山內春樹":       "山內",
    "春樹":           "山內",
    # 南云雅
    "南云雅":         "南云",
    # 七瀨翼
    "七瀨翼":         "七瀨",
    "七瀨同學":       "七瀨",
    # 椎名日和
    "椎名日和":       "椎名",
    # 橋本正義
    "橋本正義":       "橋本",
    "橋本君":         "橋本",
    # 山村美紀
    "山村美紀":       "山村",
    # 葛城康平
    "葛城康平":       "葛城",
    # 高圓寺六助
    "高圓寺六助":     "高圓寺",
    "高圓寺君":       "高圓寺",
    # 神崎隆二
    "神崎隆二":       "神崎",
    "神崎君":         "神崎",
    # 松下千秋
    "松下千秋":       "松下",
    # 佐藤麻耶
    "佐藤麻耶":       "佐藤",
    # 長谷部
    "長谷部波琉加":   "長谷部",
    "長谷部波瑠加":   "長谷部",
    "長谷部波琉":     "長谷部",
    # 星之宮老師
    "星之宮老師":     "星之宮",
    # 直江老師
    "直江老師":       "直江",
    # 阪上老師
    "阪上老師":       "阪上",
    # 白石飛鳥
    "白石飛鳥":       "白石",
    # 鬼頭隼
    "鬼頭隼":         "鬼頭",
}

SKIP_NAMES: set[str] = {
    "我", "(我)", "我 (敘述者)", "我 (主角)", "我 (自稱)",
    "敘述者", "(敘述者)", "敘述者 (主角)", "(敘述者/我)",
    "(我/敘述者)", "(我/堀北)", "(我/綾小路)",
    "(敘述者/主角)", "(無)", "無",
    "A班", "B班", "C班", "D班",
    "學生們 (群體)", "(C班學生群體)", "男人", "博士",
    "X (幕後黑手)",
}

PAREN_RE = re.compile(r"[（(][^）)]*[）)]")


def normalize(raw: str) -> str:
    name = PAREN_RE.sub("", raw).strip()
    if name in SKIP_NAMES or not name:
        return ""
    return ALIAS_MAP.get(name, name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    edge_path = Path(args.db_path) / GRAPH_EDGE_FILE
    lines = edge_path.read_text(encoding="utf-8").splitlines()

    kept, renamed, dropped = 0, 0, 0
    out: list[str] = []

    for line in lines:
        if not line.strip():
            continue
        edge = json.loads(line)

        if edge.get("type") != "character":
            out.append(line)
            kept += 1
            continue

        old_char = edge.get("character", "")
        new_char = normalize(old_char)

        if not new_char:
            dropped += 1
            continue

        if new_char != old_char:
            edge["character"] = new_char
            renamed += 1

        out.append(json.dumps(edge, ensure_ascii=False))
        kept += 1

    print(f"Total lines: {len(lines)}")
    print(f"  next/prev kept:      {kept - renamed - (kept - len([l for l in lines if l and json.loads(l).get('type') == 'character']))}")
    print(f"  character renamed:   {renamed}")
    print(f"  character dropped:   {dropped}")
    print(f"  character kept as-is:{kept - renamed - len([l for l in lines if l and json.loads(l).get('type') != 'character'])}")

    if args.dry_run:
        # show sample of renamed
        sample = [(json.loads(l).get("character",""), normalize(json.loads(l).get("character","")))
                  for l in lines if l and json.loads(l).get("type") == "character"
                  and normalize(json.loads(l).get("character","")) != json.loads(l).get("character","")
                  and normalize(json.loads(l).get("character",""))]
        seen: set[str] = set()
        print("\nSample renames:")
        for old, new in sample:
            if old not in seen:
                print(f"  {old!r:30s} → {new!r}")
                seen.add(old)
            if len(seen) >= 20:
                break
        print("\n[dry-run] nothing written.")
        return

    edge_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"\nWritten → {edge_path}")


if __name__ == "__main__":
    main()
