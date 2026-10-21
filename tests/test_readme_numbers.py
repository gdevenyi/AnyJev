"""The result tables in both READMEs are transcribed from committed JSON; this keeps them in step."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = ["README.md", "README.zh-CN.md"]
TACIT = ROOT / "bench/results_tacit/2026-10-06"
L0 = ROOT / "bench/results_v01/2026-09-22/Qwen__Qwen3-8B.json"
DI = ROOT / "bench/results_decision_index/2026-10-12/anyjev-l0-gemma-4-31b.json"
TACITUS = ROOT / "bench/results_decision_index/2026-10-21/anyjev-tacitus-31b.json"


def cells(line):
    return [c.strip().replace("**", "") for c in line.strip().strip("|").split("|")]


def expected(res):
    n = res["n"]
    one = "%.3f" % (res["one_forward"]["correct"] / n)
    if not res["settings"]["adaptive"]:
        return one, "—"
    a = res["adaptive"]
    return one, "%.3f <sub>%.1f%%</sub>" % (a["correct"] / n, 100 * a["escalated"] / n)


@pytest.mark.parametrize("readme", READMES)
def test_tacit_table_matches_its_json(readme):
    rows = [cells(x) for x in (ROOT / readme).read_text(encoding="utf-8").splitlines() if x.startswith("| [Tacit-")]
    assert len(rows) == 5
    for row in rows:
        name = re.match(r"\[(Tacit-[\w.]+)\]", row[0]).group(1)
        for col, task in ((2, "jevbench"), (4, "bev")):
            res = json.loads((TACIT / ("%s.%s.json" % (name, task))).read_text(encoding="utf-8"))
            assert res["model"] == "morriszjm/" + name
            assert (row[col], row[col + 1]) == expected(res), (readme, name, task)


@pytest.mark.parametrize("readme", READMES)
def test_raw_vs_l0_table_matches_its_json(readme):
    task = next(t for t in json.loads(L0.read_text(encoding="utf-8"))["tasks"] if t["task"] == "banking20")["levels"]
    text = (ROOT / readme).read_text(encoding="utf-8").replace("**", "")
    for metric in ("flip", "acc", "ece"):
        want = "| %.3f | %.3f |" % (task["raw"][metric], task["L0"][metric])
        assert want in text, (readme, metric, want)
    cov = "| %.1f%% | %.1f%% |" % (100 * task["raw"]["cov@5%"], 100 * task["L0"]["cov@5%"])
    assert cov in text, (readme, cov)


@pytest.mark.parametrize("readme", READMES)
def test_decision_index_numbers_match_their_json(readme):
    b = json.loads(DI.read_text(encoding="utf-8"))["board"]
    text = (ROOT / readme).read_text(encoding="utf-8")
    lat = " / ".join("{:,}".format(round(b["latency_ms"][k])) for k in ("median", "mean", "p95"))
    row = "| **%.2f** | %.2f | %.2f | %.2f | %s ms |" % (
        b["full_score"], b["public"], b["same_skills_private"], b["new_domains_private"], lat)
    assert row in text, (readme, row)
    assert "Decision%%20Index%%200.3-%.2f" % b["full_score"] in text, readme       # the badge
    assert text.count("**%.2f**" % b["full_score"]) == 2, readme                    # the callout and the table
    rank = {"README.md": "5th of %d", "README.zh-CN.md": "%d 个条目中排第 5"}[readme] % b["entries"]
    assert b["rank_by_full_score"] == 5 and text.count(rank) == 2, (readme, rank)


@pytest.mark.parametrize("readme", READMES)
def test_tacitus_numbers_match_their_json(readme):
    b = json.loads(TACITUS.read_text(encoding="utf-8"))["board"]
    text = (ROOT / readme).read_text(encoding="utf-8")
    lat = " / ".join("{:,}".format(round(b["latency_ms"][k])) for k in ("median", "mean", "p95"))
    row = "| **%.2f** | %.2f | %.2f | %.2f | %s ms |" % (
        b["full_score"], b["same_skills_private"], b["new_domains_private"], b["public"], lat)
    assert row in text, (readme, row)
    assert "Decision%%20Index%%200.4.1-Tacitus%%20%.2f" % b["full_score"] in text, readme   # the badge
    assert text.count("**%.2f**" % b["full_score"]) == 2, readme                          # the callout and the table
    rank = {"README.md": "tied %drd", "README.zh-CN.md": "并列第 %d"}[readme] % b["rank_shown_with_ties"]
    assert b["rank_shown_with_ties"] == 3 and text.count(rank) == 3, (readme, rank)      # callout, section, roadmap
