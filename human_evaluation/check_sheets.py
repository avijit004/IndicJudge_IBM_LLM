"""Audit the blank distribution package against the original candidate CSVs."""
from collections import Counter
import hashlib
from html.parser import HTMLParser

from openpyxl import load_workbook

import prepare_sheets as p


class ReadingPack(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items = {}
        self.current = None
        self.in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag == "article":
            self.current = dict(attrs)["id"]
            assert self.current not in self.items
            self.items[self.current] = []
        elif tag == "pre":
            self.in_pre = True
            self.items[self.current].append("")
        assert tag not in ("script", "iframe", "form")

    def handle_data(self, text):
        if self.in_pre:
            self.items[self.current][-1] += text

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_pre = False


def main():
    rows = p.sample()
    expected = {r["evaluation_id"]: r for r in rows}
    assert len(expected) == 200
    assert len({(r['language'], r['source_model'], r['question_id'], r['prompt_id']) for r in rows}) == 200
    assert set(Counter(r["language"] for r in rows).values()) == {20}
    p.verify(rows)
    reviewer_data = []
    for name in ("reviewer_1.xlsx", "reviewer_2.xlsx"):
        wb = load_workbook(p.OUT / name)
        assert len(wb.worksheets) == 11
        instructions = dict(wb["Instructions"].iter_rows(min_row=2, values_only=True))
        assert instructions["Review independently"] == p.RUBRIC[0][1]
        assert instructions["status"] == dict(p.RUBRIC)["status"]
        all_rows = []
        for ws in wb.worksheets:
            assert ws.sheet_state == "visible"
            assert not ws.protection.sheet
            if ws.title == "Instructions":
                continue
            assert list(next(ws.values)) == p.HEADERS
            assert ws.max_row == 21 and ws.max_column == 14
            assert ws.freeze_panes == "C2"
            assert ws.auto_filter.ref == "A1:N21"
            specs = {"I2:I21": ("list", '"True,False"', None),
                     "J2:J21": ("whole", "0", "5"),
                     "K2:K21": ("list", '"True,False"', None),
                     "M2:M21": ("list", '"Complete,Needs review,Cannot assess"', None)}
            for dv in ws.data_validations.dataValidation:
                assert (dv.type, dv.formula1, dv.formula2) == specs[str(dv.sqref)]
                assert dv.showErrorMessage and dv.errorStyle == "stop"
                assert not dv.showDropDown
                if dv.type == "whole":
                    assert dv.operator == "between"
            for cell_row in ws.iter_rows(min_row=2):
                r = expected[cell_row[0].value]
                assert ws.title.lower() == r["language"]
                for cell in cell_row:
                    assert cell.comment is None and cell.hyperlink is None
                for cell in cell_row[7:]:
                    assert cell.fill.fgColor.rgb[-6:] == "FFF2CC"
            all_rows.extend(ws.iter_rows(min_row=2, values_only=True))
        reviewer_data.append(all_rows)
    assert reviewer_data[0] == reviewer_data[1]

    parser = ReadingPack()
    parser.feed((p.OUT / "reading_pack.html").read_text(encoding="utf-8"))
    assert set(parser.items) == set(expected)
    for key, texts in parser.items.items():
        assert texts == [expected[key][k] for k in ("question", "ground_truth", "prompt", "reasoning_content", "response")]

    wb = load_workbook(p.OUT / "coordinator.xlsx")
    keys = list(wb["Sample key"].values)
    seen = set()
    for values in keys[1:]:
        record = dict(zip(keys[0], values))
        key = record["evaluation_id"]
        assert key not in seen
        seen.add(key)
        for column in keys[0][:-1]:
            assert record[column] == expected[key][column]
        assert record["input_sha256"] == hashlib.sha256((p.ROOT / record["source_file"]).read_bytes()).hexdigest()
    assert seen == set(expected)
    consensus = list(wb["Consensus"].values)
    assert len(consensus) == 201
    assert {r[0] for r in consensus[1:]} == set(expected)
    for row in consensus[1:]:
        assert row[1] == expected[row[0]]["language"]
        assert all(value is None for value in row[2:])
    print("PASS: sample balance, unique IDs, source text, references, reviewer parity, blank ratings, integer-only score validation, instructions, reading pack, coordinator joins and source hashes.")


if __name__ == "__main__":
    main()
