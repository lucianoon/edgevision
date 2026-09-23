"""MOTChallenge conversion used by the Sprint 9 tracking evaluation (no GPU, no TrackEval)."""

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mot17_eval.py"


def load_module():
    spec = importlib.util.spec_from_file_location("mot17_eval", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_to_mot_keeps_persons_and_writes_one_based_tlwh(tmp_path):
    dump = tmp_path / "tracks.jsonl"
    frames = [
        {
            "frame": 1,
            "tracks": [
                {
                    "id": 3,
                    "x1": 10.0,
                    "y1": 20.0,
                    "x2": 50.0,
                    "y2": 120.0,
                    "score": 0.9,
                    "class_name": "person",
                },
                {
                    "id": 4,
                    "x1": 1.0,
                    "y1": 1.0,
                    "x2": 5.0,
                    "y2": 5.0,
                    "score": 0.8,
                    "class_name": "car",
                },
            ],
        },
        {"frame": 2, "tracks": []},
    ]
    dump.write_text("\n".join(json.dumps(f) for f in frames) + "\n\n", encoding="utf-8")
    out = tmp_path / "out" / "MOT17-02-FRCNN.txt"

    rows = load_module().to_mot(dump, out)

    assert rows == 1
    assert out.read_text(encoding="utf-8") == "1,3,11.00,21.00,40.00,100.00,0.9000,-1,-1,-1\n"
