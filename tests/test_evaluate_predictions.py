"""Synthetic CSV contracts with a model double; no real inference or benchmark data."""

import csv
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from models import evaluate
from models.utils import drop_nan_rows


@pytest.fixture
def synthetic(monkeypatch):
    targets = evaluate._CYP_TARGETS[:2]
    train = pl.DataFrame({"SMILES": ["train"], **{target: [1.0] for target in targets}})
    val = pl.DataFrame({
        "SMILES": ["C", "bad-feature", "missing-label", "N", "O"],
        targets[0]: [10.0, 20.0, None, 40.0, 50.0],
        targets[1]: [11.0, 21.0, 31.0, 41.0, None],
    })
    instances = []
    scores = []
    split_calls = []

    class Model:
        name = "double"

        def __init__(self):
            self.calls = 0
            instances.append(self)

        def fit(self, X, y):
            np.testing.assert_array_equal(y, [1.0])

        def predict(self, X):
            self.calls += 1
            # A second call changes values, making accidental re-prediction observable.
            self.predictions = X[:, 0] + self.calls * 0.125
            return self.predictions

    class OtherModel(Model):
        name = "other_double"

    def load(**kwargs):
        split_calls.append(kwargs)
        return train, val

    def features(df, names, cache):
        if len(df) == 1:
            return np.array([[1.0]])
        # b retains one more molecule: exports must not intersect representations.
        return np.array([[1.0], [np.inf if "a" in names else 2.0], [3.0], [4.0], [5.0]])

    original_metrics = evaluate._compute_metrics

    def score(actual, predicted):
        scores.append((actual.to_list(), predicted.to_list()))
        return original_metrics(actual, predicted)

    monkeypatch.setattr(evaluate, "REGISTRY", {Model.name: Model, OtherModel.name: OtherModel})
    monkeypatch.setattr(evaluate, "INPUT_REGISTRY", {"a": None, "b": None})
    monkeypatch.setattr(evaluate, "load_data", load)
    monkeypatch.setattr(evaluate, "featurize", features)
    monkeypatch.setattr(evaluate, "_compute_metrics", score)
    return targets, instances, scores, split_calls, val


def run(monkeypatch, targets, path=None, inputs=("a", "b"), models=("all",)):
    argv = ["evaluate-models", "--models", *models, "--targets", *targets,
            "--input", *inputs, "--split", "butina", "--seed", "17", "--val-split", "0.3"]
    if path is not None:
        argv += ["--predictions-out", str(path)]
    monkeypatch.setattr(sys, "argv", argv)
    evaluate.main()


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["smiles", "isoform", "model", "input", "split", "seed", "y_true", "y_pred"]
        return list(reader)


def test_export_alignment_metadata_and_single_prediction(monkeypatch, tmp_path, synthetic):
    targets, instances, scores, split_calls, _ = synthetic
    path = tmp_path / "results" / "nested" / "validation.csv"
    run(monkeypatch, targets, path)
    rows = read_rows(path)
    assert len(rows) == 12
    assert {row["model"] for row in rows} == {"double", "other_double"}
    assert {row["isoform"] for row in rows} == set(targets)
    assert {row["input"] for row in rows} == {"a + b"}
    assert {row["split"] for row in rows} == {"butina"}
    assert {row["seed"] for row in rows} == {"17"}
    assert split_calls[0]["seed"] == 17
    assert split_calls[0]["val_fraction"] == 0.3
    for i, (model, target) in enumerate((m, t) for t in targets for m in ("double", "other_double")):
        group = [row for row in rows if row["model"] == model and row["isoform"] == target]
        expected_smiles = ["C", "N", "O"] if target == targets[0] else ["C", "missing-label", "N"]
        assert [row["smiles"] for row in group] == expected_smiles
        actual = [float(row["y_true"]) for row in group]
        predictions = [float(row["y_pred"]) for row in group]
        assert actual == ([10.0, 40.0, 50.0] if target == targets[0] else [11.0, 31.0, 41.0])
        assert (actual, predictions) == scores[i]
        np.testing.assert_array_equal(predictions, instances[i].predictions)
        assert instances[i].calls == 1


def test_all_inputs_keep_each_representations_survivors(monkeypatch, tmp_path, synthetic):
    targets, instances, _, _, _ = synthetic
    path = tmp_path / "results" / "all.csv"
    run(monkeypatch, targets, path, inputs=("all",))
    rows = read_rows(path)
    assert len(rows) == 28
    assert {row["input"] for row in rows} == {"a", "b"}
    assert any(row["smiles"] == "bad-feature" and row["input"] == "b" for row in rows)
    assert all(row["smiles"] != "bad-feature" for row in rows if row["input"] == "a")
    assert len(instances) == 8
    assert all(instance.calls == 1 for instance in instances)


def test_without_export_preserves_metrics_and_output(monkeypatch, tmp_path, synthetic, capsys):
    targets, instances, scores, _, _ = synthetic
    run(monkeypatch, targets)
    without = capsys.readouterr()
    original_scores = list(scores)
    assert all(instance.calls == 1 for instance in instances)
    assert len(instances) == 4
    assert not list(tmp_path.iterdir())
    run(monkeypatch, targets, tmp_path / "results" / "validation.csv")
    with_export = capsys.readouterr()
    assert with_export.out == without.out
    assert scores[4:] == original_scores
    assert "Wrote" not in without.err


def test_filter_default_and_mask_share_the_same_rule():
    X = np.array([[1.0], [np.nan], [2.0], [3.0], [np.inf]])
    y = np.array([4.0, 5.0, np.nan, 6.0, 7.0])
    original = drop_nan_rows(X, y)
    cleaned_X, cleaned_y, mask = drop_nan_rows(X, y, return_mask=True)
    assert len(original) == 2
    np.testing.assert_array_equal(mask, [True, False, False, True, False])
    np.testing.assert_array_equal(cleaned_X, original[0])
    np.testing.assert_array_equal(cleaned_y, original[1])
    assert drop_nan_rows(X)[1] is None


@pytest.mark.parametrize("destination", ["file", "directory", "blocked_parent"])
def test_destination_errors_are_informative(monkeypatch, tmp_path, synthetic, capsys, destination):
    targets, *_ = synthetic
    path = tmp_path / "results" / "validation.csv"
    path.parent.mkdir()
    if destination == "file":
        path.write_text("keep me", encoding="utf-8")
    elif destination == "directory":
        path.mkdir()
    else:
        path.write_text("parent is a file", encoding="utf-8")
        path = path / "child.csv"
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, targets, path)
    assert exc.value.code == 1
    error = capsys.readouterr().err
    assert "Cannot write predictions CSV" in error
    assert str(path) in error
    if destination == "file":
        assert path.read_text(encoding="utf-8") == "keep me"


def test_permission_error_reports_underlying_reason(monkeypatch, tmp_path, synthetic, capsys):
    targets, *_ = synthetic
    path = tmp_path / "results" / "validation.csv"

    def denied(*args, **kwargs):
        raise PermissionError("synthetic access denied")

    monkeypatch.setattr(Path, "open", denied)
    with pytest.raises(SystemExit):
        run(monkeypatch, targets, path)
    error = capsys.readouterr().err
    assert str(path) in error
    assert "synthetic access denied" in error


def test_duplicate_surviving_smiles_rejects_export_only(monkeypatch, tmp_path, synthetic, capsys):
    targets, _, _, _, val = synthetic
    val.replace_column(0, pl.Series("SMILES", ["C", "bad-feature", "missing-label", "C", "O"]))
    path = tmp_path / "results" / "validation.csv"
    with pytest.raises(SystemExit) as exc:
        run(monkeypatch, targets, path)
    assert exc.value.code == 1
    assert not path.exists()
    error = capsys.readouterr().err
    for detail in ["duplicate SMILES 'C'", targets[0], "model=double", "input=a + b", "split=butina", "seed=17"]:
        assert detail in error
    run(monkeypatch, targets)  # Existing evaluation still accepts duplicate source rows.


def test_duplicate_in_filtered_row_is_not_ambiguous(monkeypatch, tmp_path, synthetic):
    targets, _, _, _, val = synthetic
    val.replace_column(0, pl.Series("SMILES", ["C", "C", "missing-label", "N", "O"]))
    path = tmp_path / "results" / "validation.csv"
    run(monkeypatch, targets, path)
    assert len(read_rows(path)) == 12


def test_repeated_group_selection_rejects_ambiguous_keys(monkeypatch, tmp_path, synthetic):
    targets, *_ = synthetic
    path = tmp_path / "results" / "validation.csv"
    with pytest.raises(SystemExit):
        run(monkeypatch, targets, path, models=("double", "double"))
    assert not path.exists()
