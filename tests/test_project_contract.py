from pathlib import Path


def test_training_fields_exclude_instance_labels():
    text = (Path(__file__).parents[1] / "configs/dataset/camelyon16.yaml").read_text(encoding="utf-8")
    assert "training_fields: [X, Y]" in text
    assert "y_inst" not in text.split("training_fields:", 1)[1].splitlines()[0]


def test_no_full_snapshot_download():
    text = (Path(__file__).parents[1] / "scripts/download_data.py").read_text(encoding="utf-8")
    assert "snapshot_download" not in text
    assert "features_UNI.tar.gz" in text

