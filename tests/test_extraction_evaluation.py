from evaluation.extraction_evaluator import evaluate_extraction_quality


def test_extraction_quality_meets_precision_recall_thresholds():
    report = evaluate_extraction_quality()

    assert report["sample_count"] >= 4
    assert report["entity_metrics"]["f1"] >= 0.85
    assert report["relationship_metrics"]["f1"] >= 0.85