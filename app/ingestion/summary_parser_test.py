from datasets import load_dataset
from summary_parser import parse_summary

def test_parse_summary(n_rows = 5):
    ds = load_dataset("AGBonnet/augmented-clinical-notes", split="train")
    rows = ds.select(range(n_rows))

    for i, row in enumerate(rows):
        out = parse_summary(row.get("summary"))
        print(f"Row {i} summary parsing result:")
        print("input summary type: ", type(row.get("summary")).__name__)
        print("Output dict:", out)

        # basic shape checks
        assert isinstance(out, dict)
        for key in [
            "patient_age_raw",
            "patient_age",
            "patient_age_group",
            "patient_sex",
            "visit_motivation",
            "primary_diagnosis",
        ]:
            assert key in out

if __name__ == "__main__":
    test_parse_summary(5)
    print("All tests passed!")