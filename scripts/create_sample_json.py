"""Create minimal sample JSON files for golden set generation."""

import json
from pathlib import Path

def create_sample_json():
    """Create sample JSON files for golden set generation."""
    output_dir = Path("data/sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sample trials
    trials = [
        {
            "nct_id": f"NCT{10000000 + i:05d}",
            "title": f"Study of {drug} in {condition}",
            "brief_summary": f"This study evaluates {drug} for the treatment of {condition}.",
            "phase": phase,
            "overall_status": status,
        }
        for i, (drug, condition, phase, status) in enumerate([
            ("pembrolizumab", "Non-Small Cell Lung Cancer", "Phase 3", "Completed"),
            ("nivolumab", "Melanoma", "Phase 3", "Completed"),
            ("adalimumab", "Rheumatoid Arthritis", "Phase 3", "Completed"),
            ("ustekinumab", "Psoriasis", "Phase 3", "Completed"),
            ("sacubitril/valsartan", "Heart Failure", "Phase 3", "Completed"),
            ("denosumab", "Osteoporosis", "Phase 3", "Completed"),
        ])
    ]
    (output_dir / "trials.json").write_text(json.dumps(trials, indent=2))

    # Sample PubMed
    pubmed = [
        {
            "pmid": f"{10000000 + i}",
            "title": f"Efficacy of {drug} in {condition}",
            "abstract": f"Background: {condition} is a serious condition. Methods: We conducted a randomized trial of {drug}. Results: The treatment showed significant benefit. Conclusion: {drug} is effective.",
            "journal": journal,
            "year": 2020 + (i % 4),
        }
        for i, (drug, condition, journal) in enumerate([
            ("pembrolizumab", "lung cancer", "NEJM"),
            ("nivolumab", "melanoma", "Lancet"),
            ("adalimumab", "rheumatoid arthritis", "JAMA"),
            ("ustekinumab", "psoriasis", "Nature Medicine"),
            ("sacubitril/valsartan", "heart failure", "JCO"),
            ("denosumab", "osteoporosis", "Blood"),
        ])
    ]
    (output_dir / "pubmed.json").write_text(json.dumps(pubmed, indent=2))

    # Sample labels
    labels = [
        {
            "set_id": f"sample-{drug.lower()}-{i:03d}",
            "brand_names": [brand],
            "generic_names": [drug],
            "sections": {
                "indications_and_usage": f"INDICATIONS AND USAGE\n{brand} ({drug}) is indicated for the treatment of...",
                "dosage_and_administration": f"DOSAGE AND ADMINISTRATION\nThe recommended dose of {brand} is...",
                "contraindications": f"CONTRAINDICATIONS\n{brand} is contraindicated in patients with...",
                "warnings_and_cautions": f"WARNINGS AND PRECAUTIONS\nSerious adverse reactions include...",
                "adverse_reactions": f"ADVERSE REACTIONS\nThe most common adverse reactions are...",
                "drug_interactions": f"DRUG INTERACTIONS\nNo formal drug interaction studies have been conducted...",
                "clinical_studies": f"CLINICAL STUDIES\nThe efficacy of {brand} was demonstrated in...",
            }
        }
        for i, (drug, brand) in enumerate([
            ("pembrolizumab", "Keytruda"),
            ("nivolumab", "Opdivo"),
            ("adalimumab", "Humira"),
            ("ustekinumab", "Stelara"),
            ("sacubitril/valsartan", "Entresto"),
            ("denosumab", "Prolia"),
        ])
    ]
    (output_dir / "labels.json").write_text(json.dumps(labels, indent=2))

    print(f"Created sample data in {output_dir}")

if __name__ == "__main__":
    create_sample_json()