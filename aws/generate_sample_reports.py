"""Generate synthetic dengue test reports for the RAG demo.

All data is randomly generated and fictional. No real patient data is used.
Aliases like "Patient 007" are used instead of real names.

Usage:
    python aws/generate_sample_reports.py [count]

Default count = 50. Files are written to ../sample_reports/.
"""
import os
import random
import sys
from datetime import date, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "sample_reports")

LABS = [
    "Synthetic Diagnostics (Demo)",
    "DemoCare Pathology Labs",
    "SampleHealth Reference Lab",
    "MockMed Clinical Laboratory",
    "TestData Diagnostics Center",
]

CITIES = ["Hyderabad", "Bengaluru", "Chennai", "Pune", "Mumbai", "Delhi", "Kochi"]


def _ref(low, high):
    return f"(Ref {low:,} - {high:,})"


def _flag(value, low, high):
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "NORMAL"


def make_report(idx: int) -> tuple[str, str]:
    rid = f"SYN-DENG-{idx:04d}"
    age = random.randint(2, 78)
    sex = random.choice(["Male", "Female"])
    days_ago = random.randint(1, 40)
    collected = (date(2026, 9, 20) - timedelta(days=days_ago)).isoformat()
    lab = random.choice(LABS)
    city = random.choice(CITIES)

    # Decide a scenario: acute, recovering/past, or negative.
    scenario = random.choices(
        ["acute", "past", "negative"], weights=[0.45, 0.2, 0.35]
    )[0]

    if scenario == "acute":
        ns1 = "POSITIVE"
        igm = random.choice(["POSITIVE", "POSITIVE", "NEGATIVE"])
        igg = random.choice(["NEGATIVE", "POSITIVE"])
        platelets = random.randint(20_000, 130_000)
        wbc = random.randint(1_800, 3_900)
        interpretation = (
            "NS1 positive with thrombocytopenia (low platelets) is consistent "
            "with an acute dengue infection. Close monitoring is advised."
        )
    elif scenario == "past":
        ns1 = "NEGATIVE"
        igm = "NEGATIVE"
        igg = "POSITIVE"
        platelets = random.randint(150_000, 380_000)
        wbc = random.randint(4_200, 10_500)
        interpretation = (
            "IgG positive with negative NS1/IgM and normal counts suggests past "
            "dengue exposure rather than active infection."
        )
    else:  # negative
        ns1 = "NEGATIVE"
        igm = "NEGATIVE"
        igg = "NEGATIVE"
        platelets = random.randint(155_000, 400_000)
        wbc = random.randint(4_500, 10_800)
        interpretation = (
            "NS1, IgM and IgG negative with normal platelet and WBC counts. "
            "No serological evidence of dengue in this sample."
        )

    hb = round(random.uniform(10.5, 16.5), 1)
    hct = random.randint(34, 50)

    plt_flag = _flag(platelets, 150_000, 410_000)
    wbc_flag = _flag(wbc, 4_000, 11_000)

    fever_days = random.randint(1, 7)
    symptoms = random.sample(
        ["headache", "retro-orbital pain", "body ache", "nausea",
         "rash", "joint pain", "mild bleeding gums", "fatigue"],
        k=random.randint(2, 4),
    )

    report = f"""SYNTHETIC DENGUE TEST REPORT (SAMPLE - NOT REAL PATIENT DATA)
==============================================================
Report ID       : {rid}
Patient (alias) : Patient {idx:03d}
Age / Sex       : {age} / {sex}
Location        : {city} (synthetic)
Collection Date : {collected}
Reporting Lab   : {lab}

CLINICAL HISTORY
----------------
Fever for {fever_days} day(s). Reported symptoms: {", ".join(symptoms)}.

SEROLOGY / ANTIGEN
------------------
Dengue NS1 Antigen      : {ns1}
Dengue IgM Antibody     : {igm}
Dengue IgG Antibody     : {igg}

COMPLETE BLOOD COUNT
--------------------
Hemoglobin              : {hb} g/dL     (Ref 12.0 - 16.0)
Total WBC Count         : {wbc:,} /uL    {_ref(4000, 11000)}  {wbc_flag}
Platelet Count          : {platelets:,} /uL   {_ref(150000, 410000)}  {plt_flag}
Hematocrit (PCV)        : {hct}%          (Ref 36 - 50)

INTERPRETATION
--------------
{interpretation}

RECOMMENDATION (INFORMATIONAL ONLY)
-----------------------------------
Clinical correlation advised. This is synthetic demonstration data and not
medical advice.
"""
    filename = f"synthetic_dengue_report_{idx:02d}.txt"
    return filename, report


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)  # reproducible set
    for i in range(1, count + 1):
        name, content = make_report(i)
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as fh:
            fh.write(content)
    print(f"Wrote {count} synthetic reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
