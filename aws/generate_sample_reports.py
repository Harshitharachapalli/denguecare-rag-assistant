"""Generate realistic-looking synthetic dengue pathology reports for the demo.

All data is fictional and randomly generated. No real patient data is used.
Patient names are obviously synthetic placeholders.

Usage:
    python aws/generate_sample_reports.py [count]

Default count = 50. Files are written to ../sample_reports/.
"""
import os
import random
import sys
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(os.path.dirname(HERE), "sample_reports")

LABS = [
    ("MediTrust Diagnostics", "NABL Accredited | ISO 15189:2012", "Hyderabad"),
    ("CarePath Laboratories", "NABL Accredited Laboratory", "Bengaluru"),
    ("Apollo Reference Lab (Demo)", "College of American Pathologists (mock)", "Chennai"),
    ("LifeLine Clinical Labs", "NABL Accredited | ISO 15189:2012", "Pune"),
    ("Metropolis Demo Diagnostics", "Accredited Reference Laboratory", "Mumbai"),
]

FIRST_M = ["Rahul", "Arjun", "Vikram", "Imran", "Karthik", "Sameer", "Rohan", "Aditya"]
FIRST_F = ["Ananya", "Priya", "Sneha", "Fatima", "Divya", "Meera", "Kavya", "Isha"]
LAST = ["Sharma", "Reddy", "Iyer", "Khan", "Patel", "Nair", "Gupta", "Rao", "Menon"]

REFERRERS = [
    "Dr. S. Menon, MD (General Medicine)",
    "Dr. A. Verma, MD",
    "Dr. R. Krishnan, MBBS",
    "Dr. P. Desai, MD (Internal Medicine)",
    "Self / Walk-in",
]

PATHOLOGISTS = [
    "Dr. Neha Kulkarni, MD (Pathology)",
    "Dr. Rajesh Pillai, MD (Clinical Pathology)",
    "Dr. Anjali Bose, MD, DNB (Pathology)",
]


def flag(value, low, high):
    if value < low:
        return "Low"
    if value > high:
        return "High"
    return "Normal"


def make_report(idx: int) -> tuple[str, str]:
    lab_name, lab_accred, lab_city = random.choice(LABS)
    sex = random.choice(["Male", "Female"])
    first = random.choice(FIRST_M if sex == "Male" else FIRST_F)
    name = f"{first} {random.choice(LAST)}"
    age = random.randint(3, 78)

    collected_dt = datetime(2026, 9, 20) - timedelta(
        days=random.randint(1, 40), hours=random.randint(0, 10)
    )
    reported_dt = collected_dt + timedelta(hours=random.randint(5, 20))

    rid = f"DL-{collected_dt.strftime('%y%m')}-{idx:04d}"
    uhid = f"UH{random.randint(100000, 999999)}"
    barcode = f"{random.randint(10**9, 10**10 - 1)}"

    scenario = random.choices(
        ["acute", "past", "negative"], weights=[0.45, 0.2, 0.35]
    )[0]

    if scenario == "acute":
        ns1 = "Positive"
        igm = random.choice(["Positive", "Positive", "Negative"])
        igg = random.choice(["Negative", "Positive"])
        platelets = random.randint(20_000, 130_000)
        wbc = random.randint(1_800, 3_900)
        hb = round(random.uniform(10.5, 14.5), 1)
        interpretation = (
            "Dengue NS1 antigen is reactive, indicating current dengue viral "
            "infection. Associated thrombocytopenia and leukopenia are commonly "
            "seen in the acute febrile phase. Serial platelet monitoring is advised."
        )
    elif scenario == "past":
        ns1 = "Negative"
        igm = "Negative"
        igg = "Positive"
        platelets = random.randint(150_000, 380_000)
        wbc = random.randint(4_200, 10_500)
        hb = round(random.uniform(12.0, 16.0), 1)
        interpretation = (
            "Dengue IgG is reactive with non-reactive NS1 and IgM. This pattern "
            "is suggestive of past dengue exposure / secondary immunity rather "
            "than an acute current infection. Correlate clinically."
        )
    else:
        ns1 = "Negative"
        igm = "Negative"
        igg = "Negative"
        platelets = random.randint(155_000, 400_000)
        wbc = random.randint(4_500, 10_800)
        hb = round(random.uniform(12.5, 16.5), 1)
        interpretation = (
            "Dengue NS1 antigen, IgM and IgG antibodies are non-reactive with a "
            "normal haemogram. No laboratory evidence of dengue infection in this "
            "sample. Repeat testing may be considered if clinically indicated."
        )

    hct = random.randint(34, 50)
    rbc = round(random.uniform(3.8, 5.9), 2)
    fever_days = random.randint(1, 7)

    def cell(v, unit, ref, low, high):
        return v, unit, ref, flag(v if isinstance(v, (int, float)) else 0, low, high)

    plt_flag = flag(platelets, 150_000, 410_000)
    wbc_flag = flag(wbc, 4_000, 11_000)
    hb_flag = flag(hb, 12.0, 16.0)
    hct_flag = flag(hct, 36, 50)
    rbc_flag = flag(rbc, 4.2, 5.9)

    line = "=" * 70
    thin = "-" * 70

    report = f"""{line}
{lab_name.upper():^70}
{lab_accred:^70}
{(lab_city + " | www.demo-lab.example | +91-XXXXXXXXXX"):^70}
{line}
                        DENGUE DIAGNOSTIC REPORT
                (Synthetic sample - not real patient data)
{thin}
Patient Name   : {name:<28} Report No.   : {rid}
Age / Sex      : {str(age) + " Yrs / " + sex:<28} UHID         : {uhid}
Referred By    : {random.choice(REFERRERS)}
Sample Barcode : {barcode:<28} Specimen     : Serum / EDTA Whole Blood
Collected On   : {collected_dt.strftime('%d-%b-%Y %H:%M'):<28} Reported On  : {reported_dt.strftime('%d-%b-%Y %H:%M')}
{thin}
CLINICAL NOTES
  Fever x {fever_days} day(s). Sent for dengue workup and complete blood count.

{thin}
DENGUE SEROLOGY / ANTIGEN
{thin}
  Test                              Result        Method
  Dengue NS1 Antigen                {ns1:<12}  ELISA / Rapid Immunoassay
  Dengue IgM Antibody               {igm:<12}  ELISA
  Dengue IgG Antibody               {igg:<12}  ELISA
  (Reference: Non-reactive / Negative)

{thin}
COMPLETE BLOOD COUNT (CBC)
{thin}
  Investigation           Result          Unit        Bio. Ref. Range   Flag
  Hemoglobin              {hb:<15} g/dL        12.0 - 16.0       {hb_flag}
  Total WBC Count         {wbc:<15,} /uL         4,000 - 11,000    {wbc_flag}
  Platelet Count          {platelets:<15,} /uL         150,000 - 410,000 {plt_flag}
  RBC Count               {rbc:<15} mill/uL     4.2 - 5.9         {rbc_flag}
  Hematocrit (PCV)        {str(hct) + '%':<15} %           36 - 50           {hct_flag}
  Method: Automated 5-part Haematology Analyzer

{thin}
INTERPRETATION
  {interpretation}

{thin}
COMMENTS
  - Results should be correlated with clinical findings and other
    investigations. Serological cross-reactivity may occur with other
    flaviviruses.
  - This is a SYNTHETIC report generated for a software demonstration and
    must not be used for medical decisions.

{thin}
  Verified & Authorized by
  {random.choice(PATHOLOGISTS)}

  *** End of Report ***
{line}
"""
    filename = f"synthetic_dengue_report_{idx:02d}.txt"
    return filename, report


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(42)  # reproducible set
    for i in range(1, count + 1):
        fname, content = make_report(i)
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as fh:
            fh.write(content)
    print(f"Wrote {count} realistic synthetic reports to {OUT_DIR}")


if __name__ == "__main__":
    main()
