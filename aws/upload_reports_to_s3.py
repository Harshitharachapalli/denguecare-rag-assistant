"""Upload the synthetic sample reports to your S3 bucket.

Usage:
    python aws/upload_reports_to_s3.py <bucket-name> [prefix]

Example:
    python aws/upload_reports_to_s3.py denguecare-reports-demo reports/

Requires AWS credentials via `aws configure`.
"""
import os
import sys

import boto3

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(os.path.dirname(HERE), "sample_reports")


def main():
    if len(sys.argv) < 2:
        print("Usage: python aws/upload_reports_to_s3.py <bucket> [prefix]")
        sys.exit(1)

    bucket = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "reports/"
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    s3 = boto3.client("s3")
    files = [f for f in os.listdir(SAMPLES) if f.lower().endswith((".txt", ".pdf"))]
    if not files:
        print("No sample reports found in", SAMPLES)
        sys.exit(1)

    for name in files:
        key = f"{prefix}{name}"
        s3.upload_file(os.path.join(SAMPLES, name), bucket, key)
        print(f"Uploaded s3://{bucket}/{key}")

    print(f"\nDone. Now sync your Bedrock Knowledge Base data source to ingest these.")


if __name__ == "__main__":
    main()
