"""Download the public corpora used in the paper into data_raw/ (all from the Hugging Face Hub)."""
import urllib.request
from slide.config import RAW

RAW.mkdir(parents=True, exist_ok=True)
HF = "https://huggingface.co/datasets"
FILES = {f"seven_{s}.parquet": f"{HF}/puyang2025/seven-phishing-email-datasets/resolve/main/{s}.parquet"
         for s in ("train", "eval", "test")}
FILES["gpt_phish.csv"] = f"{HF}/Dizzzy0x00/LLMGen-Phishing-Email-Dataset/resolve/main/GPT_Phishing_Email_dataset.csv"
FILES["kx_openai.jsonl"] = f"{HF}/kxm1k4m1/generate_phishing_email_final/resolve/main/generate_phishing_email_openai_2800_row.jsonl"
for name, url in FILES.items():
    if not (RAW / name).exists():
        print("downloading", name)
        urllib.request.urlretrieve(url, RAW / name)
print("ok")
