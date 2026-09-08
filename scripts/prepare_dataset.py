import os
import pandas as pd
import requests


DOCS_CSV = "RAG-Evaluation-Dataset-KO/documents.csv"
QA_CSV = "data/processed/commerce_qa.csv"

PDF_DIR = "data/raw/commerce_pdfs"
FILTERED_QA_PATH = "data/processed/commerce_qa_available.csv"
FAILED_PATH = "data/processed/failed_documents.csv"

os.makedirs(PDF_DIR, exist_ok=True)


def main():
    docs_df = pd.read_csv(DOCS_CSV)
    qa_df = pd.read_csv(QA_CSV)

    target_files = qa_df["target_file_name"].dropna().unique()

    needed_docs = docs_df[
        docs_df["file_name"].isin(target_files)
    ].copy()

    success_files = []
    failed_files = []

    for _, row in needed_docs.iterrows():
        file_name = row["file_name"]
        url = row["url"]
        save_path = os.path.join(PDF_DIR, file_name)

        try:
            response = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0"}
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if "pdf" not in content_type:
                if not response.content.startswith(b"%PDF"):
                    raise ValueError(
                        f"PDF가 아닌 응답입니다. "
                        f"Content-Type={content_type}"
                    )

            with open(save_path, "wb") as f:
                f.write(response.content)

            success_files.append(file_name)
            print(f"SUCCESS: {file_name}")

        except Exception as e:
            failed_files.append({
                "file_name": file_name,
                "url": url,
                "error": str(e)
            })

            print(f"FAILED: {file_name} - {e}")

    filtered_qa = qa_df[
        qa_df["target_file_name"].isin(success_files)
    ].copy()

    filtered_qa.to_csv(
        FILTERED_QA_PATH,
        index=False
    )

    pd.DataFrame(failed_files).to_csv(
        FAILED_PATH,
        index=False
    )

    print(f"\n필요 문서: {len(needed_docs)}")
    print(f"다운로드 성공: {len(success_files)}")
    print(f"다운로드 실패: {len(failed_files)}")
    print(f"최종 사용 QA: {len(filtered_qa)}")


if __name__ == "__main__":
    main()