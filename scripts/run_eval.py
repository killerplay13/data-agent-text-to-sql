from pathlib import Path
import json

from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService
from app.services.execution_service import ExecutionService


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATASET_PATH = BASE_DIR / "tests" / "test_dataset.json"


def load_test_dataset():
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().lower().split())


def main():
    retrieval_service = RetrievalService()
    sql_service = SQLGenerationService()
    execution_service = ExecutionService()

    test_cases = load_test_dataset()

    total = len(test_cases)
    exact_match_count = 0
    result_match_count = 0

    print("=== Evaluation Start ===\n")

    for idx, case in enumerate(test_cases, start=1):
        question = case["question"]
        ground_truth_sql = case["ground_truth_sql"]

        print(f"Case {idx}")
        print(f"Question: {question}")

        retrieval_result = retrieval_service.retrieve(question)
        sql_templates = retrieval_result.get("sql_templates", [])
        first_template = "None"
        if sql_templates:
            first_template = (
                sql_templates[0].get("name")
                or sql_templates[0].get("question_example")
                or "Unnamed template"
            )

        print(f"Retrieved SQL Template: {first_template}")
        print(f"Retrieved Schema Docs: {len(retrieval_result.get('schema_docs', []))}")
        print(f"Retrieved Business Context: {len(retrieval_result.get('business_context', []))}")

        generated_sql = sql_service.generate_sql(question, retrieval_result)

        generated_sql_error = None
        try:
            generated_result = execution_service.execute_query(generated_sql)
        except Exception as e:
            generated_sql_error = str(e)
            generated_result = None

        ground_truth_result = execution_service.execute_query(ground_truth_sql)

        exact_match = normalize_sql(generated_sql) == normalize_sql(ground_truth_sql)
        result_match = (
            generated_result == ground_truth_result
            if generated_sql_error is None
            else False
        )

        if exact_match:
            exact_match_count += 1
        if result_match:
            result_match_count += 1

        print(f"Ground Truth SQL: {ground_truth_sql}")
        print(f"Generated SQL: {generated_sql}")
        print(f"Exact SQL Match: {exact_match}")
        print(f"Result Match: {result_match}")
        if generated_sql_error:
            print(f"Generated SQL Execution Error: {generated_sql_error}")
        print("-" * 80)

    print("\n=== Evaluation Summary ===")
    print(f"Total Cases: {total}")
    print(f"Exact SQL Match: {exact_match_count}/{total}")
    print(f"Result Match: {result_match_count}/{total}")


if __name__ == "__main__":
    main()
