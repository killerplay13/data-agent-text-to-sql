import argparse
import json
from pathlib import Path

from app.services.execution_service import ExecutionService
from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATASET_PATH = BASE_DIR / "tests" / "test_dataset.json"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "eval_report.json"


def load_test_dataset():
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_sql(sql: str) -> str:
    normalized = " ".join(sql.strip().lower().split())
    return normalized.rstrip(";")


def normalize_result_rows(rows: list[dict] | None) -> list[str]:
    if rows is None:
        return []

    return sorted(
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        for row in rows
    )


def preview_rows(rows: list[dict] | None, max_rows: int = 3):
    if not rows:
        return []

    return rows[:max_rows]


def format_rate(count: int, total: int) -> str:
    if total == 0:
        return "0/0 (0.0%)"

    return f"{count}/{total} ({(count / total) * 100:.1f}%)"


def evaluate_case(
    index: int,
    case: dict,
    retrieval_service: RetrievalService,
    sql_service: SQLGenerationService,
    execution_service: ExecutionService,
    preview_limit: int,
) -> dict:
    question = case["question"]
    ground_truth_sql = case["ground_truth_sql"]

    retrieval_result = retrieval_service.retrieve(question)
    generated_sql = sql_service.generate_sql(question, retrieval_result)

    generated_result = None
    error_message = None
    try:
        generated_result = execution_service.execute_query(generated_sql)
    except Exception as exc:
        error_message = str(exc)

    try:
        ground_truth_result = execution_service.execute_query(ground_truth_sql)
    except Exception as exc:
        raise ValueError(
            f"Ground truth SQL failed for case {index}: {question}. Reason: {exc}"
        ) from exc

    sql_valid = error_message is None
    exact_match = normalize_sql(generated_sql) == normalize_sql(ground_truth_sql)
    result_match = (
        normalize_result_rows(generated_result)
        == normalize_result_rows(ground_truth_result)
        if sql_valid
        else False
    )

    sql_templates = retrieval_result.get("sql_templates", [])
    first_template = None
    if sql_templates:
        first_template = (
            sql_templates[0].get("name")
            or sql_templates[0].get("question_example")
            or "Unnamed template"
        )

    return {
        "case": index,
        "question": question,
        "retrieved_sql_template": first_template,
        "retrieved_schema_count": len(retrieval_result.get("schema_docs", [])),
        "retrieved_business_context_count": len(
            retrieval_result.get("business_context", [])
        ),
        "generated_sql": generated_sql,
        "ground_truth_sql": ground_truth_sql,
        "sql_valid": sql_valid,
        "exact_match": exact_match,
        "result_match": result_match,
        "generated_result_preview": preview_rows(generated_result, preview_limit),
        "ground_truth_result_preview": preview_rows(ground_truth_result, preview_limit),
        "error_message": error_message,
    }


def print_case_report(case_report: dict):
    print(f"Case {case_report['case']}")
    print(f"Question: {case_report['question']}")
    print(f"Retrieved SQL Template: {case_report['retrieved_sql_template'] or 'None'}")
    print(f"Retrieved Schema Docs: {case_report['retrieved_schema_count']}")
    print(
        "Retrieved Business Context: "
        f"{case_report['retrieved_business_context_count']}"
    )
    print(f"Generated SQL: {case_report['generated_sql']}")
    print(f"Ground Truth SQL: {case_report['ground_truth_sql']}")
    print(f"SQL Valid: {case_report['sql_valid']}")
    print(f"Exact SQL Match: {case_report['exact_match']}")
    print(f"Result Match: {case_report['result_match']}")
    print(f"Generated Result Preview: {case_report['generated_result_preview']}")
    print(
        "Ground Truth Result Preview: "
        f"{case_report['ground_truth_result_preview']}"
    )
    if case_report["error_message"]:
        print(f"Error Message: {case_report['error_message']}")
    print("-" * 80)


def build_summary(case_reports: list[dict]) -> dict:
    total = len(case_reports)
    valid_sql_count = sum(1 for report in case_reports if report["sql_valid"])
    exact_match_count = sum(1 for report in case_reports if report["exact_match"])
    result_match_count = sum(1 for report in case_reports if report["result_match"])

    return {
        "total_cases": total,
        "valid_sql_count": valid_sql_count,
        "exact_match_count": exact_match_count,
        "result_match_count": result_match_count,
        "valid_sql_rate": format_rate(valid_sql_count, total),
        "exact_match_rate": format_rate(exact_match_count, total),
        "result_match_rate": format_rate(result_match_count, total),
    }


def print_summary(summary: dict):
    print("\n=== Evaluation Summary ===")
    print(f"Total Cases: {summary['total_cases']}")
    print(f"Valid SQL Rate: {summary['valid_sql_rate']}")
    print(f"Exact SQL Match Rate: {summary['exact_match_rate']}")
    print(f"Result Match Rate: {summary['result_match_rate']}")


def save_report(report_path: Path, summary: dict, case_reports: list[dict]):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "cases": case_reports,
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL evaluation.")
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Path to save the JSON evaluation report.",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=3,
        help="Number of result rows to include in previews.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    retrieval_service = RetrievalService()
    sql_service = SQLGenerationService()
    execution_service = ExecutionService()

    test_cases = load_test_dataset()
    case_reports = []

    print("=== Evaluation Start ===\n")

    for index, case in enumerate(test_cases, start=1):
        case_report = evaluate_case(
            index,
            case,
            retrieval_service,
            sql_service,
            execution_service,
            args.preview_rows,
        )
        case_reports.append(case_report)
        print_case_report(case_report)

    summary = build_summary(case_reports)
    print_summary(summary)
    save_report(args.report, summary, case_reports)
    print(f"Report saved to: {args.report}")


if __name__ == "__main__":
    main()
