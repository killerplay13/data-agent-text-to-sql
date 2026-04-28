import argparse
import json
from pathlib import Path

from app.services.execution_service import ExecutionService
from app.services.retrieval_service import RetrievalService
from app.services.sql_generation_service import SQLGenerationService


BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DATASET_PATH = BASE_DIR / "tests" / "test_dataset.json"
DEFAULT_REPORT_PATH = BASE_DIR / "reports" / "eval_report.json"

RETRIEVAL_CATEGORY_CONFIG = {
    "schema": {
        "retrieval_key": "schema_docs",
        "ground_truth_key": "relevant_schema",
        "label": "Schema",
        "identifier_keys": ("table_name",),
    },
    "template": {
        "retrieval_key": "sql_templates",
        "ground_truth_key": "relevant_templates",
        "label": "Template",
        "identifier_keys": ("name", "question_example"),
    },
    "context": {
        "retrieval_key": "business_context",
        "ground_truth_key": "relevant_context",
        "label": "Context",
        "identifier_keys": ("topic",),
    },
}


def load_test_dataset():
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def get_expected_sql(case: dict) -> str:
    sql = case.get("ground_truth_sql") or case.get("expected_sql")
    if not sql:
        raise ValueError("Each test case must include ground_truth_sql or expected_sql.")

    return sql


def normalize_sql(sql: str) -> str:
    normalized = " ".join(sql.strip().lower().split())
    return normalized.rstrip(";")


def normalize_identifier(value: str) -> str:
    return " ".join(value.strip().lower().split())


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


def unique_preserving_order(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []

    for value in values:
        normalized_value = normalize_identifier(value)
        if normalized_value in seen:
            continue

        seen.add(normalized_value)
        unique_values.append(value)

    return unique_values


def get_item_identifier(item: dict, identifier_keys: tuple[str, ...]) -> str:
    for key in identifier_keys:
        value = item.get(key)
        if value:
            return str(value)

    return "unknown"


def get_retrieved_identifiers(items: list[dict], identifier_keys: tuple[str, ...]) -> list[str]:
    identifiers = [get_item_identifier(item, identifier_keys) for item in items]
    return unique_preserving_order(identifiers)


def evaluate_retrieval_category(
    case: dict,
    retrieval_result: dict,
    category_config: dict,
) -> dict:
    relevant_items = unique_preserving_order(case.get(category_config["ground_truth_key"], []))
    retrieved_items = retrieval_result.get(category_config["retrieval_key"], []) or []
    retrieved_identifiers = get_retrieved_identifiers(
        retrieved_items,
        category_config["identifier_keys"],
    )

    if not relevant_items:
        return {
            "ground_truth": [],
            "retrieved": retrieved_identifiers,
            "evaluated": False,
            "matched_items": [],
            "hit": None,
            "recall_numerator": 0,
            "recall_denominator": 0,
            "recall_rate": None,
        }

    retrieved_lookup = {normalize_identifier(item) for item in retrieved_identifiers}
    matched_items = [
        item for item in relevant_items if normalize_identifier(item) in retrieved_lookup
    ]
    recall_denominator = len(relevant_items)
    recall_numerator = len(matched_items)

    return {
        "ground_truth": relevant_items,
        "retrieved": retrieved_identifiers,
        "evaluated": True,
        "matched_items": matched_items,
        "hit": recall_numerator > 0,
        "recall_numerator": recall_numerator,
        "recall_denominator": recall_denominator,
        "recall_rate": recall_numerator / recall_denominator if recall_denominator else None,
    }


def build_retrieval_evaluation(case: dict, retrieval_result: dict) -> dict:
    evaluation = {}

    for category_name, category_config in RETRIEVAL_CATEGORY_CONFIG.items():
        evaluation[category_name] = evaluate_retrieval_category(
            case,
            retrieval_result,
            category_config,
        )

    return evaluation


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
    ground_truth_sql = get_expected_sql(case)

    retrieval_result = retrieval_service.retrieve(question)
    sql_templates = retrieval_result.get("sql_templates", [])
    first_template = None
    if sql_templates:
        first_template = (
            sql_templates[0].get("name")
            or sql_templates[0].get("question_example")
            or "Unnamed template"
        )
    retrieval_evaluation = build_retrieval_evaluation(case, retrieval_result)

    generated_sql = ""
    generated_result = []
    error_message = None
    try:
        generated_sql = sql_service.generate_sql(question, retrieval_result)
        generated_result = execution_service.execute_query(generated_sql)
    except Exception as exc:
        error_message = str(exc)

    if "expected_result" in case:
        ground_truth_result = case.get("expected_result")
    else:
        try:
            ground_truth_result = execution_service.execute_query(ground_truth_sql)
        except Exception as exc:
            raise ValueError(
                f"Ground truth SQL failed for case {index}: {question}. Reason: {exc}"
            ) from exc

    sql_valid = error_message is None and bool(generated_sql)
    exact_match = (
        normalize_sql(generated_sql) == normalize_sql(ground_truth_sql)
        if generated_sql
        else False
    )
    result_match = (
        normalize_result_rows(generated_result)
        == normalize_result_rows(ground_truth_result)
        if sql_valid
        else False
    )

    return {
        "case": index,
        "question": question,
        "retrieved_sql_template": first_template,
        "retrieved_schema_count": len(retrieval_result.get("schema_docs", [])),
        "retrieved_business_context_count": len(
            retrieval_result.get("business_context", [])
        ),
        "retrieved_schema_docs": retrieval_evaluation["schema"]["retrieved"],
        "retrieved_sql_templates": retrieval_evaluation["template"]["retrieved"],
        "retrieved_business_context": retrieval_evaluation["context"]["retrieved"],
        "relevant_schema": retrieval_evaluation["schema"]["ground_truth"],
        "relevant_templates": retrieval_evaluation["template"]["ground_truth"],
        "relevant_context": retrieval_evaluation["context"]["ground_truth"],
        "retrieval_evaluation": retrieval_evaluation,
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
    print(f"Retrieved SQL Template Top-1: {case_report['retrieved_sql_template'] or 'None'}")
    print(f"Retrieved Schema Doc Count: {case_report['retrieved_schema_count']}")
    print(
        "Retrieved Business Context Count: "
        f"{case_report['retrieved_business_context_count']}"
    )
    print(f"Retrieved Schema Docs: {case_report['retrieved_schema_docs']}")
    print(f"Retrieved SQL Templates: {case_report['retrieved_sql_templates']}")
    print(f"Retrieved Business Context: {case_report['retrieved_business_context']}")
    if case_report["relevant_schema"]:
        print(f"Relevant Schema Ground Truth: {case_report['relevant_schema']}")
    if case_report["relevant_templates"]:
        print(f"Relevant Template Ground Truth: {case_report['relevant_templates']}")
    if case_report["relevant_context"]:
        print(f"Relevant Context Ground Truth: {case_report['relevant_context']}")
    print(f"Generated SQL: {case_report['generated_sql']}")
    print(f"Expected SQL: {case_report['ground_truth_sql']}")
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

    retrieval_metrics = {}
    for category_name, category_config in RETRIEVAL_CATEGORY_CONFIG.items():
        evaluations = [
            report["retrieval_evaluation"][category_name]
            for report in case_reports
            if report["retrieval_evaluation"][category_name]["evaluated"]
        ]
        hit_count = sum(1 for item in evaluations if item["hit"])
        eligible_cases = len(evaluations)
        recall_numerator = sum(item["recall_numerator"] for item in evaluations)
        recall_denominator = sum(item["recall_denominator"] for item in evaluations)

        retrieval_metrics[category_name] = {
            "label": category_config["label"],
            "eligible_cases": eligible_cases,
            "hit_count": hit_count,
            "hit_rate": format_rate(hit_count, eligible_cases),
            "recall_numerator": recall_numerator,
            "recall_denominator": recall_denominator,
            "recall_rate": format_rate(recall_numerator, recall_denominator),
        }

    return {
        "total_cases": total,
        "valid_sql_count": valid_sql_count,
        "exact_match_count": exact_match_count,
        "result_match_count": result_match_count,
        "valid_sql_rate": format_rate(valid_sql_count, total),
        "exact_match_rate": format_rate(exact_match_count, total),
        "result_match_rate": format_rate(result_match_count, total),
        "retrieval_metrics": retrieval_metrics,
    }


def print_summary(summary: dict):
    print("\n=== Evaluation Summary ===")
    print(f"Total Cases: {summary['total_cases']}")
    print("\nSQL Metrics:")
    print(f"Valid SQL Rate: {summary['valid_sql_rate']}")
    print(f"Exact SQL Match Rate: {summary['exact_match_rate']}")
    print(f"Result Match Rate: {summary['result_match_rate']}")
    print("\nRetrieval Metrics:")
    for category_name in ("schema", "template", "context"):
        metrics = summary["retrieval_metrics"][category_name]
        print(f"{metrics['label']} Hit@K: {metrics['hit_rate']}")
        print(f"{metrics['label']} Recall@K: {metrics['recall_rate']}")


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
