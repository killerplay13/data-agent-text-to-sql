from pathlib import Path
import json


BASE_DIR = Path(__file__).resolve().parent.parent
KB_DIR = BASE_DIR / "kb"


def load_json(file_name: str):
    path = KB_DIR / file_name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    schema_docs = load_json("schema_docs.json")
    sql_templates = load_json("sql_templates.json")
    business_context = load_json("business_context.json")

    print(f"Loaded schema docs: {len(schema_docs)}")
    print(f"Loaded SQL templates: {len(sql_templates)}")
    print(f"Loaded business context entries: {len(business_context)}")

    print("\nSample schema table:", schema_docs[0]["table_name"])
    print("Sample SQL template:", sql_templates[0]["name"])
    print("Sample business topic:", business_context[0]["topic"])


if __name__ == "__main__":
    main()