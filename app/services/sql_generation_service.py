import json
import re
from pathlib import Path

from langchain_openai import ChatOpenAI
from app.core.config import settings


SQL_CODE_FENCE_PATTERN = re.compile(
    r"```(?:sql|sqlite)?\s*(?P<sql>.*?)```",
    re.IGNORECASE | re.DOTALL,
)
SQL_SELECT_PATTERN = re.compile(r"\bselect\b", re.IGNORECASE)
SQL_KEYWORD_SPACING_PATTERN = re.compile(
    r"([A-Za-z0-9_)\]])"
    r"(SELECT\b|FROM\b|WHERE\b|GROUP\s+BY\b|ORDER\s+BY\b|HAVING\b|"
    r"LIMIT\b|JOIN\b|INNER\s+JOIN\b|LEFT\s+JOIN\b|RIGHT\s+JOIN\b|"
    r"FULL\s+JOIN\b|CROSS\s+JOIN\b|ON\b)",
    re.IGNORECASE,
)
SQL_ALIAS_SPACING_PATTERN = re.compile(
    r"\b(SELECT|FROM|WHERE|JOIN|ON|BY|HAVING|LIMIT)(?=[A-Za-z_][A-Za-z0-9_]*\.)",
    re.IGNORECASE,
)
QUOTED_SQL_TEXT_PATTERN = re.compile(r"('(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")")
JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
SQL_TABLE_REFERENCE_PATTERN = re.compile(
    r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)
SQL_SELECT_CLAUSE_PATTERN = re.compile(
    r"\bselect\b(?P<select>.*?)\bfrom\b",
    re.IGNORECASE | re.DOTALL,
)
QUERIED_BRANCH_NAME_PATTERN = re.compile(r"\b([A-Za-z]+ Branch)\b", re.IGNORECASE)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
KB_DIR = BASE_DIR / "kb"
FULL_SCHEMA_DOCS_PATH = KB_DIR / "schema_docs.json"

ENTITY_TABLE_MAP = {
    "customer": "customers",
    "customers": "customers",
    "relationship_manager": "relationship_managers",
    "relationship_managers": "relationship_managers",
    "rm": "relationship_managers",
    "branch": "branches",
    "branches": "branches",
    "deposit": "deposits",
    "deposits": "deposits",
}


class SQLGenerationService:
    def __init__(self):
        self.llm = None
        self.last_query_plan = {}
        self._schema_docs = self._load_full_schema_docs()
        self._metric_to_tables = self._build_metric_table_map(self._schema_docs)

        if settings.OPENROUTER_API_KEY:
            self.llm = ChatOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url=settings.OPENROUTER_BASE_URL,
                model=settings.OPENROUTER_MODEL,
                temperature=0
            )

    def build_prompt(
        self,
        user_query: str,
        retrieval_result: dict,
        query_plan: dict | None = None,
    ) -> str:
        sections = [
            self._build_plan_block(query_plan),
            self._build_instruction_block(),
            self._build_schema_block(retrieval_result),
            self._build_template_block(retrieval_result),
            self._build_business_context_block(retrieval_result),
            self._build_user_query_block(user_query),
        ]
        return "\n\n".join(section for section in sections if section)

    def _build_instruction_block(self) -> str:
        instructions = [
            "You are a senior Text-to-SQL assistant.",
            "Generate a single valid SQLite SELECT query that answers the user query.",
            "",
            "Instructions:",
            "1. Output exactly one SQL SELECT query and nothing else.",
            "2. The response must start with SELECT.",
            "3. Do not include explanations, markdown, code fences, or bullet points.",
            "4. Only generate SQL for SQLite.",
            "5. You MUST only use tables and columns provided in the schema.",
            "6. Map user query terms to schema tables and columns.",
            "7. Do not invent tables or columns.",
            "8. Use schema to determine joins and filters.",
            "9. Always prioritize correctness based on the user query and schema.",
            (
                "10. Use SQL templates as reference examples when helpful, "
                "but do NOT blindly copy them."
            ),
            "11. You may adapt or ignore templates if they do not match the user query.",
            "12. Reuse only the parts of a template that are consistent with the query intent and schema.",
            "13. Use business context definitions to interpret business terms.",
            "14. Apply business rules only when relevant to the query.",
            "15. Only use aggregation, ranking, filtering, ordering, or limits when needed for the user query.",
            (
                '16. If the user asks for ranking with words like "highest", "top", '
                '"maximum", or "largest", identify the correct metric column and use '
                "ORDER BY <metric> DESC with LIMIT when returning the top result set."
            ),
            (
                '17. If the user asks for lowest-value ranking with words like "lowest", '
                '"smallest", or "minimum", identify the correct metric column and use '
                "ORDER BY <metric> ASC with LIMIT when returning the lowest result set."
            ),
            "18. If the user asks for counts, use COUNT() on the correct schema-backed entity.",
            "19. If the user asks for totals, use SUM() on the correct metric column.",
            "20. If the user asks for averages, use AVG() on the correct metric column.",
            (
                '21. If the query compares groups such as "which branch", '
                '"which manager", or other grouped categories, use GROUP BY on the '
                "appropriate grouping column when needed."
            ),
            (
                "22. Always identify the correct metric column from the schema or "
                "business context before applying ranking, COUNT(), SUM(), AVG(), "
                "or GROUP BY logic."
            ),
            "23. Before generating SQL, identify the business metric implied by the query.",
            (
                "24. Map business terms to the correct schema-backed metric column using "
                "the schema and business context."
            ),
            (
                '25. Examples of correct metric grounding: "deposit" -> deposit_amount, '
                '"number of customers" -> COUNT(customer_id), '
                '"average deposit" -> AVG(deposit_amount).'
            ),
            (
                "26. Do NOT use identifier columns such as customer_id, branch_id, rm_id, "
                "or other *_id fields as ranking metrics unless the user explicitly asks for IDs."
            ),
            (
                "27. Ranking must use a business metric column or derived business metric, "
                "not a surrogate key."
            ),
            (
                "28. Distinguish between the entity to return and the metric used to rank or filter."
            ),
            (
                '29. Example: "Who has the highest deposit?" means return customer_name '
                "but rank by deposit_amount DESC with LIMIT 1."
            ),
            (
                "30. If the metric column lives in another table, join to that table instead "
                "of staying on a table that only contains the return field."
            ),
            (
                '31. Example: "Who has the highest deposit?" -> return customer_name, '
                "rank by deposit_amount DESC, LIMIT 1."
            ),
            (
                '32. Example: "Which branch has the most customers?" -> GROUP BY branch, '
                "COUNT(customer_id), ORDER BY count DESC, LIMIT 1."
            ),
            (
                '33. Example: "Which RM has the highest average deposit?" -> GROUP BY rm, '
                "AVG(deposit_amount), ORDER BY avg DESC, LIMIT 1."
            ),
        ]
        return "\n".join(instructions)

    def _build_schema_block(self, retrieval_result: dict) -> str:
        schema_docs = retrieval_result.get("schema_docs", [])
        if not schema_docs:
            return "Schema:\nNo schema documents were retrieved."

        schema_entries = []
        for item in schema_docs:
            columns = "\n".join(
                [
                    (
                        f"  - {col['name']} ({col.get('data_type', 'UNKNOWN')}): "
                        f"{col.get('description', 'No description provided.')}"
                    )
                    for col in item["columns"]
                ]
            )
            schema_entries.append(
                "\n".join(
                    [
                        f"Table: {item['table_name']}",
                        f"Description: {item['description']}",
                        "Columns:",
                        columns,
                    ]
                )
            )

        return "Schema:\n" + "\n\n".join(schema_entries)

    def _build_plan_block(self, query_plan: dict | None) -> str:
        if not query_plan:
            return ""

        return "\n".join(
            [
                "Query Plan:",
                json.dumps(query_plan, ensure_ascii=False, indent=2),
                "",
                "Plan Enforcement Rules:",
                "1. You MUST follow the Query Plan exactly.",
                "2. Do NOT ignore or override the plan.",
                "3. The SQL must implement all fields in the plan.",
                "4. target_entity must determine the main entity returned by the SELECT clause.",
                "5. metric must appear in the SQL either directly or through the required aggregation.",
                '6. If aggregation is present, you MUST apply it exactly as defined in the plan.',
                '7. If group_by is present, you MUST use GROUP BY exactly as required by the plan.',
                '8. If order_by is present, you MUST use ORDER BY exactly as defined in the plan.',
                '9. If limit is present, you MUST use LIMIT with that exact value.',
                "10. If the SQL does not match the plan, it is incorrect.",
                "",
                "Plan Example:",
                "{",
                '  "target_entity": "customer",',
                '  "metric": "deposit_amount",',
                '  "aggregation": null,',
                '  "group_by": null,',
                '  "order_by": "deposit_amount DESC",',
                '  "limit": 1',
                "}",
                "",
                "Correct SQL Example:",
                "SELECT c.customer_name",
                "FROM customers c",
                "JOIN deposits d ON c.customer_id = d.customer_id",
                "ORDER BY d.deposit_amount DESC",
                "LIMIT 1",
                "",
                "Incorrect SQL Example:",
                "SELECT customer_name FROM customers LIMIT 1",
            ]
        )

    def _build_template_block(self, retrieval_result: dict) -> str:
        templates = retrieval_result.get("sql_templates", [])
        if not templates:
            return (
                "SQL Templates:\n"
                "No SQL templates were retrieved. Generate SQL directly from the user query and schema."
            )

        template_entries = []
        for item in templates:
            template_entries.append(
                "\n".join(
                    [
                        f"Template Name: {item['name']}",
                        f"Example Question: {item['question_example']}",
                        f"SQL Example: {item['sql']}",
                        f"Business Description: {item['business_description']}",
                    ]
                )
            )

        return (
            "SQL Templates:\n"
            "Use these as optional reference examples. Prefer query-schema correctness over template similarity.\n\n"
            + "\n\n".join(template_entries)
        )

    def _build_business_context_block(self, retrieval_result: dict) -> str:
        business_context = retrieval_result.get("business_context", [])
        if not business_context:
            return "Business Context:\nNo business context was retrieved."

        context_entries = []
        for item in business_context:
            context_entries.append(
                "\n".join(
                    [
                        f"Topic: {item['topic']}",
                        f"Definition: {item['description']}",
                    ]
                )
            )

        return "Business Context:\n" + "\n\n".join(context_entries)

    def _build_user_query_block(self, user_query: str) -> str:
        return "\n".join(
            [
                "User Query:",
                user_query,
                "",
                "Return the SQL query only.",
            ]
        )

    def build_plan_prompt(self, user_query: str, retrieval_result: dict) -> str:
        sections = [
            "\n".join(
                [
                    "You are a senior Text-to-SQL planner.",
                    "Analyze the user query and return a JSON object that plans the SQL query.",
                    "",
                    "Planning Rules:",
                    "1. Return valid JSON only.",
                    (
                        '2. Use exactly these keys: "target_entity", "metric", '
                        '"aggregation", "group_by", "order_by", "limit".'
                    ),
                    "3. target_entity should describe the main entity the query wants returned or compared.",
                    "4. metric should identify the business metric column or derived metric implied by the query.",
                    '5. aggregation must be one of null, "COUNT", "SUM", or "AVG".',
                    "6. group_by should be the grouping column when the query compares groups; otherwise null.",
                    "7. order_by should describe the intended ranking or sorting expression; otherwise null.",
                    "8. limit should be an integer when the query asks for a top or bottom result; otherwise null.",
                    "9. Identify the entity requested by who/which phrasing.",
                    "10. Identify the correct metric using schema and business context.",
                    "11. Use COUNT for counts, SUM for totals, AVG for averages.",
                    "12. Use ranking and limit for highest, top, maximum, largest, lowest, smallest, or minimum queries.",
                    "13. Do not use *_id columns as business ranking metrics unless the user explicitly asks for IDs.",
                    "14. If the metric lives in another table, plan for it anyway by naming the correct metric column.",
                ]
            ),
            self._build_schema_block(retrieval_result),
            self._build_template_block(retrieval_result),
            self._build_business_context_block(retrieval_result),
            "\n".join(
                [
                    "User Query:",
                    user_query,
                    "",
                    "Return the JSON plan only.",
                ]
            ),
        ]
        return "\n\n".join(section for section in sections if section)

    def build_repair_prompt(
        self,
        user_query: str,
        retrieval_result: dict,
        previous_sql: str,
        error_message: str,
        query_plan: dict | None = None,
    ) -> str:
        sections = [
            self._build_plan_block(query_plan),
            "\n".join(
                [
                    "You are a senior Text-to-SQL assistant.",
                    "Repair the SQL query so it becomes a single valid SQLite SELECT query.",
                    "",
                    "Instructions:",
                    "1. Return corrected SQL only.",
                    "2. The response must start with SELECT.",
                    "3. Do not include explanations, markdown, code fences, or bullet points.",
                    "4. Use only tables and columns provided in the schema.",
                    "5. Fix the SQL based on the user query and the reported error.",
                    "6. You may ignore the previous SQL structure if it is incorrect.",
                    "7. Use business context only when it helps interpret the user query.",
                    "8. If a Query Plan is provided, the repaired SQL MUST satisfy it exactly.",
                ]
            ),
            self._build_schema_block(retrieval_result),
            self._build_template_block(retrieval_result),
            self._build_business_context_block(retrieval_result),
            "\n".join(
                [
                    "User Query:",
                    user_query,
                    "",
                    "Previous SQL:",
                    previous_sql or "None",
                    "",
                    "Validation or Execution Error:",
                    error_message,
                    "",
                    "Return the corrected SQL query only.",
                ]
            ),
        ]
        return "\n\n".join(section for section in sections if section)

    def fallback_sql(self, retrieval_result: dict) -> str | None:
        templates = retrieval_result.get("sql_templates", [])
        if templates:
            return templates[0]["sql"]
        return None

    def clean_generated_sql(self, content: str) -> str:
        sql = self._extract_sql(content)

        if not sql:
            return ""

        return self._restore_keyword_spacing(sql)

    def _extract_sql(self, content: str) -> str:
        raw_content = content.strip()
        fenced_sql = SQL_CODE_FENCE_PATTERN.search(raw_content)

        if fenced_sql:
            return fenced_sql.group("sql").strip()

        select_match = SQL_SELECT_PATTERN.search(raw_content)
        if not select_match:
            return ""

        sql = raw_content[select_match.start():]
        final_semicolon_index = sql.rfind(";")
        if final_semicolon_index != -1:
            sql = sql[:final_semicolon_index + 1]

        return sql.strip()

    def _restore_keyword_spacing(self, sql: str) -> str:
        parts = QUOTED_SQL_TEXT_PATTERN.split(sql)

        for index in range(0, len(parts), 2):
            parts[index] = SQL_KEYWORD_SPACING_PATTERN.sub(r"\1 \2", parts[index])
            parts[index] = SQL_ALIAS_SPACING_PATTERN.sub(r"\1 ", parts[index])

        return "".join(parts)

    def repair_sql(
        self,
        user_query: str,
        retrieval_result: dict,
        previous_sql: str,
        error_message: str,
        query_plan: dict | None = None,
    ) -> str:
        if not self.llm:
            return ""

        repair_prompt = self.build_repair_prompt(
            user_query,
            retrieval_result,
            previous_sql,
            error_message,
            query_plan,
        )

        try:
            response = self.llm.invoke(repair_prompt)
            print(f"Raw LLM SQL repair response: {response.content}")
            repaired_sql = self.clean_generated_sql(response.content)
            print(f"Cleaned repaired SQL: {repaired_sql}")
            if not repaired_sql or not repaired_sql.lower().startswith("select"):
                return ""
            return repaired_sql
        except Exception as e:
            print(f"LLM SQL repair failed. Reason: {e}")
            return ""

    def plan_query(self, user_query: str, retrieval_result: dict) -> dict:
        if not self.llm:
            return {}

        plan_prompt = self.build_plan_prompt(user_query, retrieval_result)

        try:
            response = self.llm.invoke(plan_prompt)
            print(f"Raw LLM query plan response: {response.content}")
            query_plan = self._parse_query_plan(response.content)
            print(f"Parsed query plan: {query_plan}")
            return query_plan
        except Exception as e:
            print(f"LLM query planning failed. Reason: {e}")
            return {}

    def _fallback_or_raise(self, retrieval_result: dict, reason: str) -> str:
        fallback_sql = self.fallback_sql(retrieval_result)
        if fallback_sql:
            return fallback_sql
        raise ValueError(reason)

    def _parse_query_plan(self, content: str) -> dict:
        raw_content = content.strip()
        json_match = JSON_OBJECT_PATTERN.search(raw_content)
        if json_match:
            raw_content = json_match.group(0)

        try:
            parsed = json.loads(raw_content)
        except json.JSONDecodeError:
            return {}

        if not isinstance(parsed, dict):
            return {}

        normalized_plan = {
            "target_entity": parsed.get("target_entity"),
            "metric": parsed.get("metric"),
            "aggregation": parsed.get("aggregation"),
            "group_by": parsed.get("group_by"),
            "order_by": parsed.get("order_by"),
            "limit": parsed.get("limit"),
        }
        return normalized_plan

    def plan_compliance_error(
        self,
        user_query: str,
        sql: str,
        query_plan: dict | None,
    ) -> str | None:
        if not query_plan or not sql:
            return None

        sql_lower = sql.lower()
        used_tables = self._extract_sql_tables(sql_lower)
        select_clause = self._extract_select_clause(sql_lower)

        metric = query_plan.get("metric")
        aggregation = query_plan.get("aggregation")
        target_entity = query_plan.get("target_entity")
        target_table = self._entity_table(target_entity)
        metric_table = self._required_metric_table(metric)

        if metric and not self._has_metric_projection(sql_lower, select_clause, metric, aggregation):
            return (
                f"SQL plan compliance failed: missing metric projection for '{metric}'. "
                "The SQL must include the metric in SELECT or include the required aggregation."
            )

        if metric_table and metric_table not in used_tables:
            return (
                f"SQL plan compliance failed: missing required table '{metric_table}' "
                f"for metric '{metric}'."
            )

        if not self._has_required_join_path(used_tables, target_table, metric_table):
            return (
                "SQL plan compliance failed: missing required join path between the "
                f"target entity '{target_entity}' and metric '{metric}'."
            )

        filter_error = self._filter_compliance_error(user_query, sql_lower, used_tables)
        if filter_error:
            return filter_error

        return None

    def _load_full_schema_docs(self) -> list[dict]:
        with open(FULL_SCHEMA_DOCS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)

    def _build_metric_table_map(self, schema_docs: list[dict]) -> dict[str, set[str]]:
        metric_to_tables: dict[str, set[str]] = {}
        for item in schema_docs:
            table_name = item.get("table_name")
            if not table_name:
                continue

            for column in item.get("columns", []):
                column_name = column.get("name")
                if not column_name:
                    continue
                metric_to_tables.setdefault(column_name.lower(), set()).add(
                    table_name.lower()
                )

        return metric_to_tables

    def _extract_sql_tables(self, sql: str) -> set[str]:
        return {
            match.group(1).lower()
            for match in SQL_TABLE_REFERENCE_PATTERN.finditer(sql)
        }

    def _extract_select_clause(self, sql: str) -> str:
        select_match = SQL_SELECT_CLAUSE_PATTERN.search(sql)
        if not select_match:
            return ""
        return select_match.group("select")

    def _entity_table(self, target_entity: str | None) -> str | None:
        if not target_entity:
            return None
        return ENTITY_TABLE_MAP.get(target_entity.lower())

    def _required_metric_table(self, metric: str | None) -> str | None:
        if not metric:
            return None

        metric_tables = self._metric_to_tables.get(metric.lower(), set())
        if len(metric_tables) == 1:
            return next(iter(metric_tables))
        return None

    def _has_metric_projection(
        self,
        sql: str,
        select_clause: str,
        metric: str,
        aggregation: str | None,
    ) -> bool:
        metric_lower = metric.lower()
        if metric_lower in select_clause:
            return True

        if aggregation:
            aggregation_lower = aggregation.lower()
            aggregated_metric_pattern = re.compile(
                rf"\b{re.escape(aggregation_lower)}\s*\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\.)?{re.escape(metric_lower)}\s*\)",
                re.IGNORECASE,
            )
            if aggregated_metric_pattern.search(sql):
                return True

        return False

    def _has_required_join_path(
        self,
        used_tables: set[str],
        target_table: str | None,
        metric_table: str | None,
    ) -> bool:
        if not metric_table or not target_table:
            return True

        if metric_table == target_table:
            return True

        if metric_table == "deposits" and target_table in {"customers", "deposits"}:
            return "deposits" in used_tables

        if metric_table == "deposits" and target_table in {
            "relationship_managers",
            "branches",
        }:
            return {
                "deposits",
                "customers",
                target_table,
            }.issubset(used_tables)

        return True

    def _filter_compliance_error(
        self,
        user_query: str,
        sql: str,
        used_tables: set[str],
    ) -> str | None:
        user_query_lower = user_query.lower()

        branch_name_match = QUERIED_BRANCH_NAME_PATTERN.search(user_query)
        if branch_name_match:
            branch_name = branch_name_match.group(1).lower()
            if "branches" not in used_tables:
                return (
                    "SQL plan compliance failed: branch-scoped query is missing the "
                    "'branches' table."
                )
            if branch_name not in sql or "branch_name" not in sql or "where" not in sql:
                return (
                    "SQL plan compliance failed: missing branch filter for "
                    f"'{branch_name_match.group(1)}'."
                )

        if self._requires_rm_filter(user_query_lower):
            if "relationship_managers" not in used_tables:
                return (
                    "SQL plan compliance failed: relationship-manager-scoped query is "
                    "missing the 'relationship_managers' table."
                )
            if "rm_name" not in sql or "where" not in sql:
                return (
                    "SQL plan compliance failed: missing relationship manager filter."
                )

        return None

    def _requires_rm_filter(self, user_query_lower: str) -> bool:
        if "managed by" in user_query_lower and "each relationship manager" not in user_query_lower:
            return True

        if "relationship manager" in user_query_lower and any(
            token in user_query_lower for token in ["alice", "brian", "cindy", "david", "eva"]
        ):
            return True

        return False

    def _repair_for_compliance(
        self,
        user_query: str,
        retrieval_result: dict,
        sql: str,
        query_plan: dict | None,
        error_message: str,
    ) -> str:
        repaired_sql = self.repair_sql(
            user_query,
            retrieval_result,
            sql,
            error_message,
            query_plan,
        )
        if not repaired_sql or not repaired_sql.lower().startswith("select"):
            return ""

        if self.plan_compliance_error(user_query, repaired_sql, query_plan):
            return ""

        return repaired_sql

    def generate_sql(self, user_query: str, retrieval_result: dict) -> str:
        if not self.llm:
            return self._fallback_or_raise(
                retrieval_result,
                "LLM SQL generation is unavailable and no fallback SQL template was retrieved.",
            )

        query_plan = self.plan_query(user_query, retrieval_result)
        self.last_query_plan = query_plan
        prompt = self.build_prompt(user_query, retrieval_result, query_plan)

        try:
            response = self.llm.invoke(prompt)
            print(f"Raw LLM SQL response: {response.content}")
            sql = self.clean_generated_sql(response.content)
            print(f"Cleaned SQL: {sql}")
            if not sql or not sql.lower().startswith("select"):
                repaired_sql = self._repair_for_compliance(
                    user_query,
                    retrieval_result,
                    sql or response.content,
                    query_plan,
                    "Generated SQL was malformed or did not start with SELECT.",
                )
                if repaired_sql:
                    return repaired_sql
                return self._fallback_or_raise(
                    retrieval_result,
                    "SQL generation failed and no fallback SQL template was retrieved.",
                )
            compliance_error = self.plan_compliance_error(user_query, sql, query_plan)
            if compliance_error:
                print(compliance_error)
                repaired_sql = self._repair_for_compliance(
                    user_query,
                    retrieval_result,
                    sql,
                    query_plan,
                    compliance_error,
                )
                if repaired_sql:
                    return repaired_sql
                return self._fallback_or_raise(
                    retrieval_result,
                    "SQL generation failed plan compliance and no fallback SQL template was retrieved.",
                )

            return sql
        except Exception as e:
            print(f"LLM generation failed, fallback to template. Reason: {e}")
            return self._fallback_or_raise(
                retrieval_result,
                "LLM SQL generation failed and no fallback SQL template was retrieved.",
            )
