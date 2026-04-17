from app.services.answer_service import AnswerService
from app.skills.base_skill import BaseSkill


class AnswerSkill(BaseSkill):
    name = "answer"

    def __init__(self):
        self.service = AnswerService()

    def execute(self, input: dict) -> dict:
        input["answer"] = self.service.generate_answer(
            input["user_query"],
            input["query_result"],
        )
        return input
