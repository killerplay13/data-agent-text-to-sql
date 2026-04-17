from app.services.retrieval_service import RetrievalService
from app.skills.base_skill import BaseSkill


class RetrievalSkill(BaseSkill):
    name = "retrieval"

    def __init__(self):
        self.service = RetrievalService()

    def execute(self, input: dict) -> dict:
        input["retrieval_result"] = self.service.retrieve(input["user_query"])
        return input
