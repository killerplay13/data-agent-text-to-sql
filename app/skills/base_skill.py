class BaseSkill:
    name: str

    def execute(self, input: dict) -> dict:
        raise NotImplementedError
