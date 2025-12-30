import os
from langchain_openai import ChatOpenAI


class OpenAILM:
    """
    Loader đơn giản cho OpenAI Chat Model.
    Đọc OPENAI_API_KEY từ biến môi trường.
    """

    def __init__(self, model_name: str = "gpt-4.1-mini", temperature: float = 0.3):
        self.model_name = model_name
        self.temperature = temperature

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY chưa được thiết lập trong biến môi trường. "
                "Hãy export OPENAI_API_KEY trước khi chạy app."
            )

    def get_llm(self) -> ChatOpenAI:
        """
        Trả về instance ChatOpenAI đã cấu hình sẵn.
        """
        llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
        )
        return llm