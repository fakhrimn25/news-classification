import os
import sys
import json
import srsly
from openai import AsyncOpenAI
from typing import ClassVar, Dict
from configparser import ConfigParser

path_this = os.path.dirname(os.path.abspath(__file__))
path_project = os.path.dirname(os.path.join(path_this, ".."))
path_root = os.path.dirname(os.path.join(path_this, "../../.."))
sys.path.extend([path_root, path_project, path_this])

from tools import BaseAgent
from tools.utils.tool import NewsLabelFormat

class NewsClassification(BaseAgent):
    """
    Agent for classifying news content using an LLM-based approach.

    This class loads a system prompt from configuration and sends the
    news context to the LLM to obtain a structured classification result.
    """

    config: ClassVar[ConfigParser] = ConfigParser()
    config.read(os.path.join(path_root, "config.conf"))

    llm: AsyncOpenAI

    async def run(self, context: str) -> Dict:
        """
        Run the news classification process.

        Args:
            context (str): The news content or text to be classified.

        Returns:
            Dict: A dictionary containing the classification result
                  along with the LLM response ID.
        """
        path_json = os.path.join(path_root, self.config["default"]["agent_system_messages_path_general"])
        prompts = srsly.read_json(path_json)
        prompt_input = prompts["news_classification"]["system_message"].format(
            context=context
        )
        response = await self.llm.chat.completions.parse(
            model=self.config["llm"]["model"],
            messages=[
                {
                    "role": "user",
                    "content": prompt_input
                }
            ],
            response_format=NewsLabelFormat
        )
        response_id = response.id
        output_content = response.choices[0].message.content

        result = json.loads(output_content)
        result["response_id"] = response_id

        return result