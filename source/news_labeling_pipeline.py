import srsly
import asyncio
from loguru import logger
from openai import AsyncOpenAI
from typing import Dict, List, Any
from configparser import ConfigParser
from datasets import load_dataset, DownloadConfig

from tools import (
    NewsClassification,
    BaseAgent
)

class NewsLabelingPipeline(BaseAgent):
    """
    Pipeline for labeling news articles using an LLM-based classifier.

    This pipeline performs the following steps:
        1. Load configuration values.
        2. Download the news dataset from HuggingFace.
        3. Run LLM-based classification for each news item.
        4. Execute the classification tasks concurrently.
        5. Save the prediction results to a JSON file.

    The pipeline is designed to support asynchronous processing in order
    to improve throughput when labeling large datasets.
    """

    config: ConfigParser
    classifier: Any
    download_config: DownloadConfig

    def load_dataset(self) -> List[Dict]:
        """
        Load the news dataset from HuggingFace.

        Returns:
            List[Dict]: List of news data records.
        """

        dataset = load_dataset(
            self.config["hf"]["hf_news"],
            split="test",
            download_config=self.download_config,
        )

        return dataset.to_list()
    
    async def process_news(self, data: Dict) -> Dict:
        """
        Process and classify a single news item.

        Args:
            data (Dict): A dictionary containing news data
                         (title, content, label, id).

        Returns:
            Dict: Classification result containing:
                - id
                - true label
                - predicted label
                - LLM response id
        """

        context = f"Title: {data['title']}\nContent: {data['content']}"
        output = await self.classifier.run(context)

        return {
            "id": data["id"],
            "label": data["label"],
            "predicted_label": output["label"],
            "response_id": output["response_id"],
        }

    async def run(self, worker: int = 5) -> List[Dict]:
        """
        Execute the full labeling pipeline.

        Args:
            concurrency (int): Number of concurrent tasks.

        Returns:
            List[Dict]: List of prediction results.
        """
        logger.info("Starting dataset download from HuggingFace")

        dataset_list = self.load_dataset()

        logger.info(
            "Dataset successfully loaded | total_records={}",
            len(dataset_list)
        )

        logger.info("Preparing news labeling tasks")
        tasks = [
            self.process_news(data)
            for data in dataset_list[:2]
        ]
        results = await self.parallel_processing(
            tasks,
            worker
        )
        logger.info(
            "Labeling tasks created | tasks={} | workers={}",
            len(tasks),
            worker
        )
        return results
    

async def main() -> None:
    """
    Entry point for running the news labeling pipeline.
    """
    #Env
    config = ConfigParser() 
    config.read("config.conf")

    llm = AsyncOpenAI(
        base_url=config["llm"]["base_url"],
        api_key=config["llm"]["api_key"]
    )

    nc = NewsClassification(llm=llm)

    download_config = DownloadConfig(
        token=config["hf"]["hf_token"]
    )
    pipeline = NewsLabelingPipeline(
        config=config,
        classifier=nc,
        download_config=download_config
    )
    results = await pipeline.run()
    srsly.write_json("output_label.json", results)

if __name__ == "__main__":
    asyncio.run(main())