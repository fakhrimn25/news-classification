import asyncio
import traceback
from typing import *
from pydantic import BaseModel
from tqdm.asyncio import tqdm_asyncio

class BaseAgent(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    async def parallel_processing(self, tasks: list, num_workers: int):
        """
        Executes tasks in parallel while preserving their original order.
        """
        semaphore = asyncio.Semaphore(num_workers)

        async def worker(task):
            async with semaphore:
                try:
                    return await task
                except Exception as e:
                    return (
                    f"error: [{type(e).__name__}] {str(e)} | "
                    f"Traceback: {traceback.format_exc().strip()}"
                )

        wrapped_tasks = [worker(task) for task in tasks]

        results = await tqdm_asyncio.gather(
            *wrapped_tasks,
            desc="Processing tasks",
            total=len(tasks)
        )

        return results