from typing import *
from pydantic import BaseModel, Field

class NewsLabelFormat(BaseModel):
    label: Literal[
        "BENCANA_LINGKUNGAN",
        "EKONOMI_BISNIS",
        "HUKUM_KRIMINAL",
        "OLAHRAGA",
        "POLITIK_PEMERINTAHAN",
        "TEKNOLOGI_DIGITAL",
    ] = Field(
        description="Predicted category label for the processed news article."
    )