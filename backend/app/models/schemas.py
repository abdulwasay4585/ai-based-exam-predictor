from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class YearOccurrence(BaseModel):
    """Tracks how often a question/topic appeared in specific years and exam types."""
    year: int = Field(..., example=2023, description="The year the question appeared.")
    frequency: int = Field(default=1, example=2, description="How many times it appeared in this exact year.")
    paper_reference: Optional[str] = Field(None, example="Final Exam / Midterm", description="Meta info about the past paper.")

class Topic(BaseModel):
    """Latent topic cluster mapped from the NLP LDA model."""
    topic_id: str = Field(..., example="Topic 3", description="Unique identifier for the latent cluster topic.")
    topic_keywords: List[str] = Field(default_factory=list, description="Top N keywords defining the topic.")
    confidence_score: float = Field(..., example=0.85, description="Probability score from the ML model matching this topic.")

class QuestionRecord(BaseModel):
    """
    Core data structure to store individual extracted exam questions, 
    their mapped AI topics, and historical frequency across years.
    """
    question_id: str = Field(..., description="Unique generated UUID for the question.")
    
    # Text Payload
    raw_text: str = Field(..., description="The original unmodified text scraped from the PDF.")
    cleaned_text: str = Field(..., description="The lemmatized and normalized textual version for ML consumption.")
    
    # ML Analysis Mapping
    associated_topics: List[Topic] = Field(default_factory=list, description="Latent topics recognized in this question.")
    
    # Historical Analytics
    occurrences: List[YearOccurrence] = Field(default_factory=list, description="Historical temporal spread of this question.")
    total_historical_frequency: int = Field(default=0, description="Total absolute cumulative frequency across all years.")
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this record was ingested.")

class TopicPredictionProfile(BaseModel):
    """
    Analytical View used to predict likelihood of topics emerging in future years.
    """
    topic_id: str
    defining_keywords: List[str]
    yearly_distribution: dict[int, int] = Field(..., description="Dictionary mapping {Year: Frequency}")
    predicted_future_probability: float = Field(default=0.0, description="Inference output: probability of appearing in the upcoming exam.")
