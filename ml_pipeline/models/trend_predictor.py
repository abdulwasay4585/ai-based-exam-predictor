import numpy as np
from typing import Dict, List

class ExamTrendPredictor:
    """
    Predicts the likelihood of an exam topic appearing in the next assessment
    based on historical frequency analysis, momentum, and linear trend slopes.
    """
    
    def __init__(self, current_year: int):
        self.current_year = current_year
        self.target_year = current_year + 1

    def calculate_ema(self, frequencies: List[int], alpha: float = 0.5) -> float:
        """
        Calculate Exponential Moving Average (EMA).
        This gives more weight to recent years versus older years, reflecting changing syllabus priorities.
        """
        if not frequencies:
            return 0.0
            
        ema = frequencies[0]
        for f in frequencies[1:]:
            ema = alpha * f + (1 - alpha) * ema
        return ema

    def calculate_trend_slope(self, years: List[int], frequencies: List[int]) -> float:
        """
        Calculate simple linear regression slope.
        - Positive slope: Topic is trending upwards.
        - Negative slope: Topic is fading out of the syllabus.
        """
        if len(years) < 2:
            return 0.0
        
        # Fit a 1-degree polynomial (linear line) y = mx + c -> returns [m, c]
        slope, _ = np.polyfit(years, frequencies, 1)
        return slope
        
    def _sigmoid(self, x: float) -> float:
        """Standard sigmoid activation to normalize heuristic scores into a 0-1 probability."""
        return 1 / (1 + np.exp(-x))

    def predict_topic_probability(self, yearly_distribution: Dict[int, int]) -> float:
        """
        Combines base historical frequency, recent momentum (EMA), and overall trend (slope) 
        to output a normalized likelihood score for the targeted next year.
        
        Args:
            yearly_distribution (Dict[int, int]): A mapping of {Year: Absolute Frequency}
            
        Returns:
            float: Predicted probability (e.g., 0.82 representing 82% chance of appearing)
        """
        if not yearly_distribution:
            return 0.0

        # Sort chronologically to maintain time-series integrity
        sorted_years = sorted(yearly_distribution.keys())
        freqs = [yearly_distribution[y] for y in sorted_years]

        # 1. Base Density: Average historical appearances per year across the entire timeframe
        base_avg = np.mean(freqs)

        # 2. Momentum: Exponential Moving Average (alpha=0.6 strongly weights the last 2 years)
        momentum_ema = self.calculate_ema(freqs, alpha=0.6)

        # 3. Trend: Slope over the years
        trend_slope = self.calculate_trend_slope(sorted_years, freqs)

        # 4. Weighted combination logic:
        # Base avg guarantees a solid score for 'staple' questions that appear every single year.
        # Momentum captures immediate sudden shifts (e.g., a new professor changing topics).
        # Trend slope heavily rewards consistently growing topics.
        heuristic_score = (base_avg * 0.4) + (momentum_ema * 0.8) + (trend_slope * 1.5)
        
        # We subtract a bias threshold (e.g., 1.0) so that zero-activity topics fall closer to 0% 
        normalized_probability = self._sigmoid(heuristic_score - 1.0)
        
        return float(normalized_probability)


# Quick Evaluation Test
if __name__ == "__main__":
    predictor = ExamTrendPredictor(current_year=2023)
    
    # Simulate a syllabus topic: "Transformers / Attention Mechanisms"
    # Barely appeared in 2020, but has gained massive traction recently.
    transformers_history = {
        2020: 0,
        2021: 1,
        2022: 3,
        2023: 5
    }

    # Simulate a syllabus topic: "Expert Systems"
    # Legacy topic slowly dying out of the curriculum.
    expert_systems_history = {
        2020: 4,
        2021: 2,
        2022: 1,
        2023: 0
    }
    
    # Simulate a staple topic: "Backpropagation"
    # Appears essentially exactly 2 times every single year without fail.
    backprop_history = {
        2020: 2,
        2021: 2,
        2022: 2,
        2023: 2
    }

    print("--- 2024 Exam Topic Prediction Likelihood ---")
    print(f"Transformers:   {predictor.predict_topic_probability(transformers_history):.2%}")
    print(f"Expert Systems: {predictor.predict_topic_probability(expert_systems_history):.2%}")
    print(f"Backprop:       {predictor.predict_topic_probability(backprop_history):.2%}")
