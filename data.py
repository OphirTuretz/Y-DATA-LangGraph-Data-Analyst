from typing import Any, Optional, Dict, List
from app.const import DATASET_NAME, DATASET_SPLIT_NAME
from datasets import load_dataset
import pandas as pd


class Dataset:
    singleton_dataset: Optional[pd.DataFrame] = None

    def __init__(
        self,
        filter_by: Dict[str, List[str]] = None,
    ):
        if filter_by is None:
            filter_by = {"category": [], "intent": []}

        self.filter_by: Dict[str, List[str]] = filter_by
        if Dataset.singleton_dataset is None:
            Dataset.singleton_dataset = self.load_dataset()

    def load_dataset(self) -> Optional[pd.DataFrame]:
        try:
            dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT_NAME)
            return dataset.to_pandas()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    @property
    def dataset(self) -> pd.DataFrame:
        if Dataset.singleton_dataset is None:
            raise ValueError("Dataset not loaded properly.")

        filtered_df = Dataset.singleton_dataset.copy()

        for column, values in self.filter_by.items():
            if values:
                filtered_df = filtered_df[filtered_df[column].isin(values)]

        return filtered_df

    # For checkpointing (serialization)
    def __getstate__(self) -> Dict[str, Any]:
        return {
            "filter_by": self.filter_by,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.filter_by = state["filter_by"]
        if Dataset.singleton_dataset is None:
            Dataset.singleton_dataset = self.load_dataset()

    def __reduce__(self):
        # Return a tuple of (callable, args) to reconstruct the object
        return (self.__class__, (self.filter_by,))

    def get_possible_intents(self) -> List[str]:
        """
        Get a list of unique intents from the DataFrame.
        Returns:
            List[str]: A list of unique intent names.
        """
        return self.dataset["intent"].unique().tolist()

    def get_possible_categories(self) -> List[str]:
        """
        Get a list of unique categories from the DataFrame.
        Returns:
            List[str]: A list of unique category names.
        """
        return self.dataset["category"].unique().tolist()

    def set_filter(
        self, category_names: List[str] = None, intent_names: List[str] = None
    ) -> "Dataset":
        """Set filters for categories and/or intents"""
        if category_names is not None:
            self.filter_by["category"] = category_names
        if intent_names is not None:
            self.filter_by["intent"] = intent_names
        return self

    def clear_filters(self) -> "Dataset":
        """Reset all filters"""
        self.filter_by = {"category": [], "intent": []}
        return self

    def select_semantic_intent(self, intent_names: List[str]) -> "Dataset":
        """
        Select rows from the DataFrame where the 'intent' column matches any of the provided intent names.
        Args:
            intent_names (List[str]): List of intent names to filter by.
        Returns:
            self: The Dataset object with the filtered DataFrame.
        """
        return self.set_filter(intent_names=intent_names)

    def select_semantic_category(self, category_names: List[str]) -> "Dataset":
        """
        Select rows from the DataFrame where the 'category' column matches any of the provided category names.
        Args:
            category_names (List[str]): List of category names to filter by.
        Returns:
            self: The Dataset object with the filtered DataFrame.
        """
        return self.set_filter(category_names=category_names)

    def count_rows(self) -> int:
        """
        Get the number of rows in the DataFrame.
        Returns:
            int: The number of rows in the DataFrame.
        """
        return len(self.dataset)

    def count_category(self, category: str) -> int:
        """
        Count the number of rows in the DataFrame that match a specific category.
        Args:
            category (str): The category to count in the DataFrame.
        Returns:
            int: The count of rows matching the specified category.
        """
        return self.dataset[self.dataset["category"] == category].shape[0]

    def count_intent(self, intent: str) -> int:
        """
        Count the number of rows in the DataFrame that match a specific intent.
        Args:
            intent (str): The intent to count in the DataFrame.
        Returns:
            int: The count of rows matching the specified intent.
        """
        return self.dataset[self.dataset["intent"] == intent].shape[0]

    def show_examples(self, n: int) -> pd.DataFrame:
        """
        Show a sample of n examples from the DataFrame.
        Args:
            n (int): The number of examples to show.
        Returns:
            pd.DataFrame: A DataFrame containing n random samples from the dataset.
        """
        return self.dataset.sample(n)
