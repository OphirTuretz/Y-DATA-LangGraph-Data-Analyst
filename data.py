from app.const import DATASET_NAME, DATASET_SPLIT_NAME
from datasets import load_dataset
import pandas as pd
from typing import List


class Dataset:
    def __init__(self, dataset_name=DATASET_NAME, split=DATASET_SPLIT_NAME):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = self.load_dataset()

    def load_dataset(self):
        try:
            dataset = load_dataset(self.dataset_name, split=self.split)
            return dataset.to_pandas()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

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

    def select_semantic_intent(self, intent_names: List[str]) -> "Dataset":
        """
        Select rows from the DataFrame where the 'intent' column matches any of the provided intent names.
        Args:
            intent_names (List[str]): List of intent names to filter by.
        Returns:
            self: The Dataset object with the filtered DataFrame.
        """
        self.dataset = self.dataset[self.dataset["intent"].isin(intent_names)]
        return self

    def select_semantic_category(self, category_names: List[str]) -> "Dataset":
        """
        Select rows from the DataFrame where the 'category' column matches any of the provided category names.
        Args:
            category_names (List[str]): List of category names to filter by.
        Returns:
            self: The Dataset object with the filtered DataFrame.
        """
        self.dataset = self.dataset[self.dataset["category"].isin(category_names)]
        return self

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
