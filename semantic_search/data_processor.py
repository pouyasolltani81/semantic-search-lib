import pandas as pd

class DataProcessor:
    def __init__(self, csv_path):
        try:
            self.df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file at {csv_path}: {e}") from e

    def format_data(self, template):
        if self.df.empty:
            raise ValueError("CSV file is empty.")
        try:
            # Validate template against the first row
            sample = self.df.iloc[0].to_dict()
            template.format(**sample)
        except KeyError as e:
            raise ValueError(f"Template references a missing column: {e}") from e
        except Exception as e:
            raise ValueError("Error processing the template formatting.") from e
        try:
            return [template.format(**row) for _, row in self.df.iterrows()]
        except Exception as e:
            raise RuntimeError("Error during formatting data with the template.") from e
