class Prompt:
    """
    A class for generating prompts for different tasks in a consistent and reusable manner.
    """

    def __init__(self, query="", context="", task_type="default"):
        self.query = query
        self.context = context
        self.task_type = task_type

    def generate_prompt(self):
        """
        Generates a prompt based on the task type.
        Returns:
            str: The generated prompt.
        """
        if self.task_type == "query":
            return self._query_prompt()
        elif self.task_type == "summarization":
            return self._summarization_prompt()
        elif self.task_type == "chart":
            return self._chart_prompt()
        elif self.task_type == "custom":
            return self._custom_prompt()
        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

    def _query_prompt(self):
        """
        Generates a prompt for querying based on the query and context.
        """
        return f"User's query: {self.query}\nRelevant context:\n{self.context}"

    def _summarization_prompt(self):
        """
        Generates a prompt for summarization.
        """
        return f"Summarize the following text:\n\n{self.context}"

    def _chart_prompt(self):
        """
        Generates a prompt for chart generation.
        """
        return (
            f"User's query: {self.query}\nRelevant information:\n{self.context}\n"
            "Based on the trend, suggest the best chart type and provide the data in JSON format suitable for creating the chart. "
            "The JSON should include:\n"
            "- 'chartType': The type of chart (e.g., 'Line Chart', 'Bar Chart').\n"
            "- 'chartLabel': A label for the chart (e.g., 'Income and Expense Trend').\n"
            "- 'data': An array of objects, where each object includes:\n"
            "    - 'category' or 'x': The label for the x-axis (e.g., 'Time Period').\n"
            "    - 'series': An array of key-value pairs for y-axis values (e.g., 'income', 'expense').\n"
            "Provide the JSON in a clean, consistent format without additional text or explanations."
        )


    def _custom_prompt(self):
        """
        Generates a custom prompt by directly using the context.
        """
        return self.context

    @staticmethod
    def example_prompts():
        """
        Provides example prompts for different task types.
        """
        return {
            "query": "What is the current market trend in the context provided?",
            "summarization": "Summarize this document in 100 words or less.",
            "chart": "Provide a line chart of sales trends over the past 12 months.",
            "custom": "Write a poem about AI and creativity."
        }
