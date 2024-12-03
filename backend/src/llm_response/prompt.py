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
        elif self.task_type == "report":
            return self._report_prompt()
        elif self.task_type == "code":
            return self._code_prompt()
        elif self.task_type == "table":
            return self._table_prompt()
        elif self.task_type == "explanation":
            return self._explanation_prompt()
        elif self.task_type == "qa":
            return self._qa_prompt()
        elif self.task_type == "simulation":
            return self._simulation_prompt()
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
    
    def _report_prompt(self):
        return (
            f"User's query: {self.query}\nRelevant information:\n{self.context}\n"
            "Generate a structured report with the following sections:\n"
            "- Key Highlights\n"
            "- Analysis\n"
            "- Recommendations\n"
            "The report should be concise and professional."
        )

    def _code_prompt(self):
        return (
            f"User's query: {self.query}\nContext:\n{self.context}\n"
            "Generate well-documented and clean code in the required language."
        )

    def _table_prompt(self):
        return (
            f"User's query: {self.query}\nContext:\n{self.context}\n"
            "Provide the data in a tabular JSON format with the following columns:\n"
            "- Column headers\n"
            "- Rows with data\n"
            "Ensure data integrity and consistency."
        )

    def _explanation_prompt(self):
        return (
            f"User's query: {self.query}\nContext:\n{self.context}\n"
            "Provide a clear and simple explanation for the given data or concept."
        )

    def _qa_prompt(self):
        return (
            f"Context:\n{self.context}\n"
            "Based on the provided information, generate questions and answers that are accurate and concise."
        )

    def _simulation_prompt(self):
        return (
            f"User's query: {self.query}\nContext:\n{self.context}\n"
            "Simulate a conversation for the given scenario. Provide responses and queries in a natural dialogue format."
        )

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
