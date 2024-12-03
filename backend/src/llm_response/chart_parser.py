import json
import matplotlib.pyplot as plt
import re
import os
import uuid
from datetime import datetime
from config import CHARTS_DIR


def infer_key_roles(data_point):
    """
    Dynamically infer key roles (x, y, label) based on data patterns.
    """
    roles = {"x": None, "y": None, "label": None}
    for key, value in data_point.items():
        if isinstance(value, (int, float)) and not roles["y"]:
            roles["y"] = key  # Numeric values are likely 'y'
        elif isinstance(value, str) and any(char.isalpha() for char in value) and not roles["x"]:
            roles["x"] = key  # String values are likely 'x'
        else:
            roles["label"] = roles["label"] or key  # Descriptive text is likely 'label'
    return roles


def normalize_chart_type(chart_type):
    """
    Normalize the chart type to a standard format.
    """
    chart_type_mapping = {
        "line": ["linechart", "line chart", "Line Chart", "LineChart", "line"],
        "bar": ["barchart", "bar chart", "Bar Chart", "BarChart", "bar"],
        "pie": ["piechart", "pie chart", "Pie Chart", "PieChart", "pie"],
        "scatter": ["scatterplot", "scatter plot", "Scatter Plot", "ScatterPlot", "scatter"],
    }

    for standard_type, variations in chart_type_mapping.items():
        if chart_type.lower() in [v.lower() for v in variations]:
            return standard_type
    return None  # Unsupported chart type


def normalize_series_data(raw_data):
    """
    Normalize series data dynamically by inferring x and y values.
    """
    normalized_series = []
    for item in raw_data:
        roles = infer_key_roles(item)
        x_value = item.get(roles["x"], "Unknown")
        for metric in item.get("series", []):
            metric_name = metric.get("name", "Metric")
            metric_value = metric.get("value", 0)

            # Find or create a series for this metric
            series = next((s for s in normalized_series if s["label"] == metric_name), None)
            if not series:
                series = {"label": metric_name, "data": []}
                normalized_series.append(series)

            # Append the data point
            series["data"].append({"x": x_value, "y": metric_value})
    return normalized_series


def parse_response_and_generate_chart(llm_response):
    """
    Parse the LLM response to extract JSON data and generate a chart.
    """
    try:
        # Extract JSON part
        print(llm_response)
        # Extract JSON part
        if "```json" in llm_response:
            json_match = re.search(r"```json(.*?)```", llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON data found within triple backticks.")
            raw_json = json_match.group(1).strip()
        else:
            json_match = re.search(r"(\{.*\})", llm_response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON data found in plain response.")
            raw_json = json_match.group(1).strip()
        # Sanitize the JSON
        raw_json = raw_json.replace("\n", "").replace("\r", "").strip()

        # Load JSON data
        response_json = json.loads(raw_json)


        # Normalize chart type
        raw_chart_type = response_json.get("chartType", "line")
        chart_type = normalize_chart_type(raw_chart_type)
        if not chart_type:
            raise ValueError(f"Unsupported chart type: {raw_chart_type}")

        chart_label = response_json.get("chartLabel", "Generated Chart")
        raw_data = response_json.get("data", [])

        # Normalize series data
        normalized_series = normalize_series_data(raw_data)

        # Generate the chart
        return generate_chart(chart_type, chart_label, "Time Period", "Values", normalized_series)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"Error parsing response or generating chart: {e}")
    return None


def generate_chart(chart_type, chart_title, x_axis_title, y_axis_title, series_data):
    """
    Generate and save a chart based on the type and provided data.
    """
    try:
        plt.figure(figsize=(12, 8))

        if chart_type == "line":
            for series in series_data:
                x_values = [dp["x"] for dp in series["data"]]
                y_values = [dp["y"] for dp in series["data"]]
                plt.plot(x_values, y_values, label=series["label"], marker="o", linewidth=2)
        elif chart_type == "bar":
            x = range(len(series_data[0]["data"]))
            width = 0.2
            for i, series in enumerate(series_data):
                y_values = [dp["y"] for dp in series["data"]]
                x_offset = [p + i * width for p in x]
                plt.bar(x_offset, y_values, width=width, label=series["label"])
            plt.xticks([p + width for p in x], [dp["x"] for dp in series_data[0]["data"]])
        elif chart_type == "pie":
            labels = [dp["x"] for dp in series_data[0]["data"]]
            values = [dp["y"] for dp in series_data[0]["data"]]
            plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
        elif chart_type == "scatter":
            for series in series_data:
                x_values = [dp["x"] for dp in series["data"]]
                y_values = [dp["y"] for dp in series["data"]]
                plt.scatter(x_values, y_values, label=series["label"])
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

        # Customize chart
        plt.title(chart_title, fontsize=16)
        plt.xlabel(x_axis_title, fontsize=12)
        plt.ylabel(y_axis_title, fontsize=12)
        if chart_type != "pie":
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)

        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        chart_name = f'generated_chart_{chart_type}_{timestamp}_{unique_id}.png'
        chart_path = os.path.join(CHARTS_DIR, chart_name)
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        print(f"Chart saved as {chart_name}")
        return chart_name, chart_type
    except Exception as e:
        print(f"Error generating chart: {e}")
    return None
