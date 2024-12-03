import re
import json
import matplotlib.pyplot as plt

def parse_response_and_generate_chart(llm_response):
    """
    Parses the LLM response to extract JSON data and chart type, then generates a chart.
    """
    try:
        # Extract JSON part using regex
        #print(llm_response)
        json_match = re.search(r"```json(.*?)```", llm_response, re.DOTALL)
        
        print('1')
        if not json_match:
            print("No JSON data found in the response.")
            return None
        
        # Clean up invalid JSON keys/values
        raw_json = json_match.group(1).strip()
         # Replace invalid chart type value
        print('2')
        response_json = json.loads(raw_json)
        print('3')
        # Extract chart type from the corrected JSON
        chart_type = response_json.get("chartType", "Line Chart").lower()
        print(chart_type)
        chart_title = response_json.get("chartTitle", "Generated Chart")
        print(chart_title)
        x_axis_title = response_json.get("xAxisTitle", "X-Axis")
        print(x_axis_title)
        y_axis_title = response_json.get("yAxisTitle", "Y-Axis")
        print(y_axis_title)
        series_data = response_json.get("data", [])
        print(series_data)

        # Generate chart
        return generate_chart(chart_type, chart_title, x_axis_title, y_axis_title, series_data)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"Error parsing response or generating chart: {e}")
        return None


def generate_chart(chart_type, chart_title, x_axis_title, y_axis_title, series_data):
    """
    Generates a chart based on the type and data provided.
    """
    try:
        plt.figure(figsize=(10, 6))

        if chart_type == "line chart":
            for series in series_data:
                data_points = series.get("data", [])
                x_values = [dp["x"] for dp in data_points]
                y_values = [dp["y"] for dp in data_points]
                plt.plot(
                    x_values,
                    y_values,
                    label=series["label"],  # Fixed: Use 'label' instead of 'name'
                    marker="o",
                    linewidth=2
                )
        elif chart_type == "bar_chart":
            x = range(len(series_data[0]["data"]))
            width = 0.2
            for i, series in enumerate(series_data):
                data_points = series.get("data", [])
                y_values = [dp["y"] for dp in data_points]
                x_offset = [p + i * width for p in x]
                plt.bar(
                    x_offset,
                    y_values,
                    width=width,
                    label=series["label"]  # Fixed: Use 'label' instead of 'name'
                )
            plt.xticks([p + width for p in x], [dp["x"] for dp in series_data[0]["data"]])
        else:
            print(f"Unsupported chart type: {chart_type}")
            return None

        # Customize chart
        plt.title(chart_title, fontsize=16)
        plt.xlabel(x_axis_title, fontsize=12)
        plt.ylabel(y_axis_title, fontsize=12)
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        # Save chart
        chart_path = "generated_chart.png"
        plt.savefig(chart_path, bbox_inches="tight")
        plt.close()

        print(f"Chart saved as {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None

    
response = '''
LLM response: Based on the trend, I suggest using a Line Chart to visualize the income and expense trend. A Line Chart is suitable for showing trends over time and can effectively display the fluctuations in income and expenses.

Here is the data in JSON format suitable for creating the chart:

```json
{
  "chartType": "Line Chart",
  "data": [
    {
      "label": "Net Income",
      "data": [
        {"x": "Q1 2023", "y": 167},
        {"x": "Q2 2023", "y": 167},
        {"x": "Q3 2023", "y": 4217},
        {"x": "Q4 2023", "y": 4217},
        {"x": "Q1 2024", "y": 167},
        {"x": "Q2 2024", "y": 167},
        {"x": "Q3 2024", "y": 4821},
        {"x": "Q4 2024", "y": 7031}
      ]
    },
    {
      "label": "Total Revenues",
      "data": [
        {"x": "Q1 2023", "y": 23350},
        {"x": "Q2 2023", "y": 23350},
        {"x": "Q3 2023", "y": 71606},
        {"x": "Q4 2023", "y": 71606},
        {"x": "Q1 2024", "y": 25182},
        {"x": "Q2 2024", "y": 25182},
        {"x": "Q3 2024", "y": 71983},
        {"x": "Q4 2024", "y": 71983}
      ]
    },
    {
      "label": "Total Cost of Revenues",
      "data": [
        {"x": "Q1 2023", "y": 19172},
        {"x": "Q2 2023", "y": 19172},
        {"x": "Q3 2023", "y": 58384},
        {"x": "Q4 2023", "y": 58384},
        {"x": "Q1 2024", "y": 20185},
        {"x": "Q2 2024", "y": 20185},
        {"x": "Q3 2024", "y": 58712},
        {"x": "Q4 2024", "y": 58712}
      ]
    }
  ]
}
```

This data includes three lines: Net Income, Total Revenues, and Total Cost of Revenues. Each line has data points for each quarter of 2023 and 2024. The x-axis represents the quarter, and the y-axis represents the value of each data point.


'''

chart_path = parse_response_and_generate_chart(response)
print(chart_path)