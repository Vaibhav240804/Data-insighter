from lida import Manager
import pandas as pd

# Load your data (replace with your data source)
data = pd.read_csv("dummy.csv")

# Initialize LIDA manager
manager = Manager()
# set openAI API key
# manager.set_openai_key("your-openai")
# Summarize data and define visualization goal
summary = manager.summarize(data)
goal = {"type": "chart", "x": "Remaining Stock", "y": "Aging range : 31-40"}

# Generate visualization
charts = manager.visualize(summary, goal, library="matplotlib")

# Display or save the chart
charts[0].show()  # Display in notebook
charts[0].savefig("chart.png")  # Save as image
