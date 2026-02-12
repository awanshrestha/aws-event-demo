import gradio as gr

import matplotlib

# REQUIRED: Use a non-interactive backend for AWS/Server deployment
# Otherwise it will crash on the cloud!
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def generate_chart(labels_text, values_text, chart_type):
    try:
        # 1. Process the Inputs
        # Split the text by commas and clean up whitespace
        labels = [l.strip() for l in labels_text.split(',')]
        values = [float(v.strip()) for v in values_text.split(',')]

        # Check for errors (mismatched data)
        if len(labels) != len(values):
            raise ValueError(f"You have {len(labels)} labels but {len(values)} numbers!")

        # 2. Create the Chart
        fig = plt.figure(figsize=(8, 6))
        
        if chart_type == "Bar Chart":
            plt.bar(labels, values, color=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'])
            plt.xlabel("Categories")
            plt.ylabel("Values")
        elif chart_type == "Pie Chart":
            plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'])
        elif chart_type == "Line Chart":
            plt.plot(labels, values, marker='o', linestyle='-', color='#36A2EB')
            plt.grid(True)

        plt.title(f"My Awesome {chart_type}")
        
        # 3. Return the figure object
        return fig
        
    except Exception as e:
        # If something goes wrong, return a blank figure with the error
        fig = plt.figure()
        plt.text(0.5, 0.5, str(e), ha='center', va='center', color='red')
        return fig

# 4. Build the UI
with gr.Blocks() as demo:
    gr.Markdown("# 📊 The 30-Minute Chart Builder")
    gr.Markdown("Enter data below to instantly generate a visualization.")
    
    with gr.Row():
        with gr.Column():
            # Inputs
            txt_labels = gr.Textbox(label="Categories (comma separated)", placeholder="Rent, Food, Gym, Netflix", value="Rent, Food, Gym, Netflix")
            txt_values = gr.Textbox(label="Values (comma separated)", placeholder="1000, 400, 50, 20", value="1000, 400, 50, 20")
            radio_type = gr.Radio(["Bar Chart", "Pie Chart", "Line Chart"], label="Chart Type", value="Bar Chart")
            btn = gr.Button("Generate Chart 🚀", variant="primary")
        
        with gr.Column():
            # Output
            plot_output = gr.Plot(label="Your Chart")

    btn.click(generate_chart, inputs=[txt_labels, txt_values, radio_type], outputs=plot_output)

# 5. Launch 
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)