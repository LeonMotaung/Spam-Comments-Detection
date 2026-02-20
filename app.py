import io
import base64
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Ensure plot is generated in background (no GUI)
import matplotlib.pyplot as plt
from flask import Flask, render_template_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

app = Flask(__name__)

# Function to train models and get their accuracies
def get_accuracies():
    data = pd.read_csv("Youtube01-Psy.csv")
    data["CLASS"] = data["CLASS"].map({0: "Not Spam", 1: "Spam Comment"})

    x = np.array(data["CONTENT"])
    y = np.array(data["CLASS"])

    cv = CountVectorizer()
    x = cv.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train Naive Bayes
    nb = BernoulliNB()
    nb.fit(xtrain, ytrain)
    nb_acc = nb.score(xtest, ytest)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(xtrain, ytrain)
    rf_acc = rf.score(xtest, ytest)

    # Train Ensemble
    ensemble = VotingClassifier(estimators=[('rf', rf), ('nb', nb)], voting='hard')
    ensemble.fit(xtrain, ytrain)
    ens_acc = ensemble.score(xtest, ytest)
    
    return {'Naive Bayes': nb_acc, 'Random Forest': rf_acc, 'Ensemble': ens_acc}

@app.route('/')
def home():
    accuracies = get_accuracies()
    
    # Generate the bar plot
    models = list(accuracies.keys())
    scores = list(accuracies.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Accuracy Score')
    ax.set_title('Spam Detection Models Comparison', fontsize=14, fontweight='bold')
    
    # Display the score above the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f"{round(yval * 100, 2)}%", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Convert the plot to a PNG image in memory buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    
    # Encode the PNG image to base64 string
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    # HTML template to render the page and image
    HTML_TEMPLATE = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spam Detection Accuracy Visualizer</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                text-align: center; 
                background-color: #f4f4f9; 
                color: #333; 
                margin: 40px; 
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background-color: #fff;
                padding: 30px;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            img { 
                max-width: 100%;
                border-radius: 8px; 
                margin-top: 20px; 
            }
            h1 { color: #2c3e50; }
            p { font-size: 1.1em; color: #555; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Spam Classification Performance</h1>
            <p>Comparing the accuracy of Naive Bayes, Random Forest, and our Ensemble model.</p>
            
            <!-- Plot the base64 encoded image directly -->
            <img src="data:image/png;base64,{{ plot_url }}" alt="Model Accuracy Chart">
        </div>
    </body>
    </html>
    """
    return render_template_string(HTML_TEMPLATE, plot_url=plot_url)

if __name__ == '__main__':
    # Run server on port 5058
    app.run(host='127.0.0.1', port=5058, debug=True)
