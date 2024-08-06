# FSA trading Project 
A research that evaluates the effectiveness of on Machine Learning (ML) models in the Financial
Sentiment Analysis (FSA), comparing the ability to follow market trends (stock prices) 
based on sentiment predictions by the model, generated by two type of sources: news articles from 
known publishers and user-based contents on social platforms.
## Overview

Read the full paper here: https://biturl.top/mUNnQf

## Code Functionality
The code in this project performs the following tasks:

1. Data Preprocessing: All the data collected are available publicly, which is processed (tokenised, sanitised, etc.) and filtered specifically for training the model.
2. Sentiment Analysis: The code utilizes NLP techniques, and a pre-trained model FinBERT to determine the sentiment (positive, negative, or neutral) of financial text, logged its publishing time and compare this with historical prices data for certain group of tickers.
3. Evaluation benchmark: Evaluate the accuracy, precision, recall, f1-score of the model's predictions by comparing the model's sentiment predictions with the actual price changes for both data types. For each metrics, the 2 methods of calculations are used, as explained in evalute.py

The project was completed as a Final Year Project at Lancaster University. The process of training and running the model required a significant amount of computational power and time. As a result, this repository serves primarily to showcase the results and findings that supported the dissertation.
## Results
Here some interesting findings and analysis of the results:



## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use the code for academic, research, or commercial purposes.