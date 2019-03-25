# Machine Learning Engineer Nanodegree
## Capstone Project
### Joe Importico March 25, 2019

***
### Forecasting the Cross-Section of Asset Returns

***

Reproducing this project requires the modules listed below:

- Python 3.5 
- Keras
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pandas_datareader

To save the reviewer time, I've stored the datasets used for this analysis to the .csv files listed below. I recommend reading these files into the `CapstoneProject_Importico.ipynb` as some of the routines are computationally expensive. In addition to the files, I'll detail the sections they fall within. 

- assets.csv: The initial dataset warehousing a list of tickers that was used as the foundation for constructing the custom dataset used for this project. I recommend skipping this file.

- input_data.csv: Contains OHLCV data for a series of tickers and ETF's over time.

- famafrenchformatted.csv: A re-formatted version of the 5 factor Fama & French model.

- all_factors.csv: This is the final dataset used in the Exploratory Data Analysis & Machine Learning sections.


In addition to the modules listed above, the `all_factors.csv` file must be used to carry out all routines after the **Exploratory Data Analysis** section. This file is too large to store in Github and will be provided separately.
