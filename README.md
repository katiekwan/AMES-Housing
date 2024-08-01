# AMES-Housing
This is a Machine Learning, predictive and descriptive modeling study of housing data from AMES Iowa from January 2006 - June 2010.

Business Case: Develop a model with minimal, easy to use features, that delivers high accuracy. Users of this model would be real estate agents and prospective buyers/sellers.
- This includes descriptive modelling to analyze neighborhoods, macroeconomic factors, and the impact of the 2008 housing crisis on the AMES Market. 
- Information, Analysis and Visuals have been summarized in the AMES Presentation (pdf).

For detailed code work, please view sequentially:

1. EDA_1 Prelim EDA: Loading and Cleaning
- Mapping to Plotly Map
- Reconciling Missing Data
- Create a new ‘housing_cleaned.csv’

2. EDA_2 X & Y Visualizations 
- Y Transformations (As is vs. Log vs. Box Cox)
- Continuous Features: Plot Feature Variance
- Categorical Features: Plot Feature Count

3. EDA_3 Correlations and Multicollinearity
- Continuous Features: Scatterplots, Heatmaps, Top 15 Correlated Features, R2 for Multicolinearity.
- Categorical Features: F Statistic, PValue and Boxplots

4. Modelling_1 Linear and Penalized Models
- Using Linear Regresion, Lasso, Ridge, Elastic Net and LassoCV to analyze saturated models.
- Using recursive feature elimination to narrow feature set.
- Feature engineering. 
- Output: An Easy Fatures set, where 9 easy to calculate features devlier an adjusted r squared of 87.4%

5. Modelling_2 Tree Based Models
- Use Decision Tree, Random Forest, and Gradient Boosting to model saturated and abbreviated features sets.
- Tuning models.
- Visualizing features, trees, and partial dependence plots (PDPs).

6. Descriptive Modelling
- Deep dive into key influencers of housing Salesprice, outside of the four walls. 
- Advanced Analysis of Neighborhoods, Volume v Price Correlations, and mortgage rates.

7. ** Models.py (accessory code)
- Functional Programming to Load Data, Encode and Split Data, Preprocess and Model Data using a range of models. Output includes, acurracy scores, sorted feature set, and regresor objects. 

