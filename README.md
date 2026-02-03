# Portfolio

### Education
- MA in Quantitative Methods in the Social Sciences (Concentration: Data Science), Columbia University, Graduate School of Arts and Sciences (GSAS)
- MPA in International Affairs (Specializations: United Nations Studies, International Conflict Resolution), Columbia University, School of International & Public Affairs (SIPA)
- BA in Political Science (Concentration: International Affairs), Northwestern University

### Work Experience
- Founder, CEO, and Lead Geospatial Machine Learning Engineer @ Crisis Forecast - Las Vegas, NV - 2023 - Present
- Data Scientist Intern @ Volkswagen Group of America (VGoA) - Belmont, CA - 2022-2023
- SIPA Capstone Project Consultant @ International Peace Institute (IPI) - New York, NY - 2020
- Protection of Civilians (PoC) Team Intern, Division of Policy, Evaluation, and Training @ United Nations Department of Peace Operations (UN DPO) - New York, NY - 2019
- Refugee Status Determination (RSD) Intern @ United Nations High Commissioner for Refugees (UNHCR) - Cairo, Egypt - 2016

### Conflict Data Science Projects
- GIS & Spatial Analysis:
    - **Title:** ["Predicting Attacks on Civilians in the Eastern Democratic Republic of Congo" (5 Pages)](./ML-Geospatial%20Violence%20Forecast-E%20DRC/part_1-spatial_forecast-east_drc-attacks_on_civilians.html)
        - Focus: Geospatial Machine Learning
        - Technology Used: R Markdown, QGIS, mlr3, spatially aware cross-validation, XGBoost, Random Forest, Support Vector Machines, Principal Component Analysis, multi-objective tuning, hyperband, nsga2R, sf, raster, terra, spdep, blockCV, ggplot2, rayshader, pROC, PRROC, caret, data.tables, future (parallel processing)
        - Contents: Geospatial machine learning model creation and performance comparison, feature engineering, creating dynamically weighted and asymmetrically penalized metrics for highly skewed data, spatial weight matrix creation, feature selection, feature importance, creating maps with fishnet grid cell observations, kernel density estimation heatmaps, Global and Local Moran's I calculation, variogram analysis for determining the range of spatial autocorrelation, and data visualization of 2D and 3D maps, bar charts, histograms, risk category distribution charts, ROC Curves, and Precision-Recall Curves.
    - **Title:** ["Examining Geospatial Covariate Relationships with Civilian Killings in South Sudan’s Civil War"](./GIS/GIS_covariate_relationships-killings-south_sudan.html)
        - Focus: Geospatial Statistics
        - Technology Used: QGIS, R, GeoDa
        - Contents: Geospatial interpolation, hot spot analysis, Moran's I calculation, Moran's I residual analysis, Lagrange Multiplier (LM) and Robust LM lag and error diagnostics, and Spatial Durbin models.
- Data Visualization Storytelling:
    - **Title: "UNAMID: Did the UN’s Withdrawal from Darfur Lead to More Violence against Civilians?" (9 Pages)**
    - [Version showing all code](./Data%20Viz-Darfur%20Violence%20as%20UN%20Left/visual_1-darfur_violence-code_included.html)
    - [Version not showing code](./Data%20Viz-Darfur%20Violence%20as%20UN%20Left/visual_1-darfur_violence.html)
        - Focus: GIS Mapping
        - Technology Used: R Markdown, ggmap, tmap, sp, sf, rgdal, stadia/stamen maps, plotly, dplyr, ggplot2, ggthemes, ggpubr, stringr, scales, and kableExtra
        - Contents: Interactive and static charts, GIS maps, heat map tables, density maps, and union buffers and using statistics within them
- Graph Network Visualization:
    - **Title:** ["Eastern DRC Conflict Network Graph"](./Network%20Analysis/eastern_drc_conflict_network_graph.html)
        - Focus: Graph Networks
        - Technology Used: R Markdown, sf, tidyr, dplyr, htmlwidgets, igraph, and visNetwork    
        - Contents: Interactive undirected network graph, with nodes weighted by eigenvector centrality, edges weighted by total conflict episodes between node dyads, community clusters determined by the Walktrap graph cluster algorithm, edge type distinguished by color, and pop-up displays of edge attributes as users hover over edges.
- Natural Language Processing (NLP):
    - **Title:** ["Using News Articles to Predict Political Violence in Nigeria"](./NLP/Using_Nigerian_News-based_ML_Models_to_Predict_Political_Violence.html)
        - Focus: NLP for Time-Series Forecasting
        - Technology Used: Python, TF-IDF word transformation, NLTK library, Scikit-Learn machine learning models, Scikit-Learn's TimeSeriesSplit, Augmented Dickey-Fuller Test, first-differencing, lags
        - Contents: Converting news articles by publishing date into time-series machine learning forecasting models. Performance comparison between Ridge, Lasso, Random Forest, and XGBoost regression models
    - **Title: "LDA Topic Modeling & VADER Sentiment Analysis for Political News Articles on Nigeria"**
    - [Primary document (Python)](./NLP/Nigeria_News_LDA_&_Sentiment_Analysis.html)
    - [Visualization for the project (R)](./NLP/Nigeria_News_Sentiment_Analysis-Viz-Created_in_R.html)
        - Focus: Topic Modeling & Sentiment Analysis
        - Technology Used: Python, R Markdown, Excel, NLTK for stopwords, PorterStemmer, and PunktSentenceTokenizer, gensim library for CoherenceModel, LdaModel, and corpora, Jaccard similarity, vaderSentiment library, itertools, ggplot2
        - Contents: Text data cleaning, Latent Dirichlet Allocation (LDA) topic modeling of Nigerian news article text, VADER (Valence Aware Dictionary for Sentiment Reasoning) sentiment analysis scores for articles containing specific political words, compared across quarters of the year.

### Non-Conflict Data Science Projects
- Statistical Analysis:
    - **Title:** ["Analyzing Promotional Strategy Effectiveness: A Mixed Effects Approach to Observational Retail Data"](./Econometrics/mixed_effects_study_portfolio.html)
        - Focus: Investigating the association between promotional strategies and retail sales using hierarchical modeling techniques
        - Technology Used: Python, statsmodels (MixedLM), pandas, NumPy, SciPy, seaborn, matplotlib
        - Contents: An observational study analyzing 548 sales observations across 137 retail locations and 10 markets. Employed linear mixed effects models with nested random effects to account for hierarchical data structure (locations within markets). Identified bimodal distribution in sales, stratified analysis by market segment, and conducted pairwise comparisons with Bonferroni correction. Found Promotion 1 fairly consistently associated with higher sales compared to alternatives across market segments.
- Large Language Models (LLMs):
    - **Title:** ["RAG for LLMs"](./LLMs/RAG/NVIDIA_NIM_RAG_Demo/RAG_Demo.html)
        - Focus: Enhancing LLM Performance and Utility for Internal Organizational Tasks
        - Technology Used: Python, NVIDIA NIM, LangChain, Streamlit, IPython.display, NVIDIA Vector Embeddings, FAISS Vector Store, Meta Llama 3.1-8B LLM
        - Contents: A proof-of-concept for using Retrieval Augmented Generation (RAG) with an LLM to search PDF documents on a computer, providing context for chatbot queries.
- Web Scraping:
    - **Title:** ["Web Scraping Using BeautifulSoup"](./Web_Scraping/Web_Scraping_Using_BeautifulSoup.html)
        - Focus: Illustrating how to scrape text and tables anonymously from embedded URL links located within URLs placed in a user's predefined list.
        - Technology Used: Python, BeautifulSoup, TOR, Regular Expression (regex), IPython.display
        - Contents: Web scraping using BeautifulSoup and TOR, and structuring originating URLs, scraped URLs, text and tables from scraped URLs in Pandas DataFrames for analysis. Extensive cleaning of scraped text via regex. Viewing styled Pandas DataFrames. Viewing scraped tables in html.     
- Convolutional Neural Networks (CNNs) for Image Classification:
    - **Title:** ["Comparative Analysis of CNN Deep Learning Models for X-ray Illness Classification"](./Neural%20Network%20Models/X-Ray%20Deep%20Learning%20Classificaton%20Models.html)
        - Focus: Deep Learning for Medical Image Classification
        - Technology Used: Python, Keras, CNNs, Transfer Learning, ImageDataGenerator, flow_from_directory, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        - Contents: Comparing the performance of 8 CNN deep learning models on X-ray images from three classes (COVID-19, viral pneumonia, and healthy). These include transfer learning models (e.g., InceptionV3), and various techniques to improve model generalization and help avoid overfitting (e.g., dropout, batch normalization, early stopping, data augmentation, L1 and L2 regularization, fire modules, and ways of using deep networks effectively). I also demonstrate best practices for structuring filters/kernels, channels, layers, activation functions, pooling, convolutional blocks, and other model components for optimal performance. Metrics include confusion matrixes, accuracy, precision, recall, F1-score, ROC curve, and AUC. Analysis of non-augmented vs. augmented data models with specific augmentation techniques are shown. Architectures and training strategies for each model are detailed.
- Recurrent Neural Network (RNN) Models:
    - **Title:** ["RNN Text Classification Models for Twitter COVID-19 Misinformation Detection"](./Neural%20Network%20Models/Sequential_RNN_Text_Classification_Models_for_Twitter_Misinformation_Detection.html)
        - Focus: RNN Sequential Models for Text Classification
        - Technology Used: Python, Pandas, NumPy, TensorFlow (TF), Keras, TF/Keras functions (Tokenizer, pad_sequences, Sequential, Embedding, Dense, Flatten, LSTM, GRU, Bidirectional, ModelCheckpoint, ReduceLROnPlateau, RMSprop), Scikit-Learn metrics, one-hot encoding.
        - Contents: Comparing the performance of 5 RNN sequential text classification models (having varying architectures) that were trained on text from accurate and misleading tweets about COVID-19, in order to classify unseen tweets as containing true or false information.