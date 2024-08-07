# Portfolio

### Education
- MA in Quantitative Methods in the Social Sciences (Concentration: Data Science), Columbia University, Graduate School of Arts and Sciences (GSAS)
- MPA in International Affairs (Specializations: United Nations Studies, International Conflict Resolution), Columbia University, School of International & Public Affairs (SIPA)
- BA in Political Science (Concentration: International Affairs), Northwestern University

### Work Experience
- Data Scientist Intern @ Volkswagen Group of America (VGoA) - Belmont, CA - 2022-2023
- SIPA Capstone Project Consultant @ International Peace Institute (IPI) - New York, NY - 2020
- Protection of Civilians (PoC) Team Intern, Division of Policy, Evaluation, and Training @ United Nations Department of Peace Operations (UN DPO) - New York, NY - 2019
- Refugee Status Determination (RSD) Intern @ United Nations High Commissioner for Refugees (UNHCR) - Cairo, Egypt - 2016

### Conflict Data Science Projects
- GIS & Spatial Analysis:
    - **Title:** ["Examining Geospatial Covariate Relationships with Civilian Killings in South Sudan’s Civil War"](./GIS/GIS_covariate_relationships-killings-south_sudan.html)
        - Focus: Geospatial Statistics
        - Technology Used: QGIS, R, GeoDa
        - Contents: Geospatial interpolation, hot spot analysis, Moran's I calculation, Moran's I residual analysis, Lagrange Multiplier (LM) and Robust LM lag and error diagnostics, and Spatial Durbin models.
- Data Visualization Storytelling:
    - **Title: "UNAMID: Did the UN’s Withdrawal from Darfur Lead to More Violence against Civilians?"**
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
- Large Language Models (LLMs):
    - **Title:** ["RAG for LLMs"](./LLMs/RAG/NVIDIA_NIM_RAG_Demo/RAG_Demo.html)
        - Focus: Enhancing LLM Performance and Utility for Internal Organizational Tasks
        - Technology Used: Python, NVIDIA NIM, LangChain, Streamlit, IPython.display, NVIDIA Vector Embeddings, FAISS Vector Store, Meta Llama 3.1-8B LLM
        - Contents: Proof-of-concept version of a RAG system for using LLMs to search pdf documents stored on a computer system to provide accurate context when answering a user's chatbot queries.
- CNN Deep Learning for Imaging:
    - **Title:** ["Comparative Analysis of CNN Deep Learning Models for X-ray Illness Classification"](./Neural%20Network%20Models/X-Ray%20Deep%20Learning%20Classificaton%20Models.html)
        - Focus: Deep Learning for Image Classification
        - Technology Used: Python, Keras, CNNs, Transfer Learning, ImageDataGenerator, flow_from_directory, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        - Contents: Comparing the performance of 8 CNN deep learning models on X-ray images from three classes (COVID-19, viral pneumonia, and healthy). These include transfer learning models (e.g., InceptionV3), and various techniques to improve model generalization and help avoid overfitting (e.g., dropout, batch normalization, early stopping, data augmentation, L1 and L2 regularization, fire modules, and ways of using deep networks effectively). I also demonstrate best practices for structuring filters/kernels, channels, layers, activation functions, pooling, convolutional blocks, and other model components for optimal performance. Metrics include confusion matrixes, accuracy, precision, recall, F1-score, ROC curve, and AUC. Analysis of non-augmented vs. augmented data models with specific augmentation techniques are shown. Architectures and training strategies for each model are detailed.