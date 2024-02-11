# stock-django

### Description
This is Stock Django Website that does to following
#### 1. Show Stock data based on provided file (The file is very specific and is not uploaded for reasons)
#### 2. Read from json file and show stock data as table from 'From Json' in navbar
#### 3. Read from csv file and show stock data as table from 'From CSV' in navbar
#### 4. '/save-to-databse' calls a view that reads data and saves that in the db.sqlite3 database
#### 5. '/add-data' renders a template that lets the user add stock data to database
#### 6. In index every row of a table can be edited using edit button that renders '/edit-data/<int:stock_id>' to edit specific stock.
#### 7. In index any row can be deleted from database using delete button.
#### 8. Index by default shows every 'Close' data of all tradecode in a line chart where x axis is date.
#### 9. There are 7 types of charts that represents spcific values of tradecode for visualization. User can chose tradecode, types of data they want to visualize. Also multi dimentional chart exist.
#### 10. A ML model (LSTM) devised to predict next date data (high, low, open, close) based on provided data that trains the model for specific tradecode and gives prediction based on last few days of data.
#### 11. admin panel has support to show database data

### Requirements
Pytohn Modules:
django, plotly, tensorflow, python-dotenv, numpy, pandas, scikit-learn

Files:
stock_market_data.csv and stock_market_data.json must be present in utils for this to work. those contains date, tradecode, high, low, open, close and volume. wuthout it, website will not run.

### Notables
#### 1. For charts, line, bar and pie type chart needs Y axis data. bar + line, 3d chart needs both Y axis and Z axis data. Candlestick and Sunburst chart need only trade code. Total in line chart only need Y axis, no tradecode required.

#### 2. Prediction will take tradecode, what type of data one wants to predict (close, high, low, open) and how many days they want to look back. To lower complexity, epoch and layers of nural network is significantly lower. one can change it from Test_ML file inside utils. But for users, only tradecode, type and how many days they want to look bac to train the data is available to change only.

### Learning
#### 1. Code written in very simple format and readability is considered while the code was written.
#### 2. Complexisty of the code is reduced as much as possible for understanding. This may result in slight performance issue.
#### 3. Error handling considered as much as possible while making things simple.
#### 4. Learning: Use of web-framework for python, Understanding and application of python language, Integration of machine learning is application for specified subject, interaction and communication between frontend and baackend, use of charts and graphs in website to visualize data.
