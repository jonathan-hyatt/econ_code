# econ_code
I've included work that I have done in STATA and Python, as well as a writing sample from LaTeX.

In the STATA folder I have included one of my econometrics projects. I took the lead on the coding portion of the project and wrote nearly all the code myself. The project included merging data from multiple sources, creating new variables, and running regression analysis.

In the Python folder I have included some of the code written as a research assistant for Dr. Scott Condie. 
* The Binance_us.py file is an example of a script that collects data from BinanceUs' trading api that we used to collect data about exchange liquidity. The code collects the data and stores it to a database for future analysis. I wrote a similar script to collect and parse data from 25+ other APIs.
* The run_all.py is a script used to run all subscripts to collect the data.
* The statistics_functions.py file is a module where I include the analysis and vizualization functions used in the stats_notebook.ipynb file. I accessed the database where the data was stored and then analyzed different time periods. The statistics_functions.py file has complex data manipulation, visualization, and includes code to write LaTeX. 

The Exchange Liquidity pdf is a writing sample from my research with Dr. Condie. This is a draft I wrote that summarizes some of our findings. The LaTeX I wrote includes well formatted tables and plots. 

The other projects folder contains other projects I work on off and on. 
* Solow_growth_app.py is a streamlit file that is a dynamic solow growth model. It allows for parameters to be input and shows changes. It shows a plot of capital, output, consumption, and labor over time as well as an Acemoglu diagram. It also shows before and after plots once changes are made. 
