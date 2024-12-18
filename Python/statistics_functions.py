# Jonathan Hyatt
# Extract, Clean, and Visualize Data
import pandas as pd
from pandas import Interval
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import ipywidgets as widgets
from IPython.display import display, HTML
import os
from scipy import stats
import credentials as creds
import liquidity_data
import streamlit as st
from matplotlib import font_manager as fm, rcParams
from datetime import datetime
from adjustText import adjust_text

"""import matplotlib.pyplot as plt
from matplotlib import font_manager
font_path = './latinmodern-math.otf'
font_manager.fontManager.addfont(font_path)"""

plt.rcParams['font.family'] = 'Latin Modern Roman'
plt.rcParams['legend.frameon'] = False
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

def univariate_stats(df: pd.DataFrame, gen_charts: bool, bw: bool = False) -> pd.DataFrame:
  #Step 1 generate output dataframe
  output_df = pd.DataFrame(columns=['Count','Null', 'Total' ,'Unique', 'Type', 'Min', 'Max', '25%', '50%', '75%', 'Mean', 'Std', 'Skew', 'Kurt'])
  #df = df.reset_index(drop=True)
  #df.dropna(axis=1, inplace=True)
  #df.replace({False: 0, True: 1}, inplace=True)
  #Step 2 iterate through columns in df
  for col in df:
    #Check if there are any completely null columns
    if df[col].notna().any():

      #Step 3 determine if column is numeric
      if pd.api.types.is_numeric_dtype(df[col]):
      #Step 4a generate stats for numeric col
        oRow = [df[col].count(),df[col].isna().sum(),df[col].count()+df[col].isna().sum(),df[col].value_counts().count(),df[col].dtype,df[col].min(),df[col].max(),
        df[col].quantile(0.25,),df[col].median(),df[col].quantile(0.75),round(df[col].mean(),2),df[col].std(),df[col].skew(),df[col].kurt()]
        output_df.loc[col] = oRow
        #Step 5a generate hist plot if boolean is true
        if gen_charts:
          str_text = 'Kurtosis is ' + str(round(df[col].kurt(),3)) +'\n' + 'Skew is ' + str(round(df[col].skew(),3))
          plt.text(df[col].min()+0.1*df[col].max(), 0.1*df[col].count(),str_text)
          sns.histplot(data=df, x=col, color='black' if bw else None)
          plt.show()

      else:
        #Step 4b generate stats for categorical col
        oRow = [df[col].count(), df[col].isna().sum(),df[col].count()+df[col].isna().sum(), df[col].value_counts().count(),df[col].dtype,"-","-",
        "-","-","-","-","-","-","-"]

        #Step 5b generate count plot if boolean is true
        if gen_charts:
          plt.xticks(rotation =45)
          sns.countplot(data=df, x=col, palette="Greens_d" if not bw else None, color='black' if bw else None)
          plt.show()
      #Step 6 add row to dataframe
      output_df.loc[col] = oRow

    else:
      df.drop(col, inplace=True,axis=1)

  #step 7 output df
  return output_df

# generate visualizations and statistics for all different bivariate relationships for one column
def bivariate_stats(col: str, df: pd.DataFrame, bw: bool = False) -> None:
  ###### FUNCTIONS USED######
  def anova(feature, label):
    groups = df[feature].unique() # discover each unique group value
    grouped_values = []           # create an overall list of keep track of the label sub-lists
    for group in groups:          # for each unique group value
      grouped_values.append(df[df[feature]==group][label])  # append a sub-list of label values into the overall list
    return stats.f_oneway(*grouped_values)
  ###########################

  # Step 1 create if statement to see if col is numeric
  if pd.api.types.is_numeric_dtype(df[col]):
    # Step 2a iterate through columns of dataframe
    for col_2 in df:
      # Step 3a see if col_2 is numeric
      if pd.api.types.is_numeric_dtype(df[col_2]):
        corr = stats.pearsonr(df[col], df[col_2])
        plt.text(100.5,50,f' p-value: {round(corr[1],4)} \n r: {round(corr[0],4)} ')
        sns.scatterplot(data=df, x=col_2, y=col, palette="pastel" if not bw else None, color='black' if bw else None)
        plt.show()
      else:
        f,p = anova(col_2,col)
        sns.catplot(data=df, x=col_2, y=col, palette="pastel" if not bw else None, color='black' if bw else None)
        plt.text(len(df[col_2].unique()),50,f' p-value: {round(p,4)} \n F Stat: {round(f,4)} ')
        plt.xticks(rotation =90)
        plt.xlabel = col_2
        plt.show()
        print(col,col_2,' are numeric and categorical')

  else:
    # Step 2b iterate through columns of dataframe
    for col_2 in df:
      # Step 3b see if col_2 is numeric
      if pd.api.types.is_numeric_dtype(df[col_2]):
        # Step 4c generate chart and stats for num to cat
        sns.catplot(data=df, x=col, y=col_2, palette="pastel" if not bw else None, color='black' if bw else None)
        plt.xticks(rotation =90)
        plt.xlabel = col_2
        plt.show()
      else:
        # Step 4d generate chart and stats for cat to cat
        viz = sns.catplot(x=col, hue=col_2, kind="count", palette="pastel" if not bw else None, color='black' if bw else None, data=df);
        viz.set_xticklabels(rotation=25);
        plt.show()

def extract_from_liquidity(db_name: str, num_obs: int, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Does a similar thing to extract but specifically from liquidity.
      Args:
      db_name: str, the name of the db to connect to
      num_obs: int, the number of observations to extract (can be None in which case all observations between start_time and end_time are extracted)
      
      Kwargs:
      start_time: datetime, the start time of the observations to extract. Can be None in which case all observations up to end_time (if present) are extracted.
      end_time: datetime, the end time of the observations to extract. Can be None in which case all observations from start_time (if present) are extracted.
      
      If both start_time and end_time are None, then the most recent num_obs observations are extracted.
      If start_time, end_time and num_obs are all None, then all observations are extracted.
      
      returns a df"""
    # Connect to the liquidity_data db that has some of the summary stats
    db = creds.get_db(db_name)
    with db.transaction():
        # Build Peewee queries based on parameters
        if start_time or end_time:
            if start_time and end_time:
                obs = liquidity_data.Liquidity.select().where(liquidity_data.Liquidity.time.between(start_time, end_time))
                if num_obs:
                    obs = obs.order_by(liquidity_data.Liquidity.time.desc()).limit(num_obs)
            elif start_time and not end_time:
                obs = liquidity_data.Liquidity.select().where(liquidity_data.Liquidity.time > start_time)
                if num_obs:
                    obs = obs.order_by(liquidity_data.Liquidity.time.desc()).limit(num_obs)
            elif end_time and not start_time:
                obs = liquidity_data.Liquidity.select().where(liquidity_data.Liquidity.time < end_time)
                if num_obs:
                    obs = obs.order_by(liquidity_data.Liquidity.time.desc()).limit(num_obs)
        else:
            obs = (liquidity_data.Liquidity.select()
                    .order_by(liquidity_data.Liquidity.time.desc())
                    .limit(num_obs))

        obs_df = []
        for ob in obs:
            new_dict = {}
            new_dict.update({'time': ob.time})
            new_dict.update({'exchange': ob.exchange})
            new_dict.update({'midpoint': ob.midpoint})
            new_dict.update({'spread': ob.spread})
            new_dict.update({'price_impact_bid': ob.price_impact_bid})
            new_dict.update({'price_impact_ask': ob.price_impact_ask})
            new_dict.update({'price_impact_bid_quantity': ob.price_impact_bid_quantity})
            new_dict.update({'price_impact_ask_quantity': ob.price_impact_ask_quantity})
            
            obs_df.append(new_dict)
    obs_df = pd.DataFrame(obs_df)
    return obs_df

def sort_and_select(df: pd.DataFrame, by_conditions: list, column_to_return: str) -> pd.Series:
    """
    Sorts the DataFrame based on the given conditions and returns the specified column.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to be sorted.
    by_conditions (list or tuple): The columns to sort by.
    column_to_return (str): The column to return after sorting.
    
    Returns:
    pd.Series: The sorted column from the DataFrame.
    """
    # Sort the DataFrame by the specified conditions
    df_sorted = df.sort_values(by=by_conditions, ascending=False)
    
    # Return the specified column from the sorted DataFrame
    return df_sorted[column_to_return]

def graph_dist_seperate(filtered_df: pd.DataFrame, columns: list, bins: int = 10, bw: bool = False) -> None:
   """Graphs the distributions of the columns we care about all on one histogram for comparison.
      filtered_df: dataframe with the columns we care about as columns.
      columns: list of columns that we care about graphing
      bins (optional, default 10): integer, number of bins we would like on the histogram"""
   for i in range(len(filtered_df[0])):
      df = filtered_df[1].iloc[i]
      for item in columns:
          plt.hist(df[item], bins=bins, color='black' if bw else None)
          plt.legend(frameon=False)
          plt.xlabel(f'{make_title(item)}')
          plt.ylabel('Count')
          #plt.title(f'Distribution of {item.capitalize()} from {str(filtered_df[0][i]).capitalize()}')
          plt.axvline(x=df[item].mean(),linestyle='--',label='Mean')
          plt.show()

def graph_dist_together_single_plots(filtered_df: pd.DataFrame, columns: list, bins: int = 10, bw: bool = False) -> None:
  print('Need to finish this')
  line_styles = ['-', '--', ':', '-.']  
  for item in columns:
    for i in range(len(filtered_df[0])):
      title = make_title(item)
      df = filtered_df[1].iloc[i]
      plt.hist(df[item], bins=bins, alpha=0.5, color='black' if bw else None)
      plt.legend(frameon=False)   
      plt.xlabel(f'{title}')
      plt.ylabel('Count')
      #plt.title(f'Distribution of {title}')
      plt.axvline(x=df[item].mean(),linestyle=line_styles[i % len(line_styles)],label='Mean')
      #labels.append(item)
    plt.show()

def graph_dist_together_subplots(filtered_df: pd.DataFrame, columns: list, bins: int = 10, binance: bool = True, bw: bool = False) -> None:
    """
    Graphs the distributions of the columns we care about all on one histogram for comparison.
    
    Parameters:
    filtered_df: dataframe with the columns we care about as columns.
    columns: list of columns that we care about graphing
    bins (optional, default 10): integer, number of bins we would like on the histogram
    binance (optional, default True): whether to include Binance US data
    bw (optional, default False): if True, use black and white style with different line styles and hatches
    """
    
    num_columns = len(columns)
    num_plots = len(filtered_df[1])
    line_styles = ['-', '--', ':', '-.']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '*', '.']
    
    if not binance:
        colors = colors[1:]  # Skip the first color if binance is False
        line_styles = line_styles[1:]  # Skip the first line style if binance is False
        hatches = hatches[1:]  # Skip the first hatch if binance is False

    # Calculate the number of rows needed
    num_rows = int(np.ceil(num_columns / 2))
    
    # Create subplots with two columns
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))
    
    # Flatten axes if necessary
    if num_columns == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Iterate over each column
    for idx, item in enumerate(columns):
        title = make_title(item)
        
        # Plot histogram in the corresponding subplot
        ax = axes[idx]
        maximum = 0
        minimum = 70000
        for i in range(num_plots):
            df = filtered_df[1].iloc[i]
            maximum = max(maximum, df[item].max())
            minimum = min(minimum, df[item].min())
        bins = np.linspace(minimum, maximum, 75)

        for i in range(num_plots):
            df = filtered_df[1].iloc[i]
            if bw:
                ax.hist(df[item], bins=bins, alpha=0.5, density=True, 
                        label=str(filtered_df[0].iloc[i]).title(), 
                        edgecolor='black', facecolor='white', 
                        hatch=hatches[i % len(hatches)])
                ax.axvline(df[item].mean(), color='black', linestyle=line_styles[i % len(line_styles)])
            else:
                ax.hist(df[item], bins=bins, alpha=0.5, density=True, 
                        label=str(filtered_df[0].iloc[i]).title(), 
                        color=colors[i % len(colors)])
                ax.axvline(df[item].mean(), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
        
        # Set labels and title for the subplot
        ax.set_xlabel(f'{title}')
        ax.set_ylabel('Density')
        ax.legend(frameon=False)
        
        # Save the subplot as a JPG file
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        for i in range(num_plots):
            df = filtered_df[1].iloc[i]
            if bw:
                ax_single.hist(df[item], bins=bins, alpha=0.5, density=True, 
                               label=str(filtered_df[0].iloc[i]).title(), 
                               edgecolor='black', facecolor='white', 
                               hatch=hatches[i % len(hatches)])
                ax_single.axvline(df[item].mean(), color='black', linestyle=line_styles[i % len(line_styles)])
            else:
                ax_single.hist(df[item], bins=bins, alpha=0.5, density=True, 
                               label=str(filtered_df[0].iloc[i]).title(), 
                               color=colors[i % len(colors)])
                ax_single.axvline(df[item].mean(), color=colors[i % len(colors)], linestyle=line_styles[i % len(line_styles)])
        ax_single.set_xlabel(f'{title}')
        ax_single.set_ylabel('Density')
        ax_single.legend(frameon=False)
        filename = f'./figures/dist_plots/dist_{title}{"_Excluding_BinanceUS" if not binance else ""}.jpg'
        fig_single.savefig(filename)
        plt.close(fig_single)

    # Hide unused subplots and adjust layout
    for ax in axes[num_columns:]:
        ax.axis('off')
    
    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def graph_cdf_seperate(filtered_df: pd.DataFrame, columns: list, bins: int = 10, bw: bool = False) -> None:
   """Graphs the distributions of the columns we care about all on one histogram for comparison.
      filtered_df: dataframe with the columns we care about as columns.
      columns: list of columns that we care about graphing
      bins (optional, default 10): integer, number of bins we would like on the histogram"""
   for i in range(len(filtered_df[0])):
      df = filtered_df[1].iloc[i]
      for item in columns:
          ax = sns.ecdfplot(df[item], color='black' if bw else None)
          x, y = ax.lines[0].get_xydata().T
          plt.fill_between(x,y,alpha=0.5)
          plt.xlabel(f'{make_title(item)}')
          plt.ylabel('Count')
          #plt.title(f'Distribution of {item.capitalize()} from {str(filtered_df[0][i]).capitalize()}')
          plt.axvline(x=df[item].mean(),linestyle='--',label='Mean')
          plt.show()

def graph_cdf_together(filtered_df: pd.DataFrame, columns: list, bins: int = 10, save_dir: str = './figures/cdf_plots/', binance: bool = True, bw: bool = False) -> None:
    """
    Graphs the cumulative distribution functions (CDF) of the columns we care about all on one plot for comparison.
    
    Parameters:
    filtered_df: dataframe with the columns we care about as columns.
    columns: list of columns that we care about graphing
    bins (optional, default 10): integer, number of bins we would like on the histogram
    save_dir (optional, default './figures/cdf_plots/'): directory to save the plots
    binance (optional, default True): whether to include Binance US data
    bw (optional, default False): if True, use black and white style with different line styles
    """

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    line_styles = ['-', '--', ':', '-.']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    if not binance:
        colors = colors[1:]  # Skip the first color if binance is False
        line_styles = line_styles[1:]  # Skip the first line style if binance is False

    num_columns = len(columns)
    num_plots = len(filtered_df[1])

    # Calculate the number of rows needed
    num_rows = int(np.ceil(num_columns / 2))

    # Create subplots with two columns
    fig, axes = plt.subplots(num_rows, 2, figsize=(15, num_rows * 5))

    # Flatten axes if necessary
    if num_columns == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, item in enumerate(columns):
        title = make_title(item)
        if not binance:
            title += ' Excluding Binance US'
        
        # Plot CDF in the corresponding subplot
        ax = axes[idx]
        for i in range(num_plots):
            df = filtered_df[1].iloc[i]
            if bw:
                sns.ecdfplot(data=df[item], ax=ax, label=make_title(str(filtered_df[0].iloc[i])), 
                             color='black', linestyle=line_styles[i % len(line_styles)])
            else:
                sns.ecdfplot(data=df[item], ax=ax, label=make_title(str(filtered_df[0].iloc[i])), 
                             color=colors[i % len(colors)])
        
        ax.set_xlabel(f'{title}')
        ax.set_ylabel('Cumulative Probability')
        ax.legend(frameon=False)

        # Save each subplot as a separate image
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        for i in range(num_plots):
            df = filtered_df[1].iloc[i]
            if bw:
                sns.ecdfplot(data=df[item], ax=ax_single, label=make_title(str(filtered_df[0].iloc[i])), 
                             color='black', linestyle=line_styles[i % len(line_styles)])
            else:
                sns.ecdfplot(data=df[item], ax=ax_single, label=make_title(str(filtered_df[0].iloc[i])), 
                             color=colors[i % len(colors)])
        
        ax_single.set_xlabel(f'{title}')
        ax_single.set_ylabel('Cumulative Probability')
        ax_single.legend(frameon=False)

        filename = os.path.join(save_dir, f'cdf_{title}.png')
        fig_single.savefig(filename)
        plt.close(fig_single)

    # Hide unused subplots and adjust layout
    for ax in axes[num_columns:]:
        ax.axis('off')

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()

def minute_range_from_times(start_time: datetime, end_time: datetime) -> pd.Series:
  """
  Creates a pandas Series with minutes as index and times as values.

  Args:
      start_time: The start time as a datetime object.
      end_time: The end time as a datetime object.

  Returns:
      A pandas Series with minutes as index and datetime objects as values.
  """

  # Calculate the total number of minutes
  total_minutes = int((end_time - start_time).total_seconds() / 60)

  # Create a range of datetime objects for each minute
  minute_range = pd.date_range(start=start_time, periods=total_minutes+1, freq='min')[:-1]

  # Create a pandas Series with a counting index from 0
  minute_series = pd.Series(index=range(total_minutes), data=minute_range)

  return minute_series

def display_and_save_correlation_matrix(df: pd.DataFrame, title: str, rounding: int = 2, save_directory: str = './figures/corr_tables/', bw: bool = False) -> None:
    """
    Display the upper triangle of the correlation matrix of a DataFrame as a styled table with a specified title.
    Save the table as an image file in the specified directory.

    Parameters:
    df (pd.DataFrame): The DataFrame to compute the correlation matrix from.
    title (str): The title for the table and used as the filename.
    rounding (int): The number of decimal places to round the correlation coefficients.
    save_directory (str): Directory path where the image file will be saved. Default is './figures/corr_tables/'.

    Returns:
    None
    """
    # Compute the correlation matrix
    corr_matrix = df.corr().round(rounding)
    
    # Mask the lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    masked_corr_matrix = corr_matrix.where(mask)
    
    # Convert NaN values to empty strings
    masked_corr_matrix = masked_corr_matrix.map(lambda x: '' if pd.isnull(x) else x)
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    
    # Render the table
    table = ax.table(cellText=masked_corr_matrix.values, colLabels=masked_corr_matrix.columns, 
                     rowLabels=masked_corr_matrix.index, cellLoc='center', loc='center', fontsize=14, 
                     cellColours=[['white' if bw else None]*len(masked_corr_matrix.columns)]*len(masked_corr_matrix.index))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.5, 1.2)  # Adjust column width to one quarter
    
    # Remove cell colors
    for key, cell in table.get_celld().items():
        cell.set_facecolor('white')
        cell.set_text_props(fontsize=12)

    # Set the header style to white background with black text
    for key, cell in table.get_celld().items():
        if key[0] == 0 or key[1] == -1:
            cell.set_text_props(fontweight='bold', color='black')
            cell.set_facecolor('white')
    
    # Set the title
    plt.title(title, size=15)
    
    # Save the table as an image file
    file_path = os.path.join(save_directory, f"{title}.png")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    
    # Display the table
    plt.show()

def output_tables(df: pd.DataFrame, title: str = "") -> None:
    """Args: df, pandas dataframe
       Kwargs: title, the title for the table
       This function takes a df and returns an HTML table with a title and prints a LaTeX table."""
    
    # Convert the DataFrame to HTML without the index
    df_html = df.to_html(index=False)
    
    # Create the full HTML with the title
    html_output = f"<h2>{title}</h2>" + df_html
    
    # Create an interactive output widget
    output = widgets.Output()
    
    with output:
        display(HTML(html_output))
    
    display(output)
    
    # Prepare the LaTeX table string
    latex_table = (
        "\\begin{center}\n"
        "\\begin{table}[ht]\n"
        "    \\centering\n"
        "    \\begin{tabular}{c " + "c" * (len(df.columns) - 1) + "}\n"
        "    \\toprule\n"
        "     & " + " & ".join(df.columns) + " \\\\\n"
        "     \\midrule\n"
    )
    
    # Add rows for each index
    for index, row in df.iterrows():
        formatted_row = []
        for val in row:
            if isinstance(val, (int, float)):
                formatted_row.append(f"{val:.2f}")
            else:
                formatted_row.append(str(val))
        latex_table += f"  " + " ".join(formatted_row) + " \\\\\n"
    
    # Close the table with the specific caption
    latex_table += (
        "    \\bottomrule\n"
        "    \\end{tabular}\n"
        "    \\caption{"
        f"{title}"
        "}\n"
        "    \\label{tab:my_label}\n"
        "\\end{table}\n"
        "\\end{center}"
    )
    
    print(latex_table)

def display_correlation_matrix(df: pd.DataFrame, title: str, rounding: int = 2) -> None:
    """
    Display the upper triangle of the correlation matrix of a DataFrame as an HTML table with a specified title.

    Parameters:
    df (pd.DataFrame): The DataFrame to compute the correlation matrix from.
    title (str): The title for the HTML output to be displayed between the <h2> tags.
    rounding (int): The number of decimal places to round the correlation coefficients.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr().round(rounding)
    
    # Mask the lower triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    masked_corr_matrix = corr_matrix.where(mask)
    
    # Convert the masked correlation matrix to HTML with NaN replaced by empty strings
    html_output = masked_corr_matrix.to_html(na_rep='')
    
    # Create an interactive output widget
    output = widgets.Output()
    
    with output:
        display(HTML(f"<h2>{title}</h2>" + html_output))
    
    display(output)

def display_save_regressions(df: pd.DataFrame, title: str, rounding: int = 2, save_directory: str = './figures/regressions/') -> None:
    """
    Display and save a regression table.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the regression results
    title (str): The title for the table and used as the filename.
    rounding (int): The number of decimal places to round the correlation coefficients.
    save_directory (str): Directory path where the image file will be saved. Default is './figures/corr_tables/'.

    Returns:
    None
    """    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    
    # Render the table
    table = ax.table(cellText=df.values.round(rounding), colLabels=df.columns, 
                     rowLabels=df.index, cellLoc='center', loc='center', fontsize=14)

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.85, 1.2)  # Adjust column width to one quarter
    
    # Remove cell colors
    for key, cell in table.get_celld().items():
        cell.set_facecolor('white')
        cell.set_text_props(fontsize=12)

    # Set the header style to white background with black text
    for key, cell in table.get_celld().items():
        if key[0] == 0 or key[1] == -1:
            cell.set_text_props(fontweight='bold', color='black')
            cell.set_facecolor('white')
    
    # Set the title
    plt.title(title, size=15)
    
    # Save the table as an image file
    file_path = os.path.join(save_directory, f"{title}.png")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    
    # Display the table
    plt.show()

def subplot_plot_over_time(ax: plt.Axes, df: pd.DataFrame, y: str, start_time: datetime = None, end_time: datetime = None) -> None:
    sns.lineplot(ax=ax, data=df, x='time', y=y, hue='exchange', style='exchange', markers=False, dashes=False, palette='tab10')
    ax.set_xlabel('Time')
    if start_time != None:
        ax.xlim(start_time,end_time)
    ax.set_ylabel(y)
    #ax.set_title(f'{y} Over Time')
    ax.legend(title='Exchange',frameon=False)
    ax.grid(True)

def single_plot_over_time(df: pd.DataFrame, y: str, start_time: datetime = None, end_time: datetime = None) -> None:
    # Plotting using seaborn for better aesthetics
    plt.figure(figsize=(12, 6))

    sns.lineplot(data=df, x='time', y=y, hue='exchange', style='exchange', markers=False, dashes=False, palette='tab10')

    if start_time and end_time:
        plt.xlim(start_time, end_time)
        
    plt.xlabel('Time')
    plt.ylabel('Midpoint Price')
    #plt.title('Midpoint Price Over Time for Four Exchanges')
    plt.legend(title='Exchange', frameon=False)
    plt.grid(True)

    plt.show()

def together_plot_attr_over_time(df: pd.DataFrame, y: str, start_time: datetime = None, end_time: datetime = None, ylim: tuple = None, streamlit: bool = False, binned: bool = False, bw: bool = False) -> None:
    """
    Plots time series data over time with options for x-axis limits, y-axis limits, binning, and black and white style.
    
    Parameters:
    - df: DataFrame containing the time series data.
    - y: Column name in the DataFrame to plot on the y-axis.
    - start_time: Start time for the x-axis (optional).
    - end_time: End time for the x-axis (optional).
    - ylim: Tuple specifying y-axis limits, e.g., (0, 100) (optional).
    - streamlit: Boolean indicating whether to display the plot in a Streamlit app (optional).
    - binned: Boolean indicating whether to bin the data by every 15 seconds (optional).
    - bw: Boolean indicating whether to use black and white style (optional).
    """
    # Bin the data if required
    if binned:
        # Ensure the 'time' column is a datetime object
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        # Resample the data by 15-second intervals
        df = (
            df.set_index('time')
            .groupby('exchange')
            .resample('120s')
            .mean()
            .reset_index()
        )
    
    # Create a new figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define line styles and markers for black and white plotting
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']

    # Plot using seaborn for better aesthetics
    if bw:
        for i, exchange in enumerate(df['exchange'].unique()):
            exchange_data = df[df['exchange'] == exchange]
            ax.plot('time', y, data=exchange_data, 
                    linestyle=line_styles[i % len(line_styles)],
                    marker=markers[i % len(markers)],
                    markevery=0.1,  # Plot marker every 10th point to reduce clutter
                    markersize=5,
                    color='black',
                    label=exchange)
    else:
        sns.lineplot(data=df, x='time', y=y, hue='exchange', style='exchange', 
                     markers=True, dashes=False, palette='tab10', ax=ax)

    # Set x-axis limits if provided
    if start_time and end_time:
        ax.set_xlim(start_time, end_time)
    
    # Set y-axis limits if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Set axis labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(f'{make_title(y)}')
    
    # Add legend and grid
    ax.legend(title='Exchange', frameon=False)
    ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Display the plot in Streamlit if required
    if streamlit:
        st.pyplot(fig)

def together_plot_exc_over_time(df: pd.DataFrame, columns: list, start_time: datetime = None, end_time: datetime = None) -> None:
    # Filter the DataFrame by the specified exchange

    plt.figure(figsize=(12, 6))

    # Loop through each column and plot it
    for column in columns:
        if column == 'midpoint':
           df[column] = df[column] / 10000
        sns.lineplot(data=df, x='time', y=column, label=column)

    if start_time and end_time:
        plt.xlim(start_time, end_time)

    plt.xlabel('Time')
    plt.ylabel('Value')
    #plt.title(f'Attributes Over Time for {str(df['exchange'].iloc[0]).capitalize()}')
    plt.legend(title='Attributes', frameon=False)
    plt.grid(True)

    plt.show()

def merge_dfs_within_time_tolerance(dfs: list[pd.DataFrame], reference_df_index: int = 0, time_suffix: str = 'time', tolerance: pd.Timedelta = pd.Timedelta('3s')) -> pd.DataFrame:
    # Find time column in each dataframe
    def find_time_column(df, suffix):
        return next(col for col in df.columns if col.endswith(suffix))

    # Ensure all dfs have the time column as datetime and sorted
    for i in range(len(dfs)):
        time_col = find_time_column(dfs[i], time_suffix)
        dfs[i][time_col] = pd.to_datetime(dfs[i][time_col])
        dfs[i] = dfs[i].sort_values(by=time_col)

    # Set the reference dataframe
    reference_df = dfs[reference_df_index]
    reference_time_col = find_time_column(reference_df, time_suffix)

    # Merge dataframes
    result_df = reference_df.copy()
    for i in range(len(dfs)):
        if i == reference_df_index:
            continue
        time_col = find_time_column(dfs[i], time_suffix)
        result_df = pd.merge_asof(result_df, dfs[i], left_on=reference_time_col, right_on=time_col, tolerance=tolerance, direction='nearest', suffixes=('', f'_{i}'))

    return result_df

def analyze_data_gaps(df: pd.DataFrame, time_column: str = 'time', threshold_seconds: int = 6) -> pd.DataFrame:
    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Sort the DataFrame by the time column to ensure correct time difference calculations
    df = df.sort_values(by=time_column)
    
    # Calculate the time differences between consecutive rows
    df['time_diff'] = df[time_column].diff().dt.total_seconds()
    
    # Identify gaps longer than the threshold
    gaps = df[df['time_diff'] > threshold_seconds].copy()
    
    # Collect information about the gaps
    gap_info = []
    for index, row in gaps.iterrows():
        gap_start = df.loc[index - 1, time_column]  # The time before the gap
        gap_end = row[time_column]  # The time after the gap
        gap_duration = row['time_diff']  # Duration of the gap
        
        # Convert the gap duration to a readable format
        if gap_duration < 60:
            duration_str = f"{gap_duration:.2f} seconds"
        elif gap_duration < 3600:
            duration_str = f"{gap_duration / 60:.2f} minutes"
        elif gap_duration < 86400:
            duration_str = f"{gap_duration / 3600:.2f} hours"
        else:
            duration_str = f"{gap_duration / 86400:.2f} days"
        
        gap_info.append({
            'start': gap_start,
            'end': gap_end,
            'start_day': gap_start.date(),
            'start_time': gap_start.time(),
            'end_day': gap_end.date(),
            'end_time': gap_end.time(),
            'duration_mod': duration_str,
            'duration': gap_duration
        })
    
    # Create a DataFrame with the gap information
    gap_df = pd.DataFrame(gap_info)
    
    # Hide the original start and end columns
    columns_to_show = ['start_day', 'start_time', 'end_day' , 'end_time', 'duration_mod', 'duration']

    gap_df = gap_df[columns_to_show]
    
    return gap_df

def save_tables(df: pd.DataFrame, title: str = "", file_path: str = "output.png", save_directory: str = './figures/gap/') -> None:
    """
    Args: 
        df (pd.DataFrame): The DataFrame containing the data.
        title (str): The title for the table and used as the filename.
        file_path (str): The file path to save the chart.
        rounding (int): The number of decimal places to round the numeric data.
        save_directory (str): Directory path where the image file will be saved. Default is './figures/tables/'.
        
    Returns:
        None
    """
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    
    # Render the table
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, rowLabels=df.index, 
                     cellLoc='center', loc='center', fontsize=14)
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.85, 1.2)  # Adjust column width to one quarter
    
    # Remove cell colors
    for key, cell in table.get_celld().items():
        cell.set_facecolor('white')
        cell.set_text_props(fontsize=12)
    
    # Set the header style to white background with black text
    for key, cell in table.get_celld().items():
        if key[0] == 0 or key[1] == -1:
            cell.set_text_props(fontweight='bold', color='black')
            cell.set_facecolor('white')
    
    # Set the title
    plt.title(title, size=15)
    
    # Save the table as an image file
    file_path = os.path.join(save_directory, f"{title}.png")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(file_path, bbox_inches='tight', dpi=300)
    
    # Display the table
    plt.show()

def get_consistent_intervals(df: pd.DataFrame, time_column: str = 'time', interval_seconds: int = 10, tolerance_seconds: int = 5) -> pd.DataFrame:
    # Ensure the time column is in datetime format
    df[time_column] = pd.to_datetime(df[time_column])
    
    # Calculate the time differences between consecutive rows
    df['time_diff'] = df[time_column].diff().dt.total_seconds()
    
    # Identify periods with consistent intervals
    consistent_intervals = []
    start_idx = None

    for idx, diff in df['time_diff'].items():
        if pd.notnull(diff) and abs(diff - interval_seconds) <= tolerance_seconds:
            if start_idx is None:
                start_idx = idx - 1  # Start of the consistent interval
        else:
            if start_idx is not None:
                end_idx = idx - 1  # End of the consistent interval
                start_time = df.loc[start_idx, time_column]
                end_time = row[time_column]
                
                # If start_day and end_day are different, merge the interval into one
                if start_time.date() != end_time.date():
                    end_time = df.loc[idx - 1, time_column]
                
                consistent_intervals.append({
                    'start': start_time,
                    'end': end_time,
                    'start_day': start_time.date(),
                    'start_time': start_time.time(),
                    'end_day': end_time.date(),
                    'end_time': end_time.time(),
                    'duration': abs(end_time - start_time)
                })
                start_idx = None

    # Handle case where the last interval is consistent
    if start_idx is not None:
        start_time = df.loc[start_idx, time_column]
        end_time = df.loc[df.index[-1], time_column]
        
        # If start_day and end_day are different, merge the interval into one
        if start_time.date() != end_time.date():
            end_time = df.loc[df.index[-1], time_column]
        
        consistent_intervals.append({
            'start': start_time,
            'end': end_time,
            'start_day': start_time.date(),
            'start_time': start_time.time(),
            'end_day': end_time.date(),
            'end_time': end_time.time(),
            'duration': abs(end_time - start_time)
        })

    # Create a DataFrame with the interval information
    intervals_df = pd.DataFrame(consistent_intervals)
    
    return intervals_df

def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df['start'] = pd.to_datetime(df['start'])
    df['end'] = pd.to_datetime(df['end'])
    return df

def rename_columns_for_merge(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    Rename columns in a DataFrame to avoid conflicts during merge.
    """
    df_renamed = df.rename(columns=lambda x: f"{x}_{suffix}" if x != 'start' else x)
    return df_renamed

def merge_and_filter_durations(dfs: list[pd.DataFrame], on: str = 'start', max_time_diff_minutes: int = 10) -> pd.DataFrame:
    """
    Perform an asof merge between multiple DataFrames, filter based on a maximum time difference,
    add a `min_duration` column, compute the `end_time`, and display the result.

    Parameters:
    - dfs: List of DataFrames to merge. Each DataFrame must have the 'start' column and a 'duration' column.
    - on: Column name to join on (default is 'start').
    - max_time_diff_minutes: Maximum allowed time difference in minutes (default is 10).

    Returns:
    - Filtered DataFrame with `start`, `min_duration`, and `end_time` columns.
    """
    if len(dfs) < 2:
        raise ValueError("At least two DataFrames are required for merging.")

    # Rename columns in each DataFrame to avoid conflicts
    dfs_renamed = [rename_columns_for_merge(df, idx) for idx, df in enumerate(dfs)]

    # Perform sequential asof merges
    merged_df = dfs_renamed[0]
    for df in dfs_renamed[1:]:
        merged_df = pd.merge_asof(merged_df, df, on=on)

    # Define the maximum allowed time difference
    max_time_diff = pd.Timedelta(minutes=max_time_diff_minutes)

    # Calculate time difference between rows
    merged_df['time_diff'] = merged_df[on] - merged_df[on].shift(-1)

    # Filter rows where the time difference is within the allowed range
    filtered_df = merged_df[merged_df['time_diff'] <= max_time_diff]

    # Drop the time_diff column
    filtered_df = filtered_df.drop(columns='time_diff')

    # Find all duration columns
    duration_cols = [col for col in filtered_df.columns if 'duration' in col]

    # Add min_duration column
    filtered_df['min_duration'] = filtered_df[duration_cols].min(axis=1)
    
    # Calculate end_time
    filtered_df['end_time'] = filtered_df['start'] + filtered_df['min_duration'].apply(pd.to_timedelta)

    # Display the result sorted by min_duration in descending order
    result_df = filtered_df.sort_values(by='min_duration', ascending=False)[['start', 'min_duration', 'end_time']]
    
    return result_df

def add_duration_column(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure 'start' and 'end' columns are in datetime format
    df['start'] = pd.to_datetime(df['start'], errors='coerce')
    df['end'] = pd.to_datetime(df['end'], errors='coerce')
    
    # Calculate duration and add it as a new column
    df['duration'] = df['end'] - df['start']
    
    return df

def find_overlap_periods(df: pd.DataFrame, max_time_diff: pd.Timedelta = pd.Timedelta(seconds=6)) -> list[tuple]:
    # Convert time to datetime if it's not already
    df.loc[:,'time'] = pd.to_datetime(df['time'])
    
    # Sort the dataframe by time
    df = df.sort_values('time')
    
    # Group by exchange and create intervals
    exchange_intervals = {}
    for exchange, group in df.groupby('exchange'):
        intervals = []
        for _, row in group.iterrows():
            start = row['time']
            end = start + max_time_diff
            intervals.append(Interval(start, end, closed='both'))
        exchange_intervals[exchange] = intervals
    
    # Find the overall start and end times
    overall_start = df['time'].min()
    overall_end = df['time'].max() + max_time_diff
    
    # Create a range of time points (e.g., every second)
    time_range = pd.date_range(start=overall_start, end=overall_end, freq='1s')
    
    # Check which time points are in all exchanges' intervals
    overlap_points = []
    for t in time_range:
        if all(any(t in interval for interval in intervals) 
               for intervals in exchange_intervals.values()):
            overlap_points.append(t)
    
    # Merge consecutive points into periods
    overlap_periods = []
    if overlap_points:
        start = overlap_points[0]
        for i in range(1, len(overlap_points)):
            if overlap_points[i] - overlap_points[i-1] > pd.Timedelta(seconds=1):
                overlap_periods.append((start, overlap_points[i-1]))
                start = overlap_points[i]
        overlap_periods.append((start, overlap_points[-1]))
    
    return overlap_periods

def make_title(text: str) -> str:
    """
    Make the exchange names look better for titles and other presentation in plots.

    Example Usage: 

    \t make_title("binance_us") -> "Binance US"
    """
    text = text.replace("_", " ").capitalize()
    # Specific changes for price impact and spread
    if 'ask' in str(text).lower():
        text = 'Price Impact (Ask)'
    if 'bid' in str(text).lower():
        text = 'Price Impact (Bid)' 
    if 'midpoint' in str(text).lower():
        text = 'Bid-Ask Midpoint'
    if 'binanceus' in str(text).lower():
        text = 'Binance US'
    return text

def analyze_and_plot_liquidity(
    db_name: str, 
    num_obs: int, 
    start_time: datetime, 
    end_time: datetime, 
    to_keep: list[str], 
    columns: list[str], 
    colors: list = colors,
    bw: bool = False
) -> None:
    """
    Analyzes liquidity data and plots overlaid histograms for specified exchanges and metrics.

    Parameters:
    - db_name (str): Database name to extract data from.
    - num_obs (int): Number of observations to extract.
    - start_time (datetime): Start time for data extraction.
    - end_time (datetime): End time for data extraction.
    - to_keep (list): List of exchanges to analyze (e.g., ['coinbase', 'gemini']).
    - columns (list): List of metrics to analyze (e.g., ['price_impact_bid', 'price_impact_ask', 'spread']).
    - colors (list): List of colors for the exchanges (e.g., ['blue', 'green', 'red', 'orange']).
    """
    print(f'Extracting data from the database for start_time: {start_time} to end_time: {end_time}')
    
    # Extract the data
    obs_df = extract_from_liquidity(db_name, num_obs, start_time, end_time)

    # Filter by exchanges of interest
    obs_df = obs_df[obs_df['exchange'].isin(to_keep)]

    # Compute mean values for each exchange and metric
    mean_values = obs_df.groupby('exchange')[columns].mean()

    # Prepare data for overlaid histograms
    x = np.arange(len(columns))  # One bar per metric
    bar_width = 0.2  # Reduced width to avoid overlap
    
    # Plot
    plt.figure(figsize=(10, 6))
    for idx, (exchange, color) in enumerate(zip(to_keep, colors)):
        exchange_means = mean_values.loc[exchange].values
        plt.bar(
            x + idx * bar_width,  # Offset each exchange's bars slightly
            exchange_means, 
            bar_width, 
            label=make_title(exchange), 
            color='black' if bw else color, 
            alpha=0.6
        )

    # Add labels, title, and legend
    plt.xticks(x + (bar_width * (len(to_keep) - 1) / 2), [make_title(col) for col in columns], rotation=45)
    plt.xlabel("Metrics")
    plt.ylabel("Mean Value")
    title = f"Mean of Price Impact and Spread ({str(start_time.strftime('%B %d')).capitalize()} - {end_time.strftime('%B %d')})"
    plt.title(make_title(title))
    plt.legend(title="Exchanges", frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Save or show the plot
    # plt.savefig(f'./figures/summary_figs/overlaid_histogram_{start_time.strftime("%Y%m%d")}.jpg')
    plt.show()
    print(f"Analysis complete for start_time: {start_time}, end_time: {end_time}")

def analyze_and_plot_liquidity_bw(
    db_name: str, 
    num_obs: int, 
    start_time: datetime, 
    end_time: datetime, 
    to_keep: list[str], 
    columns: list[str],
    bw: bool = False
) -> None:
    """
    Analyzes liquidity data and plots overlaid histograms for specified exchanges and metrics.

    Parameters:
    - db_name (str): Database name to extract data from.
    - num_obs (int): Number of observations to extract.
    - start_time (datetime): Start time for data extraction.
    - end_time (datetime): End time for data extraction.
    - to_keep (list): List of exchanges to analyze (e.g., ['coinbase', 'gemini']).
    - columns (list): List of metrics to analyze (e.g., ['price_impact_bid', 'price_impact_ask', 'spread']).
    """
    print(f'Extracting data from the database for start_time: {start_time} to end_time: {end_time}')
    
    # Extract the data
    obs_df = extract_from_liquidity(db_name, num_obs, start_time, end_time)

    # Filter by exchanges of interest
    obs_df = obs_df[obs_df['exchange'].isin(to_keep)]

    # Compute mean values for each exchange and metric
    mean_values = obs_df.groupby('exchange')[columns].mean()

    # Prepare data for overlaid histograms
    x = np.arange(len(columns))  # One bar per metric
    bar_width = 0.2  # Reduced width to avoid overlap
    
    # Define patterns and grayscale colors
    patterns = ['/', '\\', 'x', '+', 'o', '*']
    grays = ['0.1', '0.3', '0.5', '0.7', '0.9']

    # Plot
    plt.figure(figsize=(12, 6))
    for idx, exchange in enumerate(to_keep):
        exchange_means = mean_values.loc[exchange].values
        plt.bar(
            x + idx * bar_width,  # Offset each exchange's bars slightly
            exchange_means, 
            bar_width, 
            label=make_title(exchange), 
            color='black' if bw else grays[idx % len(grays)],
            edgecolor='black',
            hatch=patterns[idx % len(patterns)] if bw else None,
            alpha=0.8
        )

    # Add labels, title, and legend
    plt.xticks(x + (bar_width * (len(to_keep) - 1) / 2), [make_title(col) for col in columns], rotation=45)
    plt.xlabel("Metrics")
    plt.ylabel("Mean Value")
    title = f"Mean of Price Impact and Spread ({str(start_time.strftime('%B %d')).capitalize()} - {end_time.strftime('%B %d')})"
    plt.title(make_title(title))
    plt.legend(title="Exchanges", frameon=False)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    print(f"Analysis complete for start_time: {start_time}, end_time: {end_time}")

def scatter_ex(desc_df_selected: pd.DataFrame, x_col: str = 'spread', y_col: str = 'price_impact_bid', x_offset: float = 0, y_offset: float = 0, bw: bool = False) -> None:
    """
    Create a scatter plot with non-overlapping annotations for exchanges.
    
    Parameters:
    desc_df_selected (DataFrame): The DataFrame containing the data.
    x_col (str): The column name for the x-axis data (default: 'spread').
    y_col (str): The column name for the y-axis data (default: 'price_impact_bid').
    x_offset (float): Initial offset for x-coordinate of annotations (default: 0).
    y_offset (float): Initial offset for y-coordinate of annotations (default: 0).
    """
    X = desc_df_selected[x_col]['mean']
    y = desc_df_selected[y_col]['mean']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(X, y, color='black' if bw else None)
    ax.set_xlabel(f'Mean {make_title(x_col)}')
    ax.set_ylabel(f'Mean {make_title(y_col)}')
    
    texts = []
    for i, row in desc_df_selected.iterrows():
        texts.append(ax.text(X[i] + x_offset, y[i] + y_offset, make_title(i)))
    
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pass
