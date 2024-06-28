import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
# Import the data from medical_examination.csv and assign it to the df variable
df = pd.read_csv('medical_examination.csv')

# 2
# Calculate BMI for each row.  If value > 25, mark BMI 1 for overweight
# Metric BMI = weight(kg) / height(cm)^2 * 10000
df['overweight'] = df.apply(lambda row: 1 if (row['weight'] / (row['height'] ** 2)) * 10000 > 25 else 0, axis=1)

# 3
# Normalize data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1, set 
# the value to 0. If the value is more than 1, set the value to 1.
df[['cholesterol', 'gluc']] = df[['cholesterol', 'gluc']].apply(lambda x: x.gt(1).astype(int), axis=1)

# 4 
# Draw the Categorical Plot
def draw_cat_plot():
    # 5
    # Create a DataFrame for the cat plot using pd.melt with values from cholesterol, gluc, smoke, alco, 
    # active, and overweight in the df_cat variable
    df_cat = pd.melt(
        df, 
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6
    # Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. 
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index()
    # You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.rename(columns={0: 'total'})

    # 7
    # Convert the data into long format and create a chart that shows the value counts of the categorical 
    # features using the following method provided by the seaborn library import : sns.catplot()
    sns.catplot(data=df_cat, x='variable', y='total', kind='bar', hue='value', col='cardio')

    # 8 
    # Get the figure for the output and store it in the fig variable
    fig = sns.catplot(data=df_cat, x='variable', y='total', kind='bar', hue='value', col='cardio').fig

    # 9
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# 10
# Draw the Heat Map
def draw_heat_map():
    # 11
    # Clean the data in the df_heat variable by filtering out the following patient segments that represent 
    # incorrect data:
    #
    #   diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
    #   height is less than the 2.5th percentile (Keep the correct data with 
    #       (df['height'] >= df['height'].quantile(0.025)))
    #   height is more than the 97.5th percentile
    #   weight is less than the 2.5th percentile
    #   weight is more than the 97.5th percentile

    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= (df['height'].quantile(0.025))) &
        (df['height'] <= (df['height'].quantile(0.975))) &
        (df['weight'] >= (df['weight'].quantile(0.025))) &
        (df['weight'] <= (df['weight'].quantile(0.975)))
    ]

    # 12
    # Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()

    # 13
    # Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(corr)

    # 14
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(9,9))

    # 15
    # Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
    sns.heatmap(
      corr,
      mask=mask,
      annot=True,
      square=True,
      ax=ax,
      linewidth=0.5,
      fmt='.1f'
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
