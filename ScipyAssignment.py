import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.stats import skew, kurtosis
from scipy.stats import ttest_rel, ttest_1samp

# loading the data
file_path = "/Users/yvngsaid/Downloads/players_stats_by_season_full_details.csv"
players_stats = pd.read_csv(file_path)

# dataset filtering
nba_regular_season_data = players_stats[
    (players_stats['League'] == 'NBA') & (players_stats['Stage'] == 'Regular_Season')
]
#print(nba_regular_season_data.head())


most_regular_seasons_player = nba_regular_season_data['Player'].value_counts().idxmax()
most_regular_seasons_count = nba_regular_season_data['Player'].value_counts().max()

#print(f"The player who has played the most NBA regular seasons is {most_regular_seasons_player} with {most_regular_seasons_count} seasons.")

player_data = nba_regular_season_data[nba_regular_season_data['Player'] == most_regular_seasons_player]
player_data['Three_Point_Accuracy'] = (player_data['3PM'] / player_data['3PA']).fillna(0)

# columns to display
player_three_point_accuracy = player_data[['Season', 'Three_Point_Accuracy']]

# display 3pa result 
print(player_three_point_accuracy)

# getting the relevantq data
seasons = player_data['Season'].str.extract(r'(\d{4})').astype(int)
three_point_accuracy = player_data['Three_Point_Accuracy']

# linear regression
slope, intercept, r_value, p_value, std_err = linregress(seasons[0], three_point_accuracy)


line_of_best_fit = slope * seasons[0] + intercept
'''

# plotting the data and the line of best fit
plt.figure(figsize=(10, 6))
plt.scatter(seasons[0], three_point_accuracy, color='blue', label='Three-Point Accuracy')
plt.plot(seasons[0], line_of_best_fit, color='red', label='Line of Best Fit')
plt.xlabel('Season (Starting Year)')
plt.ylabel('Three-Point Accuracy')
plt.title(f'Three-Point Accuracy Over Seasons for {most_regular_seasons_player}')
plt.legend()
plt.grid(True)
plt.show()

'''

# defining the line of best fit function
def line_of_best_fit_func(x):
    return slope * x + intercept

# range of seasons (earliest to latest)
earliest_season = seasons[0].min()
latest_season = seasons[0].max()
integrated_value, _ = quad(line_of_best_fit_func, earliest_season, latest_season)

average_accuracy_integration = integrated_value / (latest_season - earliest_season)

# actual average 3 point accuracy
actual_average_accuracy = three_point_accuracy.mean()

#print(f"Average Three-Point Accuracy (Integration): {average_accuracy_integration:.4f}")
#print(f"Actual Average Three-Point Accuracy: {actual_average_accuracy:.4f}")


# relevant data for interpolation
seasons = player_data['Season'].str.extract(r'(\d{4})').astype(int)
three_point_accuracy = player_data['Three_Point_Accuracy']

# seasons that are missing
missing_seasons = [2002, 2015]

# interpolation
f_interpolate = interp1d(seasons[0], three_point_accuracy, kind='linear', fill_value='extrapolate')

# estimatr missing values using interpolation
estimated_values = f_interpolate(missing_seasons)

# add the missing values to the dataset
missing_data = pd.DataFrame({
    'Season': [f"{season} - {season + 1}" for season in missing_seasons],
    'Three_Point_Accuracy': estimated_values
})
player_data = pd.concat([player_data, missing_data], ignore_index=True)

# sort the dataset by season
player_data['Season_Year'] = player_data['Season'].str.extract(r'(\d{4})').astype(int)
player_data = player_data.sort_values(by='Season_Year').drop(columns=['Season_Year'])

#print(player_data[['Season', 'Three_Point_Accuracy']])

fgm_mean = player_data['FGM'].mean()
fgm_variance = player_data['FGM'].var()
fgm_skew = skew(player_data['FGM'], nan_policy='omit')
fgm_kurtosis = kurtosis(player_data['FGM'], nan_policy='omit')

fga_mean = player_data['FGA'].mean()
fga_variance = player_data['FGA'].var()
fga_skew = skew(player_data['FGA'], nan_policy='omit')
fga_kurtosis = kurtosis(player_data['FGA'], nan_policy='omit')

# statistics for FGM and FGA
#print(f"FGM - Mean: {fgm_mean:.2f}, Variance: {fgm_variance:.2f}, Skew: {fgm_skew:.2f}, Kurtosis: {fgm_kurtosis:.2f}")
#print(f"FGA - Mean: {fga_mean:.2f}, Variance: {fga_variance:.2f}, Skew: {fga_skew:.2f}, Kurtosis: {fga_kurtosis:.2f}")

# paired t test (related samples) between fgm and fga
t_stat_rel, p_value_rel = ttest_rel(player_data['FGM'], player_data['FGA'], nan_policy='omit')

# individual t tests for fgm and fga against a population mean of 0
t_stat_fgm, p_value_fgm = ttest_1samp(player_data['FGM'], 0, nan_policy='omit')
t_stat_fga, p_value_fga = ttest_1samp(player_data['FGA'], 0, nan_policy='omit')


print(f"Paired t-test (FGM vs FGA) - t-statistic: {t_stat_rel:.2f}, p-value: {p_value_rel:.4f}")
print(f"One-sample t-test (FGM) - t-statistic: {t_stat_fgm:.2f}, p-value: {p_value_fgm:.4f}")
print(f"One-sample t-test (FGA) - t-statistic: {t_stat_fga:.2f}, p-value: {p_value_fga:.4f}")