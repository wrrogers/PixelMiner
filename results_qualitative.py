import numpy as np
import pandas as pd
from scipy.stats import binom_test

data = pd.read_csv(r'C:\Users\william\OneDrive\Desktop\Second Paper\Code\qualitative_results.csv')

total = len(data)

total_pm = len(data[data.Chosen == 'PixelCNN'])

print()
print('Overall PixelMiner percent:', str(total_pm/total))

chosen_ln = len(data[data.Chosen == 'Linear'])
chosen_not_ln = len(data[data.Not_Chosen == 'Linear'])

chosen_bs = len(data[data.Chosen == 'BSpline'])
chosen_not_bs = len(data[data.Not_Chosen == 'BSpline'])

chosen_nn = len(data[data.Chosen == 'NearestNeighbor'])
chosen_not_nn = len(data[data.Not_Chosen == 'NearestNeighbor'])

chosen_ws = len(data[data.Chosen == 'CosineWindowedSinc'])
chosen_not_ws = len(data[data.Not_Chosen == 'CosineWindowedSinc'])


print('Percent not Linear:', 1- (chosen_ln / (chosen_ln + chosen_not_ln)))
print('Percent not BSpline:', 1- (chosen_bs / (chosen_bs + chosen_not_bs)))
print('Percent not Nearest:', 1- (chosen_nn / (chosen_nn + chosen_not_nn)))
print('Percent not CosineWindowedSinc:', 1-(chosen_ws / (chosen_ws + chosen_not_ws)))

print()
print('Overall:', binom_test(total_pm, total))
print('Linear:', binom_test(chosen_ln, chosen_ln + chosen_not_ln))
print('BSpline:', binom_test(chosen_bs, chosen_bs + chosen_not_bs))
print('Nearest:', binom_test(chosen_nn, chosen_nn + chosen_not_nn))
print('Win Sinc:', binom_test(chosen_ws, chosen_ws + chosen_not_ws))
