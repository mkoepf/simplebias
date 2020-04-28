from pandas import read_csv, DataFrame
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from numpy import linspace


def train_model(data: DataFrame, fit_intercept: bool, title=""):
    x = data['len_commute'].to_numpy().reshape(-1, 1)
    y = data['prob_rejection']

    clf = LinearRegression(fit_intercept=fit_intercept).fit(x, y)

    x_coords = linspace(0, 10, 100)
    model_curve = x_coords * clf.coef_ + clf.intercept_

    type(data)
    data.plot(kind='scatter', x='len_commute', y='prob_rejection', ylim=(0, 1))
    plt.plot(x_coords, model_curve)
    plt.title(title)
    plt.show()


applicant_data_biased: DataFrame = read_csv('lin_data_bias.csv')
applicant_data_unbiased: DataFrame = read_csv('lin_data_no_bias.csv')

applicant_data_biased.plot(kind='scatter', x='len_commute', y='prob_rejection', title='biased data', ylim=(0, 1))
applicant_data_unbiased.plot(kind='scatter', x='len_commute', y='prob_rejection', title='unbiased data', ylim=(0, 1))

train_model(applicant_data_biased, False, 'model without intercept trained with biased data')
train_model(applicant_data_biased, True, 'model with intercept trained with biased data')
train_model(applicant_data_unbiased, False, 'model without intercept trained with unbiased data')
train_model(applicant_data_unbiased, True, 'model with intercept trained with unbiased data')
