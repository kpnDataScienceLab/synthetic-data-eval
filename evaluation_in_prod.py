from dython.nominal import associations
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
from numpy import asarray
from scipy.special import rel_entr
from scipy.spatial import distance
import numpy as np
import boto3
import pickle
import io
import os
import copy
from scipy.stats import ks_2samp
from datetime import datetime

class eval_metrics():
    """The goal of the evaluation script is to measure how well the generated synthetic dataset preserves
    the characteristics that exist between the attributes in the original dataset. """

    def __init__(self, origdst, synthdst):

        self.origdst = origdst
        self.synthdst = synthdst

    @staticmethod
    def to_cat(dtr, dts):

        target_cols = list(dtr.columns[11:-3])
        target_cols.insert(0, dtr.columns[1])  # channel
        target_cols.insert(0, dtr.columns[2])  # program_title
        target_cols.insert(0, dtr.columns[3])  # genre

        #         flag_same_demographic_column_values = True

        for col in target_cols:

            assigned_categories_real = dtr[col].astype('category')
            assigned_categories_synthetic = dts[col].astype('category')

            categories_real_dict = dict(enumerate(assigned_categories_real.cat.categories))
            categories_real_synthetic = dict(enumerate(assigned_categories_synthetic.cat.categories))

            if (categories_real_dict == categories_real_synthetic):
                print('For the column ', col, ' the assigned categories are the same for both datasets')
                print('================')
            else:
                for key in categories_real_dict.keys():
                    if key not in categories_real_synthetic.keys():
                        print('The value ', key, ' was not found in column ', col)
                    #                         flag_same_demographic_column_values = False
                    else:
                        categories_real_synthetic[key] = categories_real_dict[key]

            dtr[col] = assigned_categories_real.cat.codes
            dts[col] = assigned_categories_synthetic.cat.codes

        return dtr, dts

    @staticmethod
    def get_demographics(df):

        df = df[['CONTENT_ID', 'demographic_car_number_of_cars', 'demographic_age_of_the_eldest_child',
                 'demographic_home_ownership', 'demographic_income',
                 'demographic_education', 'demographic_household_composition',
                 'demographic_number_of_people', 'demographic_age']]

        return df

    def euclidean_dist(self):

        """ This metric measures the preservation of intrinsic patterns occurring between the attributes
        of the original dataset in the corresponding synthetic dataset. The lower the value is the better the data generation
        tool preserves the patterns.
        The threshold limit for this metric is a value below 14."""

        real_cat, synth_cat = self.to_cat(self.origdst, self.synthdst)

        real_cat_dem = self.get_demographics(real_cat)
        synth_cat_dem = self.get_demographics(synth_cat)

        corr_real_obj = associations(real_cat_dem, theil_u=True, bias_correction=False, plot=False)
        corr_synth_obj = associations(synth_cat_dem, theil_u=True, bias_correction=False, plot=False)

        corr_real = corr_real_obj['corr']
        corr_rand = corr_synth_obj['corr']

        eucl_matr = distance.cdist(corr_real, corr_rand, 'euclidean')

        eucl = LA.norm(eucl_matr)

        return eucl, eucl_matr

    def kolmogorov(self):

        """ The two-sample Kolmogorov-Smirnov test is used to test whether two samples come from the same distribution.
        The level of significance a is set as a = 0.05. If the generated p-value from the test is lower than a then it is
        probable that the two distributions are different.
        The threshold limit for this function is a list containing less than 10 elements"""

        real_cat, synth_cat = self.to_cat(self.origdst, self.synthdst)

        real_cat = real_cat[
            real_cat['iab_category_Family and Relationships'].notnull() & real_cat['iab_category_Travel'].notnull()]
        synth_cat = synth_cat[
            synth_cat['iab_category_Family and Relationships'].notnull() & synth_cat['iab_category_Travel'].notnull()]

        target_cols = list(real_cat.columns[11:-1])

        sample_real = real_cat[target_cols].reset_index(drop=True)
        sample_synth = synth_cat[target_cols].reset_index(drop=True)

        cols = {}
        for col in range(10):
            test = ks_2samp(sample_real.iloc[:, col], sample_synth.iloc[:, col])
            col_name = target_cols[col]
            cols[col_name] = {'statistic': test[0], 'p-value': test[1]}

        return cols

    def jensen_shannon(self):

        real_cat, synth_cat = self.to_cat(self.origdst, self.synthdst)

        target_columns = list(self.origdst.columns[11:-3])
        target_columns.append(self.origdst.columns[1])  # channel
        target_columns.append(self.origdst.columns[2])  # program_title
        target_columns.append(self.origdst.columns[3])  # genre

        js_dict = {}

        for col in target_columns:

            try:
                col_counts_orig = real_cat[col].value_counts(normalize=True).sort_index(ascending=True)
                col_counts_synth = synth_cat[col].value_counts(normalize=True).sort_index(ascending=True)

                js = distance.jensenshannon(asarray(col_counts_orig.tolist()), asarray(col_counts_synth.tolist()),
                                            base=2)

                js_dict[col] = js

            except:

                print('For the column ', col, ' you must generate the same unique values as the real dataset.')
                print('The number of unique values than you should generate for column ', col, 'is ',
                      len(self.origdst[col].unique()))

        return js_dict

    def kl_divergence(self):

        """ This metric is also defined at the variable level and examines whether the distributions of the attributes are
        identical and measures the potential level of discrepancy between them.
        The threshold limit for this metric is a value below 2"""

        target_columns = list(self.origdst.columns[11:-3])
        target_columns.append(self.origdst.columns[1])  # channel
        target_columns.append(self.origdst.columns[2])  # program_title
        target_columns.append(self.origdst.columns[3])  # genre

        kl_dict = {}

        for col in target_columns:

            try:

                col_counts_orig = self.origdst[col].value_counts(normalize=True).sort_index(ascending=True)
                col_counts_synth = self.synthdst[col].value_counts(normalize=True).sort_index(ascending=True)

                kl = sum(rel_entr(col_counts_orig.tolist(), col_counts_synth.tolist()))

                kl_dict[col] = kl

            except:

                print('For the column ', col, ' you must generate the same unique values as the real dataset.')
                print('The number of unique values than you should generate for column ', col, 'is ',
                      len(self.origdst[col].unique()))

        return kl_dict

    def pairwise_correlation_difference(self):

        """ PCD measures the difference in terms of Frobenius norm of the correlation matrices computed from real and synthetic
        datasets. The smaller the PCD, the closer the synthetic data is to the real data in terms of linear correlations across
        the variables.
        The threshold limit for this metric is a value below 2.4 """

        real_cat, synth_cat = self.to_cat(self.origdst, self.synthdst)

        real_cat_dem = self.get_demographics(real_cat)
        synth_cat_dem = self.get_demographics(synth_cat)

        corr_real_obj = associations(real_cat_dem, theil_u=True, bias_correction=False, plot=False)
        corr_synth_obj = associations(synth_cat_dem, theil_u=True, bias_correction=False, plot=False)

        corr_real = corr_real_obj['corr']
        corr_rand = corr_synth_obj['corr']

        substract_m = np.subtract(corr_real, corr_rand)
        prwcrdst = LA.norm(substract_m)

        return prwcrdst, substract_m

if __name__ == "__main__":
    # Set up access to s3
    s3 = boto3.resource('s3')
    data_bucket = os.environ['data_bucket']
    vendor_bucket = os.environ['vendor_bucket']
    og_file_key = os.environ['og_file_key']
    syn_file_key = os.environ['syn_file_key']

    now = datetime.now()
    current_time = now.strftime("%d-%m-%Y_%H-%M-%S")
    log_file_name = 'Evaluation_log_of_' + syn_file_key + '_' + current_time + '.txt'
    log_file = open(log_file_name, 'a')

    # Read in the original data from s3
    try:
        df_og = pickle.loads(s3.Bucket(data_bucket).Object(og_file_key).get()['Body'].read())
        print("Succeeded in loading original data from data bucket")
        log_file.write("Succeeded in loading original data from data bucket\n")
    except:
        print("Error loading original data from data bucket")
        log_file.write("Error loading original data from data bucket\n")

    # Read in the synthetic data from s3
    try:
        df_syn = pickle.loads(s3.Bucket(vendor_bucket).Object(syn_file_key).get()['Body'].read())
        print("Succeeded in loading synthetic data from vendor bucket")
        log_file.write("Succeeded in loading synthetic data from vendor bucket\n")
    except:
        print("Error loading synthetic data from vendor bucket")
        log_file.write("Error loading synthetic data from vendor bucket\n")

    print("Evaluating synthetic data...")
    log_file.write("Evaluating synthetic data...\n")

    ob = eval_metrics(df_og, df_syn)

    # euclidean distance
    flag_eucl = False
    eucl, eumatr = ob.euclidean_dist()
    print('Euclidean distance was calculated')
    log_file.write('Euclidean distance was calculated\n')
    print('The calculated euclidean distance is: ', eucl)
    log_file.write(f'The calculated euclidean distance is: {eucl}\n')
    print('The calculated euclidean distance matrix is:')
    print(eumatr)
    log_file.write('The calculated euclidean distance matrix is:\n')
    log_file.write(f'{eumatr}\n')
    if eucl > 14:
        print(f'The calculated Euclidean distance value between the two correlation matrices is too high it should be less than 14. The current value is: {eucl}')
        log_file.write(f'The calculated Euclidean distance value between the two correlation matrices is too high it should be less than 14. The current value is: {eucl}\n')
    else:
        print('The dataset satisfies the criteria for the euclidean distance.')
        log_file.write('The dataset satisfies the criteria for the euclidean distance.\n')
        print(f'The calculated Euclidean distance value is: {eucl}')
        log_file.write(f'The calculated Euclidean distance value is: {eucl}\n')
        flag_eucl = True
    print('---------------------------------------------------------')
    log_file.write('---------------------------------------------------------\n')

    # 2 sample Kolmogorov-Smirnov test
    kst = ob.kolmogorov()

    p_value = 0.05
    flag_klg = False
    print('Kolmogorov-Smirnov test was performed')
    log_file.write('Kolmogorov-Smirnov test was performed\n')
    print('The results of the Kolmogorov-Smirnov test is: ', kst)
    log_file.write('The results of the Kolmogorov-Smirnov test is:\n')
    log_file.write(f'{kst}\n')
    rejected = {}
    for col in kst:
        if kst[col]['p-value'] < p_value:
            rejected[col] = kst[col]
    if rejected:
        print('The dataset did not pass the Kolmogorov-Smirnov test')
        log_file.write('The dataset did not pass the Kolmogorov-Smirnov test\n')
        print(f'The columns that did not pass the test are:\n{rejected}')
        log_file.write(f'The columns that did not pass the test are:\n{rejected}\n')
    else:
        print('The dataset passed the Kolmogorov-Smirnov test')
        log_file.write('The dataset passed the Kolmogorov-Smirnov test\n')
        flag_klg = True
    print('---------------------------------------------------------')
    log_file.write('---------------------------------------------------------\n')

    # Jensen-Shannon Divergence
    dict_js = ob.jensen_shannon()
    print('Jensen-Shannon Divergence was calculated')
    log_file.write('Jensen-Shannon Divergence was calculated\n')
    print('The result of the Jensen-Shannon Divergence is: ', dict_js)
    log_file.write(f'The result of the Jensen-Shannon Divergence is:\n{dict_js}\n')
    flag_js = False

    jsd = copy.deepcopy(dict_js)

    for key in list(dict_js):
        if (dict_js[key] < 0.50) & (key not in ['GENRE', 'PROGRAM_TITLE']):
            del dict_js[key]
        if key == 'GENRE':
            if (dict_js[key] < 0.59):
                del dict_js[key]
        if key == 'PROGRAM_TITLE':
            if (dict_js[key] < 0.69):
                del dict_js[key]

    if dict_js:
        print('The dataset did not pass the Jensen-Shannon Divergence test')
        log_file.write('The dataset did not pass the Jensen-Shannon Divergence test\n')
        for key in dict_js.keys():
            print(f'The Jensen-Shannon Divergence value for the column {key} was {dict_js[key]}')
            log_file.write(f'The Jensen-Shannon Divergence value for the column {key} was {dict_js[key]}\n')
    else:
        print('The dataset passed the Jensen-Shannon Divergence test')
        log_file.write('The dataset passed the Jensen-Shannon Divergence test\n')
        flag_js = True
    print('---------------------------------------------------------')
    log_file.write('---------------------------------------------------------\n')

    # KL divergence
    dict_kl = ob.kl_divergence()
    print('KL divergence was calculated')
    log_file.write('KL divergence was calculated\n')
    print('The result of the KL divergence is:\n', dict_kl)
    log_file.write(f'The result of the KL divergence is:\n{dict_kl}\n')
    flag_kl = False

    kl = copy.deepcopy(dict_kl)

    for key in list(dict_kl):
        if dict_kl[key] < 2.20:
            del dict_kl[key]

    if dict_kl:
        print('The dataset did not pass the KL divergence evaluation test')
        log_file.write('The dataset did not pass the KL divergence evaluation test\n')
        for key in dict_kl.keys():
            print(f'The KL divergence value for the column {key} was {dict_kl[key]}')
            log_file.write(f'The KL divergence value for the column {key} was {dict_kl[key]}\n')
    else:
        print('The dataset passed the KL divergence evaluation test')
        log_file.write('The dataset passed the KL divergence evaluation test\n')
        flag_kl = True
    print('---------------------------------------------------------')
    log_file.write('---------------------------------------------------------\n')

    # pairwise correlation difference
    pair_corr_diff, pcd_matr = ob.pairwise_correlation_difference()
    print('Pairwise correlation difference was calculated')
    log_file.write('Pairwise correlation difference was calculated\n')
    print('The calculated Pairwise correlation difference was: ', pair_corr_diff)
    log_file.write(f'The calculated Pairwise correlation difference was: {pair_corr_diff}\n')
    print('The calculated Pairwise correlation difference matrix was: ', pcd_matr)
    log_file.write(f'The calculated Pairwise correlation difference matrix was:\n{pcd_matr}\n')

    flag_pcd = False
    if pair_corr_diff > 2.4:
        print(f'The calculated Pairwise correlation difference value between the two correlation matrices is too high it should be less than 2.4\n')
        log_file.write(f'The calculated Pairwise correlation difference value between the two correlation matrices is too high it should be less than 2.4\n')
    else:
        print('The dataset satisfies the criteria for the Pairwise Correlation Difference < 2.4')
        log_file.write('The dataset satisfies the criteria for the Pairwise Correlation Difference < 2.4\n')
        flag_pcd = True
    print('---------------------------------------------------------')
    log_file.write('---------------------------------------------------------\n')

    if (flag_eucl & flag_js & flag_klg & flag_kl & flag_pcd):
        print('The dataset satisfies the minimum evaluation criteria.')
        log_file.write('The dataset satisfies the minimum evaluation criteria.\n')
        log_file.close()
        s3.meta.client.upload_file(log_file_name, vendor_bucket, log_file_name)
    else:
        print('The dataset does not satisfy the minimum evaluation criteria.')
        log_file.write('The dataset does not satisfy the minimum evaluation criteria.\n')
        print('Please check the previous log messages.')
        log_file.write('Please check the previous log messages.\n')
        log_file.close()
        s3.meta.client.upload_file(log_file_name, vendor_bucket, log_file_name)