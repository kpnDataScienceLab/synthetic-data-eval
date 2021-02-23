# !pip install dython
from dython.nominal import associations
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
from numpy import asarray
from scipy.special import rel_entr
from scipy.spatial import distance
import logging
import os
from scipy.stats import ks_2samp


class eval_metrics():
    """The goal of the evaluation script is to measure how well the generated synthetic dataset preserves
    the characteristics that exist between the attributes in the original dataset. """

    def __init__(self, origdst, synthdst):

        self.origdst = origdst
        self.synthdst = synthdst

    @staticmethod
    def to_cat(dtr, dts):

        target_cols = list(dtr.columns[11:-3])
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
        target_columns.append(self.origdst.columns[3])  # genre

        js_dict = {}

        for col in target_columns:
            col_counts_orig = real_cat[col].value_counts(normalize=True).sort_index(ascending=True)
            col_counts_synth = synth_cat[col].value_counts(normalize=True).sort_index(ascending=True)

            js = distance.jensenshannon(asarray(col_counts_orig.tolist()), asarray(col_counts_synth.tolist()), base=2)

            js_dict[col] = js

        return js_dict

    def kl_divergence(self):

        """ This metric is also defined at the variable level and examines whether the distributions of the attributes are
        identical and measures the potential level of discrepancy between them.
        The threshold limit for this metric is a value below 2"""

        target_columns = list(self.origdst.columns[11:-3])
        target_columns.append(self.origdst.columns[4])  # content_id

        kl_dict = {}

        for col in target_columns:
            col_counts_orig = self.origdst[col].value_counts(normalize=True).sort_index(ascending=True)
            col_counts_synth = self.synthdst[col].value_counts(normalize=True).sort_index(ascending=True)

            kl = sum(rel_entr(col_counts_orig.tolist(), col_counts_synth.tolist()))

            kl_dict[col] = kl

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

        logging.basicConfig(filename='evaluation.log',
                            format='%(asctime)s %(message)s',
                            filemode='w')

        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        ob = eval_metrics(r, ra)

        # euclidean distance
        flag_eucl = False
        eucl, eumatr = ob.euclidean_dist()
        logger.info('Euclidean distance was calculated')
        print('The calculated euclidean distance is: ', eucl)
        print('The calculated euclidean distance matrix is:', eumatr)
        if eucl > 14:
            logger.error(f'The calculated Euclidean distance value between the two correlation matrices is too high it should be \
            less than 14. The current value is {eucl}')
            logger.info(f'The Euclidean distance matrix is \n {eumatr}')
        else:
            logger.info('The dataset satisfies the criteria for the euclidean distance.')
            logger.info(f'The calculated Euclidean distance value is \n {eucl}')
            logger.info(f'The Euclidean distance matrix is \n {eumatr}')
            flag_eucl = True
        logger.info('---------------------------------------------------------')

        # 2 sample Kolmogorov-Smirnov test
        kst = ob.kolmogorov()

        p_value = 0.05
        flag_klg = False
        logger.info('Kolmogorov-Smirnov test was performed')
        print('The results of the Kolmogorov-Smirnov test is:', kst)
        rejected = {}
        for col in kst:
            if kst[col]['p-value'] < p_value:
                rejected[col] = kst[col]
        if rejected:
            logger.info('The dataset did not pass the Kolmogorov-Smirnov test')
            logger.info(f'The columns that did not pass the test are \n {rejected}')
            logger.info(f'The overall performance for the test is \n {kst}')
        else:
            logger.info('The dataset passed the Kolmogorov-Smirnov test')
            logger.info(f'The overall performance for the test is \n {kst}')
            flag_klg = True
        logger.info('---------------------------------------------------------')

        # Jensen-Shannon Divergence
        dict_js = ob.jensen_shannon()
        logger.info('Jensen-Shannon Divergence was calculated')
        print('The result of the Jensen-Shannon Divergence is:', dict_js)
        flag_js = False

        jsd = deepcopy(dict_js)

        for key in list(dict_js):
            if (dict_js[key] < 0.50) & (key != 'CONTENT_ID'):
                del dict_js[key]
            if key == 'CONTENT_ID':
                if (dict_js[key] < 0.75):
                    del dict_js[key]

        if dict_js:
            logger.info('The dataset did not pass the Jensen-Shannon Divergence test')
            for key in dict_js.keys():
                logger.info(f'The Jensen-Shannon Divergence value for the column {key} was {dict_js[key]}')
            logger.info(f'The overall performance for each column is summarized below: \n {jsd}')
        else:
            logger.info('The dataset passed the Jensen-Shannon Divergence test')
            logger.info(f'The overall performance for each column is summarized below: \n {jsd}')
            flag_js = True
        logger.info('---------------------------------------------------------')

        # KL divergence
        dict_kl = ob.kl_divergence()
        logger.info('KL divergence was calculated')
        print('The result of the KL divergence is', dict_kl)
        flag_kl = False

        kl = deepcopy(dict_kl)

        for key in list(dict_kl):
            if dict_kl[key] < 2.20:
                del dict_kl[key]

        if dict_kl:
            logger.info('The dataset did not pass the KL divergence evaluation test')
            for key in dict_kl.keys():
                logger.info(f'The KL divergence value for the column {key} was {dict_kl[key]}')
            logger.info(f'The overall for the KL divergence performance for each column is summarized below: \n {kl}')
        else:
            logger.info('The dataset passed the KL divergence evaluation test')
            logger.info(f'The overall performance for the KL divergence for each column is summarized below: \n {kl}')
            flag_kl = True
        logger.info('---------------------------------------------------------')

        # pairwise correlation difference
        pair_corr_diff, pcd_matr = ob.pairwise_correlation_difference()
        logger.info('Pairwise correlation difference was calculated')
        print('The calculated Pairwise correlation difference was', pair_corr_diff)
        print('The calculated Pairwise correlation difference matrix was', pcd_matr)

        flag_pcd = False
        if pair_corr_diff > 2.4:
            logger.error(f'The calculated Euclidean distance value between the two correlation matrices is too high it should be \
            less than 14. The current value is {pair_corr_diff}')
            logger.info(f'The Pairwise distance distance matrix is \n {pcd_matr}')
        else:
            logger.info('The dataaset satisfies the criteria for the Pairwise Correlation Difference.')
            logger.info(f'The Pairwise distance distance value is \n {pair_corr_diff}')
            logger.info(f'The Pairwise distance distance matrix is \n {pcd_matr}')
            flag_pcd = True

        if (flag_eucl & flag_js & flag_klg & flag_kl & flag_pcd):
            logger.info('The dataaset satisfies the minimum evaluation criteria.')
        else:
            logger.info('The dataaset does not satisfy the minimum evaluation criteria.')
            logger.info('Plese check the previous log messages.')