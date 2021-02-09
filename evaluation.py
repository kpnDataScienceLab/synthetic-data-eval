# !pip install dython
from dython.nominal import associations
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from numpy import linalg as LA
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
    def to_cat(dtf):
        for col in list(dtf.columns[11:-3]):
            if type(dtf[col][0]) == str:
                dtf[col] = dtf[col].astype('category').cat.codes

        return dtf

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

        real_cat = self.to_cat(self.origdst)
        synth_cat = self.to_cat(self.synthdst)

        real_cat_dem = self.get_demographics(real_cat)
        synth_cat_dem = self.get_demographics(synth_cat)

        corr_real_obj = associations(real_cat_dem, theil_u=True, bias_correction=False, plot=False)
        corr_synth_obj = associations(synth_cat_dem, theil_u=True, bias_correction=False, plot=False)

        corr_real = corr_real_obj['corr']
        corr_rand = corr_synth_obj['corr']

        eucl_matr = distance.cdist(corr_real, corr_rand, 'euclidean')

        eucl = LA.norm(eucl_matr)

        return eucl

    def kolmogorov(self):

        """ The two-sample Kolmogorov-Smirnov test is used to test whether two samples come from the same distribution.
        The level of significance a is set as a = 0.05. If the generated p-value from the test is lower than a then it is
        probable that the two distributions are different.
        The threshold limit for this function is a list containing less than 10 elements"""

        real_cat = self.to_cat(self.origdst)
        synth_cat = self.to_cat(self.synthdst)

        real_cat = real_cat[
            real_cat['iab_category_Family and Relationships'].notnull() & real_cat['iab_category_Travel'].notnull()]
        synth_cat = synth_cat[
            synth_cat['iab_category_Family and Relationships'].notnull() & synth_cat['iab_category_Travel'].notnull()]

        target_cols = list(real_cat.columns[11:-1])

        sample_real = real_cat[target_cols].reset_index(drop=True)
        sample_synth = synth_cat[target_cols].reset_index(drop=True)

        p_value = 0.05
        rejected = []
        for col in range(10):
            test = ks_2samp(sample_real.iloc[:, col], sample_synth.iloc[:, col])
            if test[1] < p_value:
                rejected.append(target_cols[col])

        return rejected

    def kl_divergence(self):

        """ This metric is also defined at the variable level and examines whether the distributions of the attributes are
        identical and measures the potential level of discrepancy between them.
        The threshold limit for this metric is a value below 2"""

        target_columns = self.origdst.columns[11:-3]

        kl_dict = {}

        for col in target_columns:

            col_counts_orig = self.origdst[col].value_counts()
            col_counts_synth = self.synthdst[col].value_counts()

            for i, k in col_counts_orig.items():
                col_counts_orig[i] = k / col_counts_orig.sum()
            for i, k in col_counts_synth.items():
                col_counts_synth[i] = k / col_counts_synth.sum()

            kl = sum(rel_entr(col_counts_orig.tolist(), col_counts_synth.tolist()))

            kl_dict[col] = kl

            for key in list(kl_dict):
                if kl_dict[key] < 2:
                    del kl_dict[key]

        return kl_dict

    def pairwise_correlation_difference(self):

        """ PCD measures the difference in terms of Frobenius norm of the correlation matrices computed from real and synthetic
        datasets. The smaller the PCD, the closer the synthetic data is to the real data in terms of linear correlations across
        the variables.
        The threshold limit for this metric is a value below 2.4 """

        real_cat = self.to_cat(self.origdst)
        synth_cat = self.to_cat(self.synthdst)

        real_cat_dem = self.get_demographics(real_cat)
        synth_cat_dem = self.get_demographics(synth_cat)

        corr_real_obj = associations(real_cat_dem, theil_u=True, bias_correction=False, plot=False)
        corr_synth_obj = associations(synth_cat_dem, theil_u=True, bias_correction=False, plot=False)

        corr_real = corr_real_obj['corr']
        corr_rand = corr_synth_obj['corr']

        substract_m = np.subtract(corr_real, corr_rand)
        prwcrdst = LA.norm(substract_m)

        return prwcrdst


if __name__ == "__main__":

    logging.basicConfig(filename='evaluation.log',
                        format='%(asctime)s %(message)s',
                        filemode='w')

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ob = eval_metrics(real, random)

    # euclidean distance
    flag_eucl = False
    eucl = ob.euclidean_dist()
    print(eucl)
    logger.info('Euclidean distance calculated')
    if eucl > 14:
        logger.error(f'The calculated Euclidean distance value between the two correlation matrices is too high it should be \
        less than 14. The current value is {eucl}')
    else:
        logger.info('The dataaset satisfies the criteria for the euclidean distance.')
        flag_eucl = True
    logger.info('---------------------------------------------------------')

    # 2 sample Kolmogorov-Smirnov test
    kst = ob.kolmogorov()
    flag_klg = False
    print(kst)
    logger.info('Kolmogorov-Smirnov test performed')
    if kst:
        logger.info('The dataset did not pass the Kolmogorov-Smirnov test')
        logger.info(f'The columns that did not pass the test are {kst}')
    else:
        logger.info('The dataset passed the Kolmogorov-Smirnov test')
        flag_klg = True
    logger.info('---------------------------------------------------------')

    # KL divergence
    dict_kl = ob.kl_divergence()
    flag_kl = False
    print(dict_kl)
    logger.info('KL divergence calculated')
    if dict_kl:
        logger.info('The dataset did not pass the KL divergence evaluation test')
        for key in dict_kl.keys():
            logger.info(f'The KL divergence value for the column {key} was {dict_kl[key]}')
    else:
        logger.info('The dataset passed the KL divergence evaluation test')
        flag_kl = True
    logger.info('---------------------------------------------------------')

    # pairwise correlation difference
    pair_corr_diff = ob.pairwise_correlation_difference()
    flag_pcd = False
    print(pair_corr_diff)
    logger.info('Pairwise correlation difference calculated')
    if pair_corr_diff > 2.4:
        logger.error(f'The calculated Euclidean distance value between the two correlation matrices is too high it should be \
        less than 14. The current value is {pair_corr_diff}')
    else:
        logger.info('The dataaset satisfies the criteria for the Pairwise Correlation Difference.')
        flag_pcd = True

    if (flag_eucl & flag_klg & flag_kl & flag_pcd):
        logger.info('The dataaset satisfies the minimum evaluation criteria.')
    else:
        logger.info('The dataaset does not satisfy the minimum evaluation criteria.')
        logger.info('Plese check the previous log messages.')
