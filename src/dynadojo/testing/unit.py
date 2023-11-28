import unittest

import numpy as np
import pandas as pd
import pandas.testing as pd_testing

# baselines
from ..baselines.aug_ode import AugODE
from ..baselines.cnn import CNN
from ..baselines.dmd import DMD
from ..baselines.dnn import DNN
from ..baselines.ode import ODE
from ..baselines.sindy import SINDy

ALL_BASELINES = [
    AugODE,
    CNN,
    DMD,
    DNN,
    ODE,
    SINDy
]

# systems
from ..systems.ca import CASystem
from ..systems.ctln import CTLNSystem
from ..systems.heat import HeatEquation
from ..systems.kuramoto import KuramotoSystem
from ..systems.lds import LDSystem
from ..systems.santi import NBodySystem
from ..systems.snn import SNNSystem
from ..systems.epidemic import SEISSystem, SISSystem, SIRSystem
from ..systems.fbsnn_pde import BSBSystem, HJBSystem
from ..systems.lv import CompetitiveLVSystem, PreyPredatorSystem
from ..systems.opinion import ARWHKSystem, DeffuantSystem, HKSystem, MediaBiasSystem, WHKSystem

ALL_SYSTEMS = [
    CASystem,
    CTLNSystem,
    HeatEquation,
    KuramotoSystem,
    LDSystem,
    NBodySystem,
    SNNSystem,
    SEISSystem, SISSystem, SIRSystem,
    BSBSystem, HJBSystem,
    CompetitiveLVSystem, PreyPredatorSystem,
    ARWHKSystem, DeffuantSystem, HKSystem, MediaBiasSystem, WHKSystem
]

systems = ALL_SYSTEMS  # To test multiple systems, add them to this list
models = ALL_BASELINES  # To test multiple models, add them to this list


class TestReproducibility(unittest.TestCase):
    def test_make_init_cond(self):
        for system in systems:
            with self.subTest(system):
                s1 = system(seed=100)
                s2 = system(seed=100)
                i1 = s1.make_init_conds(n=5)
                i2 = s2.make_init_conds(n=5)
                np.testing.assert_array_equal(i1, i2)

    def test_make_data(self):
        for system in systems:
            with self.subTest(system):
                s1 = system(seed=100)
                i1 = s1.make_init_conds(n=5)
                d1 = s1.make_data(i1, timesteps=3, noisy=True)

                s2 = system(seed=100)
                i2 = s2.make_init_conds(n=5)
                d2 = s2.make_data(i2, timesteps=3, noisy=True)
                np.testing.assert_array_equal(d1, d2)

    def test_make_initial_conditions_less_more(self):
        """
        Test that if we make data with more initial conditions, 
        the data starts the same as when we made data with less initial conditions
        """
        for system in systems:
            with self.subTest(system):
                s1 = system(seed=100)
                i1 = s1.make_init_conds(n=5)

                s2 = system(seed=100)
                i2 = s2.make_init_conds(n=10)
                np.testing.assert_array_equal(i1, i2[:5])

    # TODO: Reproducibility test w/ control

    # TODO: Improve reproducibility of data generation
    # def test_make_data_less_more(self):
    #     """
    #     Test that if we make data with more initial conditions, 
    #     the data starts the same as when we made data with less initial conditions
    #     """
    #     for system in systems:
    #         with self.subTest(system):
    #             s1 = system(seed=100)
    #             i1 = s1.make_init_conds_wrapper(n=5)
    #             d1 = s1.make_data_wrapper(i1, timesteps=3, noisy=True)

    #             s2 = system(seed=100)
    #             i2 = s2.make_init_conds_wrapper(n=10)
    #             d2 = s2.make_data_wrapper(i2, timesteps=3, noisy=True)
    #             np.testing.assert_array_equal(d1, d2[:5])


class TestReproducibilityModel(unittest.TestCase):
    def assertDataframeEqual(self, a, b, msg):
        try:
            pd_testing.assert_frame_equal(a, b)
        except AssertionError as e:
            raise self.failureException(msg) from e

    def setUp(self):
        self.addTypeEqualityFunc(pd.DataFrame, self.assertDataframeEqual)

    def test_with_fc(self):
        for model in models:
            with self.subTest(model):
                challenge = FixedComplexity(N=[4], l=4, t=10,
                                            system_cls=LDSystem, reps=1,
                                            test_examples=2, test_timesteps=2, verbose=False)
                df1 = challenge.evaluate(model, seed=100, noisy=True,
                                         reps_filter=None,
                                         L_filter=None,
                                         algo_kwargs=None)
                df2 = challenge.evaluate(model, seed=100, noisy=True,
                                         reps_filter=None,
                                         L_filter=None,
                                         algo_kwargs=None)
                cols = ['rep', 'latent_dim', 'embed_dim', 'timesteps', 'n', 'error', 'ood_error', 'total_cost',
                        'system_seed', 'model_seed']
                df1 = df1[cols]
                df2 = df2[cols]
                self.assertEqual(df1, df2)

    def test_with_reps(self):
        """
        Test that running a single rep gives the same results as running multiple reps,
        when using filters to select a single rep
        """
        for model in models:
            with self.subTest(model):
                challenge = FixedComplexity(N=[2], l=2, t=3,
                                            system_cls=LDSystem, reps=2,
                                            test_examples=2, test_timesteps=2, verbose=False)
                df1 = challenge.evaluate(model, seed=100, noisy=True,
                                         reps_filter=[1],
                                         L_filter=None,
                                         algo_kwargs=None)
                df2 = challenge.evaluate(model, seed=100, noisy=True,
                                         reps_filter=None,
                                         L_filter=None,
                                         algo_kwargs=None)
                cols = ['rep', 'latent_dim', 'embed_dim', 'timesteps', 'n', 'error', 'ood_error', 'total_cost',
                        'system_seed', 'model_seed']
                df1 = df1[cols]
                df2 = df2[cols].loc[df2['rep'] == 1]
                self.assertEqual(df1, df2)

    """
    def test_dnn(self):
        l = 10
        e = 10 
        t = 50
        n = 100
        ood = True
        system_seed = 1276400239
        model_seed = 2949232388
        error = 11.755511444519389
        ood_error = 40.871378000678895

        challenge = FixedComplexity(N=[n], l=l, e=e, t=t, reps=1,
            test_examples=50, test_timesteps=50, verbose=False, 
            system_cls=LDSystem, system_kwargs={'seed':system_seed})

        df1 = challenge.evaluate(DNN, noisy=True, ood=ood,
                        reps_filter=None, 
                        L_filter = None,
                        model_kwargs={'seed':model_seed})

        #check that first row 'error' and 'ood_error' are correct
        self.assertEqual(df1['error'].iloc[0], error)
        self.assertEqual(df1['ood_error'].iloc[0], ood_error)

    def test_lr(self):
        l = 10
        e = 10 
        t = 50
        n = 100
        ood = True
        system_seed = 2940660954
        model_seed = 433949338
        error = 1.1618431384700412e-07
        ood_error = 1.1343960080211431e-07

        challenge = FixedTrainSize(n=n, L=[l], E=None, t=t, reps=1,
            test_examples=50, test_timesteps=50, verbose=True, 
            max_control_cost_per_dim=1, control_horizons=0,
            system_cls=LDSystem, system_kwargs={'seed':system_seed})

        df1 = challenge.evaluate(LinearRegression, noisy=True, ood=ood,
                        reps_filter=None, 
                        L_filter = None,
                        model_kwargs={'seed':model_seed})

        #check that first row 'error' and 'ood_error' are correct
        self.assertEqual(df1['error'].iloc[0], error)
        self.assertEqual(df1['ood_error'].iloc[0], ood_error)
    """


if __name__ == '__main__':
    unittest.main()
