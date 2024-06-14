import unittest
import numpy as np
from src.data.dataset import RankineData
from src.models.LETKF import LETKF

class TestLETKF(unittest.TestCase):
    def setUp(self):
        self.letkf = LETKF()

    def test_initialization(self):
        self.assertEqual(self.letkf.name, "LETKF")
        self.assertEqual(self.letkf.ensemble_size, 10)  # Replace with actual config value
        self.assertEqual(self.letkf.number_of_obs, 50)  # Replace with actual config value
        self.assertIsInstance(self.letkf.data, RankineData)

    def test_localizescale(self):
        sigma = 1.0
        distances = np.array([0, 0.5, 1.0, 1.5, 2.0])
        expected = np.array([1.0, np.exp(-0.5**2 / (2 * sigma**2)), 0, 0, 0])
        np.testing.assert_array_almost_equal(self.letkf.localizescale(sigma, distances), expected)

    def test_make_localization_matrix(self):
        sigma = 5.0
        self.letkf.make_localization_matrix(sigma)
        self.assertIsNotNone(self.letkf.L_loc)
        self.assertEqual(self.letkf.L_loc.shape, (91*91, 91*91))

    def test_make_observation_matrix(self):
        self.letkf.make_observation_matrix()
        self.assertIsNotNone(self.letkf.H)
        self.assertEqual(self.letkf.H.shape, (self.letkf.number_of_obs, 91*91))

    def test_flatten_obs_points(self):
        flatten_obs_points = self.letkf.flatten_obs_points()
        self.assertEqual(len(flatten_obs_points), self.letkf.number_of_obs)
        self.assertTrue(all(isinstance(i, int) for i in flatten_obs_points))

    def test_sort_obs(self):
        sorted_obs_points, sorted_obs_u, sorted_obs_v = self.letkf.sort_obs()
        self.assertEqual(len(sorted_obs_points), self.letkf.number_of_obs)
        self.assertEqual(len(sorted_obs_u), self.letkf.number_of_obs)
        self.assertEqual(len(sorted_obs_v), self.letkf.number_of_obs)

    def test_calculate_R_inv(self):
        self.letkf.calculate_R_inv()
        R_inv = np.load(f"{self.letkf.data.PATH}/data/R_inv.npy")
        self.assertEqual(R_inv.shape, (91*91, 91*91))
        np.testing.assert_array_almost_equal(R_inv, np.eye(91*91) * (1/3**2))

    def test_preparation_for_analysis(self):
        self.letkf.make_observation_matrix()
        self.letkf.calculate_R_inv()
        xb_mean_u, xb_mean_v, dXb_u, dXb_v, dYb_u, dYb_v, R_inv = self.letkf.preparation_for_analysis()
        self.assertEqual(xb_mean_u.shape, (91*91,))
        self.assertEqual(xb_mean_v.shape, (91*91,))
        self.assertEqual(dXb_u.shape, (91*91, self.letkf.ensemble_size))
        self.assertEqual(dXb_v.shape, (91*91, self.letkf.ensemble_size))
        self.assertEqual(dYb_u.shape, (self.letkf.number_of_obs, self.letkf.ensemble_size))
        self.assertEqual(dYb_v.shape, (self.letkf.number_of_obs, self.letkf.ensemble_size))

    def test_parallel_analysis(self):
        self.letkf.make_observation_matrix()
        self.letkf.make_localization_matrix(5)
        p_range = range(0, 10)
        delta = 0.1
        xa_u, xa_v = self.letkf.parallel_analysis(p_range, delta)
        self.assertEqual(xa_u.shape, (self.letkf.ensemble_size, len(p_range)))
        self.assertEqual(xa_v.shape, (self.letkf.ensemble_size, len(p_range)))

    def test_save_data(self):
        data = {'u_x': np.zeros((5, 5)), 'u_y': np.ones((5, 5))}
        filename = 'test_save'
        self.letkf.save_data(filename, data)
        loaded_data = np.load(f"{self.letkf.data.PATH}/results/analysis/{filename}.npz")
        np.testing.assert_array_equal(loaded_data['u_x'], data['u_x'])
        np.testing.assert_array_equal(loaded_data['u_y'], data['u_y'])

if __name__ == "__main__":
    unittest.main()
