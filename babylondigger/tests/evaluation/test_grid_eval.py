import unittest
from evaluation.grid_eval import __merge_dicts as merge, __check_is_grid_valid as check_grid, \
    __take_all_key_path as take_path

# Not valid: param1's list elements are diferent type (epochs, batch_size)
grid_dict_1 = {"param1": [{"epochs": 100}, {"batch_size": 8}],
               "param2": [{"nn_config": {"general": {"optimizer": {"type": "RMSProp"}}}}]}

# Not valid" param2 and param3 have intersection
grid_dict_2 = {"param1": [{"epochs": 100}], "param2": [{"nn_config": {"general": {"optimizer": {"type": "GD"}}}}],
               "param3": [{"nn_config": {"general": {"optimizer": {"type": "RMSProp"}}}}]}
# Valid
grid_dict_3 = {"param1": [{"epochs": 100}],
               "param2": [{"nn_config": {"general": {"optimizer": {"type": "RMSProp"}}}}]}

# Empty grid
grid_dict_4 = {}

# Valid?
grid_dict_5 = {
    1: [
        {"nn_config": {"lemma": {"config": {"shared_name": "bilstm"}}}},
        {"nn_config": {"shared_part": {"config": {"skip_connection": "sum"}}}, "lemma": {"config": {"shared_name": "bilstm_and_word_features"}}},
        {"nn_config": {"shared_part": {"config": {"skip_connection": "concat"}}}, "lemma": {"config": {"shared_name": "bilstm_and_word_features"}}}
    ],
    2: [
        {"nn_config": {"shared_part": {"config": {"lstm_cell_size": 100}}}},
        {"nn_config": {"shared_part": {"config": {"lstm_cell_size": 200}}}}
    ]}

# Not Valid? the same parameters are set in different lists
grid_dict_6 = {
    1: [
        {"nn_config": {"shared_part": {"config": {"skip_connection": "sum"}}}, "lemma": {"config": {"shared_name": "bilstm_and_word_features"}}},
        {"nn_config": {"shared_part": {"config": {"skip_connection": "concat"}}}, "lemma": {"config": {"shared_name": "bilstm_and_word_features"}}}
    ],
    2: [
        {"nn_config": {"lemma": {"config": {"shared_name": "bilstm"}}}},
        {"nn_config": {"lemma": {"config": {"shared_name": "bilstm_and_word_features"}}}}
    ]}

"bilstm_and_word_features"

# Valid
grid_dict_to_merge_1 = {"nn_config": {"general": {"optimizer": {"type": "RMSProp"}}}}

# Not valid: "general"'s value is dictionary with "optimizer" key not "type"
grid_dict_to_merge_2 = {"nn_config": {"general": {"type": "RMSProp"}}}

# Not valid: "optimizer"'s dictionary doesn't have "undefined_key" key
grid_dict_to_merge_3 = {"nn_config": {"general": {"optimizer": {"undefined_key": "RMSProp"}}}}

# Not valid: "char_cnn_kernels"'s value is list, not dictionary
grid_dict_to_merge_4 = {"nn_config": {"shared_part": {"config": {"char_cnn_kernels": {"size": 3}}}}}

grid_dict_item_path_1 = {"nn_config": {"general": {"optimizer": {"type": "RMSProp"}}}}
grid_dict_item_path_2 = {"epochs": 100}

# Hypers
hyper_dict_1 = {"epochs": 130, "batch_size": 8, "nn_config": {"general": {"optimizer": {"type": "RMSProp"}}},
                "shared_part": {"config": {"char_cnn_kernels": [{"size": 3, "count": 100, "dilation": 1}]}}}

hyper_dict_2 = {"nn_config": {"general": {"optimizer": {"initial_lr": 0.007}}}}

hyper_dict_3 = {}


class TestMergeMethod(unittest.TestCase):

    def test_hyper_is_empty(self):
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_3, grid_dict=grid_dict_to_merge_1)
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_3, grid_dict=grid_dict_to_merge_3)

    def test_grid_is_empty(self):
        self.assertEqual(merge(hyper_dict_1, grid_dict_4), hyper_dict_1)

    def test_same_dict(self):
        self.assertEqual(merge(hyper_dict_3, grid_dict_4), {})
        self.assertEqual(merge(hyper_dict_1, hyper_dict_1), hyper_dict_1)

    def test_dict_with_undefined_key(self):
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_1, grid_dict=grid_dict_to_merge_3)
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_2, grid_dict=grid_dict_to_merge_1)
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_1, grid_dict=grid_dict_to_merge_2)
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_2, grid_dict=grid_dict_to_merge_3)
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_2, grid_dict=grid_dict_to_merge_2)
        self.assertRaises(ValueError, merge, hyper_dict=hyper_dict_1, grid_dict=grid_dict_to_merge_4)


class TestCheckGridValidMethod(unittest.TestCase):

    def test_grid_has_too_many_params_in_one_key(self):
        self.assertRaises(ValueError, check_grid, grid_dict_1)

    def test_grid_has_intersections(self):
        self.assertRaises(ValueError, check_grid, grid_dict_2)

    def test_valid_grid(self):
        self.assertEqual(check_grid(grid_dict_3), None)

    def test_another_valid_grid(self):
        self.assertEqual(check_grid(grid_dict_5), None)

    def test_grid_has_intersections2(self):
        self.assertRaises(ValueError, check_grid, grid_dict_6)


class TestTakeAllKeyPathMethod(unittest.TestCase):

    def test_grid_item_path(self):
        self.assertEqual(take_path(grid_dict_item_path_1, []), ('nn_config', 'general', 'optimizer', 'type'))
        self.assertEqual(take_path(grid_dict_item_path_2, []), ('epochs',))
