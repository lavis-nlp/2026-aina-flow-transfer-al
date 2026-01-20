import numpy as np
from flc.shared.utils.numpy_helpers import hot_encoding_to_ints


class TestHotEncodingToInts:
    def test_basic_one_hot(self):
        hot_encoded = np.array([1, 0, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0]
        
    def test_single_class_middle(self):
        hot_encoded = np.array([0, 1, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [1]
        
    def test_single_class_last(self):
        hot_encoded = np.array([0, 0, 1])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [2]

    def test_binary_classification_first(self):
        hot_encoded = np.array([1, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0]
        
    def test_binary_classification_second(self):
        hot_encoded = np.array([0, 1])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [1]

    def test_many_classes(self):
        hot_encoded = np.array([0, 0, 0, 0, 1])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [4]

    def test_multilabel_basic(self):
        hot_encoded = np.array([1, 1, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0, 1]

    def test_multilabel_all_classes(self):
        hot_encoded = np.array([1, 1, 1])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0, 1, 2]

    def test_multilabel_sparse(self):
        hot_encoded = np.array([1, 0, 1])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0, 2]

    def test_float_values_multilabel(self):
        hot_encoded = np.array([1.0, 0.5, 0.0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0, 1]

    def test_boolean_values_multilabel(self):
        hot_encoded = np.array([True, True, False])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0, 1]

    def test_empty_array(self):
        hot_encoded = np.array([])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == []

    def test_all_zeros(self):
        hot_encoded = np.array([0, 0, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == []

    def test_negative_values_treated_as_zero(self):
        hot_encoded = np.array([-1, 0, 2])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [2]

    def test_return_type_consistency(self):
        hot_encoded = np.array([1, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert isinstance(result, list)
        assert all(isinstance(x, (int, np.integer)) for x in result)

    def test_threshold_behavior(self):
        hot_encoded = np.array([0.1, 0.9, 0.0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0, 1]

    def test_multilabel_preserves_order(self):
        hot_encoded = np.array([0, 1, 0, 1, 0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [1, 3]
        
    def test_single_element_array(self):
        hot_encoded = np.array([1])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == [0]
        
    def test_single_element_zero(self):
        hot_encoded = np.array([0])
        result = hot_encoding_to_ints(hot_encoded)
        assert result == []
