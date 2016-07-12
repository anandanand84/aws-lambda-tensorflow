# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Linear Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import dnn_linear_combined
from tensorflow.contrib.learn.python.learn.estimators.base import DeprecatedMixin


class LinearClassifier(dnn_linear_combined.DNNLinearCombinedClassifier):
  """Linear classifier model.

  Train a linear model to classify instances into one of multiple possible
  classes. When number of possible classes is 2, this is binary classification.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)

  # Estimator using the default optimizer.
  estimator = LinearClassifier(
      feature_columns=[occupation, education_x_occupation])

  # Or estimator using the FTRL optimizer with regularization.
  estimator = LinearClassifier(
      feature_columns=[occupation, education_x_occupation],
      optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001
      ))

  # Or estimator using the SDCAOptimizer.
  estimator = LinearClassifier(
     feature_columns=[occupation, education_x_occupation],
     optimizer=tf.contrib.learn.SDCAOptimizer(
       example_id_column='example_id',
       symmetric_l2_regularization=2.0
     ))

  # Input builders
  def input_fn_train: # returns x, y, where y is a tensor of dimension 1
    ...
  def input_fn_eval: # returns x, y, where y is a tensor of dimension 1
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
      if `weight_column_name` is not `None`, a feature with
        `key=weight_column_name` whose value is a `Tensor`.
      for each `column` in `feature_columns`:
      - if `column` is a `SparseColumn`, a feature with `key=column.name`
        whose `value` is a `SparseTensor`.
      - if `column` is a `RealValuedColumn, a feature with `key=column.name`
        whose `value` is a `Tensor`.
      - if `feauture_columns` is `None`, then `input` must contains only real
        valued `Tensor`.
  """

  def __init__(self,
               feature_columns=None,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               config=None):
    super(LinearClassifier, self).__init__(
        model_dir=model_dir,
        n_classes=n_classes,
        weight_column_name=weight_column_name,
        linear_feature_columns=feature_columns,
        linear_optimizer=optimizer,
        config=config)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if self._linear_feature_columns is None:
      self._linear_feature_columns = layers.infer_real_valued_columns(features)
    return super(LinearClassifier, self)._get_train_ops(features, targets)

  @property
  def weights_(self):
    return self.linear_weights_

  @property
  def bias_(self):
    return self.linear_bias_


class LinearRegressor(dnn_linear_combined.DNNLinearCombinedRegressor):
  """Linear regressor model.

  Train a linear regression model to predict target variable value given
  observation of feature values.

  Example:

  ```python
  education = sparse_column_with_hash_bucket(column_name="education",
                                             hash_bucket_size=1000)
  occupation = sparse_column_with_hash_bucket(column_name="occupation",
                                              hash_bucket_size=1000)

  education_x_occupation = crossed_column(columns=[education, occupation],
                                          hash_bucket_size=10000)

  estimator = LinearRegressor(
      feature_columns=[occupation, education_x_occupation])

  # Input builders
  def input_fn_train: # returns x, y, where y is a tensor of dimension 1
    ...
  def input_fn_eval: # returns x, y, where y is a tensor of dimension 1
    ...
  estimator.fit(input_fn=input_fn_train)
  estimator.evaluate(input_fn=input_fn_eval)
  estimator.predict(x=x)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a KeyError:
      if `weight_column_name` is not `None`:
        key=weight_column_name, value=a `Tensor`
      for column in `feature_columns`:
      - if isinstance(column, `SparseColumn`):
          key=column.name, value=a `SparseTensor`
      - if isinstance(column, `RealValuedColumn`):
          key=column.name, value=a `Tensor`
      - if `feauture_columns` is `None`:
          input must contains only real valued `Tensor`.
  """

  def __init__(self,
               feature_columns=None,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               config=None):
    super(LinearRegressor, self).__init__(
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        linear_feature_columns=feature_columns,
        linear_optimizer=optimizer,
        config=config)

  def _get_train_ops(self, features, targets):
    """See base class."""
    if self._linear_feature_columns is None:
      self._linear_feature_columns = layers.infer_real_valued_columns(features)
    return super(LinearRegressor, self)._get_train_ops(features, targets)

  @property
  def weights_(self):
    return self.linear_weights_

  @property
  def bias_(self):
    return self.linear_bias_


# TensorFlowLinearRegressor and TensorFlowLinearClassifier are deprecated.
class TensorFlowLinearRegressor(DeprecatedMixin, LinearRegressor,
                                _sklearn.RegressorMixin):
  pass


class TensorFlowLinearClassifier(DeprecatedMixin, LinearClassifier,
                                 _sklearn.ClassifierMixin):
  pass


TensorFlowRegressor = TensorFlowLinearRegressor
TensorFlowClassifier = TensorFlowLinearClassifier
