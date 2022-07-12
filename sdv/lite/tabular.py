"""Base class for tabular model presets."""

import logging
import pickle
import sys
import warnings

import numpy as np
import rdt

from sdv.metadata import Table
from sdv.tabular import GaussianCopula
from sdv.utils import get_package_versions, throw_version_mismatch_warning

LOGGER = logging.getLogger(__name__)

FAST_ML_PRESET = 'FAST_ML'
PRESETS = {
    FAST_ML_PRESET: 'Use this preset to minimize the time needed to create a synthetic data model.'
}


class TabularPreset():
    """Class for all tabular model presets.

    Args:
        name (str):
            The preset to use.
        metadata (dict or metadata.Table):
            Table metadata instance or dict representation.
        constraints (list[Constraint, dict]):
            List of Constraint objects or dicts.
    """

    _model = None
    _null_percentages = None
    _null_column = False
    _default_model = GaussianCopula

    def __init__(self, name=None, metadata=None, constraints=None):
        if name is None:
            raise ValueError('You must provide the name of a preset using the `name` '
                             'parameter. Use `TabularPreset.list_available_presets()` to browse '
                             'through the options.')
        if name not in PRESETS:
            raise ValueError(f'`name` must be one of {PRESETS}.')

        self.name = name

        if metadata is None:
            warnings.warn('No metadata provided. Metadata will be automatically '
                          'detected from your data. This process may not be accurate. '
                          'We recommend writing metadata to ensure correct data handling.')

        if metadata is not None and isinstance(metadata, Table):
            metadata = metadata.to_dict()

        if metadata is not None and constraints is not None:
            metadata['constraints'] = []
            for constraint in constraints:
                metadata['constraints'].append(constraint.to_dict())

            constraints = None

        if name == FAST_ML_PRESET:
            self._model = GaussianCopula(
                table_metadata=metadata,
                constraints=constraints,
                categorical_transformer='categorical_fuzzy',
                default_distribution='gaussian',
                learn_rounding_scheme=False,
            )

            # Decide if transformers should model the null column or not.
            self._null_column = constraints is not None
            if metadata is not None:
                self._null_column = len(metadata.get('constraints', [])) > 0

            # If transformers should model the null column, pass None to let each transformer
            # decide if it's necessary or not.
            transformer_model_missing_values = bool(self._null_column)

            dtype_transformers = {
                'i': rdt.transformers.FloatFormatter(
                    missing_value_replacement='mean' if self._null_column else None,
                    model_missing_values=transformer_model_missing_values,
                    enforce_min_max_values=True,
                ),
                'f': rdt.transformers.FloatFormatter(
                    missing_value_replacement='mean' if self._null_column else None,
                    model_missing_values=transformer_model_missing_values,
                    enforce_min_max_values=True,
                ),
                'O': rdt.transformers.FrequencyEncoder(add_noise=True),
                'b': rdt.transformers.BinaryEncoder(
                    missing_value_replacement=-1 if self._null_column else None,
                    model_missing_values=transformer_model_missing_values,
                ),
                'M': rdt.transformers.UnixTimestampEncoder(
                    missing_value_replacement='mean' if self._null_column else None,
                    model_missing_values=transformer_model_missing_values,
                ),
            }
            self._model._metadata._dtype_transformers.update(dtype_transformers)

    def fit(self, data):
        """Fit this model to the data.

        Args:
            data (pandas.DataFrame):
                Data to fit the model to.
        """
        if not self._null_column:
            self._null_percentages = {}

            for column, column_data in data.iteritems():
                num_nulls = column_data.isna().sum()
                if num_nulls > 0:
                    # Store null percentage for future reference.
                    self._null_percentages[column] = num_nulls / len(column_data)

        self._model.fit(data)

    def _postprocess_sampled(self, sampled):
        """Postprocess the sampled data.

        Add null values back based on null percentages captured in the fitting process.

        Args:
            sampled (pandas.DataFrame):
                The sampled data to postprocess.

        Returns:
            pandas.DataFrame
        """
        if self._null_percentages:
            for column, percentage in self._null_percentages.items():
                sampled[column] = sampled[column].mask(
                    np.random.random((len(sampled), )) < percentage)

        return sampled

    def sample(self, num_rows, randomize_samples=True, batch_size=None, output_file_path=None,
               conditions=None):
        """Sample rows from this table.

        Args:
            num_rows (int):
                Number of rows to sample. This parameter is required.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            batch_size (int or None):
                The batch size to sample. Defaults to `num_rows`, if None.
            output_file_path (str or None):
                The file to periodically write sampled rows to. If None, does not
                write rows anywhere.
            conditions:
                Deprecated argument. Use the `sample_conditions` method with
                `sdv.sampling.Condition` objects instead.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        sampled = self._model.sample(
            num_rows, randomize_samples, batch_size, output_file_path, conditions)

        return self._postprocess_sampled(sampled)

    def sample_conditions(self, conditions, max_tries=100, batch_size_per_try=None,
                          randomize_samples=True, output_file_path=None):
        """Sample rows from this table with the given conditions.

        Args:
            conditions (list[sdv.sampling.Condition]):
                A list of sdv.sampling.Condition objects, which specify the column
                values in a condition, along with the number of rows for that
                condition.
            max_tries (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size_per_try (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to
                a temporary file, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if isinstance(self._model, GaussianCopula):
            sampled = self._model.sample_conditions(
                conditions,
                batch_size=batch_size_per_try,
                randomize_samples=randomize_samples,
                output_file_path=output_file_path,
            )
        else:
            sampled = self._model.sample_conditions(
                conditions, max_tries, batch_size_per_try, randomize_samples, output_file_path)

        return self._postprocess_sampled(sampled)

    def sample_remaining_columns(self, known_columns, max_tries=100, batch_size_per_try=None,
                                 randomize_samples=True, output_file_path=None):
        """Sample rows from this table.

        Args:
            known_columns (pandas.DataFrame):
                A pandas.DataFrame with the columns that are already known. The output
                is a DataFrame such that each row in the output is sampled
                conditionally on the corresponding row in the input.
            max_tries (int):
                Number of times to try sampling discarded rows. Defaults to 100.
            batch_size_per_try (int):
                The batch size to use per attempt at sampling. Defaults to 10 times
                the number of rows.
            randomize_samples (bool):
                Whether or not to use a fixed seed when sampling. Defaults
                to True.
            output_file_path (str or None):
                The file to periodically write sampled rows to. Defaults to
                a temporary file, if None.

        Returns:
            pandas.DataFrame:
                Sampled data.
        """
        if isinstance(self._model, GaussianCopula):
            sampled = self._model.sample_remaining_columns(
                known_columns,
                batch_size=batch_size_per_try,
                randomize_samples=randomize_samples,
                output_file_path=output_file_path,
            )
        else:
            sampled = self._model.sample_remaining_columns(
                known_columns, max_tries, batch_size_per_try, randomize_samples, output_file_path)

        return self._postprocess_sampled(sampled)

    def save(self, path):
        """Save this model instance to the given path using pickle.

        Args:
            path (str):
                Path where the SDV instance will be serialized.
        """
        self._package_versions = get_package_versions(getattr(self, '_model', None))

        with open(path, 'wb') as output:
            pickle.dump(self, output)

    @classmethod
    def load(cls, path):
        """Load a TabularModel instance from a given path.

        Args:
            path (str):
                Path from which to load the instance.

        Returns:
            TabularModel:
                The loaded tabular model.
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
            throw_version_mismatch_warning(getattr(model, '_package_versions', None))

            return model

    @classmethod
    def list_available_presets(cls, out=sys.stdout):
        """List the available presets and their descriptions."""
        out.write(f'Available presets:\n{PRESETS}\n\n'
                  'Supply the desired preset using the `name` parameter.\n\n'
                  'Have any requests for custom presets? Contact the SDV team to learn '
                  'more an SDV Premium license.\n')

    def __repr__(self):
        """Represent tabular preset instance as text.

        Returns:
            str
        """
        return f'TabularPreset(name={self.name})'
