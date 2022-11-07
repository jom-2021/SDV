"""Base Multi Table Synthesizer class."""

import warnings
from collections import defaultdict
from copy import deepcopy

from sdv.single_table.copulas import GaussianCopulaSynthesizer


class BaseMultiTableSynthesizer:
    """Base class for multi table synthesizers.

    The ``BaseMultiTableSynthesizer`` class defines the common API that all the
    multi table synthesizers need to implement, as well as common functionality.

    Args:
        metadata (sdv.metadata.multi_table.MultiTableMetadata):
            Multi table metadata representing the data tables that this synthesizer will be used
            for.
    """

    _synthesizer = GaussianCopulaSynthesizer

    def _initialize_models(self):
        for table_name, table_metadata in self.metadata._tables.items():
            synthesizer_parameters = self._table_parameters.get(table_name, {})
            self._table_synthesizers[table_name] = self._synthesizer(
                metadata=table_metadata,
                **synthesizer_parameters
            )

    def __init__(self, metadata):
        self.metadata = metadata
        self.metadata.validate()
        self._table_synthesizers = {}
        self._table_parameters = defaultdict(dict)
        self._initialize_models()
        self._fitted = False

    def get_table_parameters(self, table_name):
        """Return the parameters that will be used to instantiate the table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters that will be used to instantiate the
                table's synthesizer.
        """
        return self._table_parameters.get(table_name, {})

    def get_parameters(self, table_name):
        """Return the parameters used to instantiate the table's synthesizer.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.

        Returns:
            parameters (dict):
                A dictionary representing the parameters used to instantiate the table's
                synthesizer.
        """
        return self._table_synthesizers.get(table_name).get_parameters()

    def update_table_parameters(self, table_name, table_parameters):
        """Update the table's synthesizer instantiation parameters.

        Args:
            table_name (str):
                Table name for which the parameters should be retrieved.
            table_parameters (dict):
                A dictionary with the parameters as keys and the values to be used to instantiate
                the table's synthesizer.
        """
        self._table_synthesizers[table_name] = self._synthesizer(
            metadata=self.metadata._tables[table_name],
            **table_parameters
        )
        self._table_parameters[table_name].update(deepcopy(table_parameters))

    def get_metadata(self):
        """Return the ``MultiTableMetadata`` for this synthesizer."""
        return self.metadata

    def _fit(self, processed_data):
        """Fit the model to the tables.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        raise NotImplementedError()

    def preprocess(self, data):
        """Transform the raw data to numerical space.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame``.

        Returns:
            dict:
                A dictionary with the preprocessed data.
        """
        self.validate(data)
        if self._fitted:
            warnings.warn(
                'This model has already been fitted. To use the new preprocessed data, '
                "please refit the model using 'fit' or 'fit_processed_data'."
            )

        processed_data = {}
        for table_name, table_data in data.items():
            synthesizer = self._table_synthesizers[table_name]
            processed_data[table_name] = synthesizer.preprocess(table_data)

        return processed_data

    def fit_processed_data(self, processed_data):
        """Fit this model to the transformed data.

        Args:
            processed_data (dict):
                Dictionary mapping each table name to a preprocessed ``pandas.DataFrame``.
        """
        self._fitted = False
        self._fit(processed_data)
        self._fitted = True

    def fit(self, data):
        """Fit this model to the original data.

        Args:
            data (dict):
                Dictionary mapping each table name to a ``pandas.DataFrame`` in the raw format
                (before any transformations).
        """
        processed_data = self.preprocess(data)
        self.fit_processed_data(processed_data)
