import os
import re
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from sdv.metadata.dataset import Metadata, _load_csv, _parse_dtypes, _read_csv_dtypes


def test__read_csv_dtypes():
    """Test read csv data types"""
    # Run
    table_meta = {
        'fields': {
            'a_field': {
                'type': 'categorical'
            },
            'b_field': {
                'type': 'boolean'
            },
            'c_field': {
                'type': 'datetime',
            },
            'd_field': {
                'type': 'id',
                'subtype': 'string'
            },
            'e_field': {
                'type': 'id',
                'subtype': 'integer'
            }
        }
    }
    result = _read_csv_dtypes(table_meta)

    # Asserts
    assert result == {'d_field': str}


def test__parse_dtypes():
    """Test parse data types"""
    # Run
    data = pd.DataFrame({
        'a_field': ['1996-10-17', '1965-05-23'],
        'b_field': ['7', '14'],
        'c_field': ['1', '2'],
        'd_field': ['other', 'data']
    })
    table_meta = {
        'fields': {
            'a_field': {
                'type': 'datetime',
                'format': '%Y-%m-%d'
            },
            'b_field': {
                'type': 'numerical',
                'subtype': 'integer'
            },
            'c_field': {
                'type': 'id',
                'subtype': 'integer'
            },
            'd_field': {
                'type': 'other'
            }
        }
    }
    result = _parse_dtypes(data, table_meta)

    # Asserts
    expected = pd.DataFrame({
        'a_field': pd.to_datetime(['1996-10-17', '1965-05-23'], format='%Y-%m-%d'),
        'b_field': [7, 14],
        'c_field': [1, 2],
        'd_field': ['other', 'data']
    })
    pd.testing.assert_frame_equal(result, expected)


@patch('sdv.metadata.dataset._parse_dtypes')
@patch('sdv.metadata.dataset.pd.read_csv')
@patch('sdv.metadata.dataset._read_csv_dtypes')
def test__load_csv(rcdtypes_mock, read_csv_mock, pdtypes_mock):
    # Run
    table_meta = {
        'path': 'filename.csv',
        'other': 'stuff'
    }
    result = _load_csv(os.path.join('a', 'path'), table_meta)

    # Asserts
    assert result == pdtypes_mock.return_value
    rcdtypes_mock.assert_called_once_with(table_meta)
    dtypes = rcdtypes_mock.return_value
    read_csv_mock.assert_called_once_with(os.path.join('a', 'path', 'filename.csv'), dtype=dtypes)
    pdtypes_mock.assert_called_once_with(read_csv_mock.return_value, table_meta)


class TestMetadata(TestCase):
    """Test Metadata class."""

    def test__analyze_relationships(self):
        """Test get relationships"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        _metadata = {
            'tables': {
                'test': {
                    'use': True,
                    'name': 'test',
                    'fields': {
                        'test_field': {
                            'ref': {'table': 'table_ref', 'field': 'field_ref'},
                            'name': 'test_field'
                        }
                    }
                },
                'test_not_use': {
                    'use': False,
                    'name': 'test_not_use',
                    'fields': {
                        'test_field_not_use': {
                            'ref': {'table': 'table_ref', 'field': 'field_ref'},
                            'name': 'test_field_not_use'
                        }
                    }
                }
            }
        }
        metadata._metadata = _metadata

        # Run
        Metadata._analyze_relationships(metadata)

        # Asserts
        assert metadata._child_map == {'table_ref': {'test'}}
        assert metadata._parent_map == {'test': {'table_ref'}}

    def test__dict_metadata_list(self):
        """Test dict_metadata"""
        # Run
        metadata = {
            'tables': [
                {
                    'name': 'test',
                    'fields': [
                        {
                            'ref': {
                                'table': 'table_ref',
                                'field': 'field_ref'
                            },
                            'name': 'test_field'
                        }
                    ]
                },
                {
                    'name': 'other',
                    'use': False,
                }
            ]
        }
        result = Metadata._dict_metadata(metadata)

        # Asserts
        expected = {
            'tables': {
                'test': {
                    'fields': {
                        'test_field': {
                            'ref': {
                                'table': 'table_ref',
                                'field': 'field_ref'
                            }
                        }
                    }
                }
            }
        }
        assert result == expected

    def test__dict_metadata_dict(self):
        """Test dict_metadata"""
        # Run
        metadata = {
            'tables': {
                'test': {
                    'fields': {
                        'test_field': {
                            'ref': {
                                'table': 'table_ref',
                                'field': 'field_ref'
                            }
                        }
                    }
                },
                'other': {
                    'use': False,
                }
            }
        }
        result = Metadata._dict_metadata(metadata)

        # Asserts
        expected = {
            'tables': {
                'test': {
                    'fields': {
                        'test_field': {
                            'ref': {
                                'table': 'table_ref',
                                'field': 'field_ref'
                            }
                        }
                    }
                }
            }
        }
        assert result == expected

    @patch('sdv.metadata.dataset.Metadata._analyze_relationships')
    @patch('sdv.metadata.dataset.Metadata._dict_metadata')
    def test___init__default_metadata_dict(self, mock_meta, mock_relationships):
        """Test create Metadata instance default with a dict"""
        # Run
        metadata = Metadata({'some': 'meta'})

        # Asserts
        mock_meta.assert_called_once_with({'some': 'meta'})
        mock_relationships.assert_called_once_with()
        assert metadata.root_path == '.'

    def test_get_children(self):
        """Test get children"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata._child_map = {
            'test': 'child_table'
        }

        # Run
        result = Metadata.get_children(metadata, 'test')

        # Asserts
        assert result == 'child_table'

    def test_get_parents(self):
        """Test get parents"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata._parent_map = {
            'test': 'parent_table'
        }

        # Run
        result = Metadata.get_parents(metadata, 'test')

        # Asserts
        assert result == 'parent_table'

    def test_get_table_meta(self):
        """Test get table meta"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata._metadata = {
            'tables': {
                'test': {'some': 'data'}
            }
        }

        # Run
        result = Metadata.get_table_meta(metadata, 'test')

        # Asserts
        assert result == {'some': 'data'}

    @patch('sdv.metadata.dataset._load_csv')
    def test_load_table(self, mock_load_csv):
        """Test load table"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.root_path = 'a/path'
        metadata.get_table_meta.return_value = {'some': 'data'}
        mock_load_csv.return_value = 'data'

        # Run
        result = Metadata.load_table(metadata, 'test')

        # Asserts
        assert result == 'data'

        metadata.get_table_meta.assert_called_once_with('test')
        mock_load_csv.assert_called_once_with('a/path', {'some': 'data'})

    def test_get_dtypes_with_ids(self):
        """Test get data types including ids."""
        # Setup
        table_meta = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'numerical', 'subtype': 'integer'},
                'item 2': {'type': 'numerical', 'subtype': 'float'},
                'item 3': {'type': 'categorical'},
                'item 4': {'type': 'boolean'},
                'item 5': {'type': 'datetime'}
            },
            'primary_key': 'item 0'
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta
        metadata._DTYPES = Metadata._DTYPES

        # Run
        result = Metadata.get_dtypes(metadata, 'test', ids=True)

        # Asserts
        expected = {
            'item 0': 'int',
            'item 1': 'int',
            'item 2': 'float',
            'item 3': 'object',
            'item 4': 'bool',
            'item 5': 'datetime64',
        }
        assert result == expected

    def test_get_dtypes_no_ids(self):
        """Test get data types excluding ids."""
        # Setup
        table_meta = {
            'fields': {
                'item 0': {'type': 'id', 'subtype': 'integer'},
                'item 1': {'type': 'numerical', 'subtype': 'integer'},
                'item 2': {'type': 'numerical', 'subtype': 'float'},
                'item 3': {'type': 'categorical'},
                'item 4': {'type': 'boolean'},
                'item 5': {'type': 'datetime'},
            }
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta
        metadata._DTYPES = Metadata._DTYPES

        # Run
        result = Metadata.get_dtypes(metadata, 'test')

        # Asserts
        expected = {
            'item 1': 'int',
            'item 2': 'float',
            'item 3': 'object',
            'item 4': 'bool',
            'item 5': 'datetime64',
        }
        assert result == expected

    def test_get_dtypes_error_invalid_type(self):
        """Test get data types with an invalid type."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'unknown'}
            }
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta
        metadata._DTYPES = Metadata._DTYPES

        # Run
        errors = []
        Metadata.get_dtypes(metadata, 'test', errors=errors)

        assert len(errors) == 1

    def test_get_dtypes_error_subtype_numerical(self):
        """Test get data types with an invalid numerical subtype."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'numerical', 'subtype': 'boolean'}
            }
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta
        metadata._DTYPES = Metadata._DTYPES

        # Run
        errors = []
        Metadata.get_dtypes(metadata, 'test', errors=errors)

        assert len(errors) == 1

    def test_get_dtypes_error_subtype_id(self):
        """Test get data types with an invalid id subtype."""
        # Setup
        table_meta = {
            'fields': {
                'item': {'type': 'id', 'subtype': 'boolean'}
            }
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta
        metadata._DTYPES = Metadata._DTYPES

        # Run
        errors = []
        Metadata.get_dtypes(metadata, 'test', ids=True, errors=errors)

        assert len(errors) == 1

    def test_load_tables(self):
        """Test get tables"""
        # Setup
        table_names = ['foo', 'bar', 'tar']
        table_data = [
            pd.DataFrame({'foo': [1, 2]}),
            pd.DataFrame({'bar': [3, 4]}),
            pd.DataFrame({'tar': [5, 6]})
        ]
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.side_effect = table_names
        metadata.load_table.side_effect = table_data

        # Run
        tables = ['table 1', 'table 2', 'table 3']
        result = Metadata.load_tables(metadata, tables=tables)

        # Asserts
        expected = {
            'table 1': pd.DataFrame({'foo': [1, 2]}),
            'table 2': pd.DataFrame({'bar': [3, 4]}),
            'table 3': pd.DataFrame({'tar': [5, 6]})
        }
        assert result.keys() == expected.keys()

        for key, value in result.items():
            pd.testing.assert_frame_equal(value, expected[key])

    def test_get_fields(self):
        """Test get fields"""
        # Setup
        table_meta = {
            'fields': {
                'a_field': 'some data',
                'b_field': 'other data'
            }
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta

        # Run
        result = Metadata.get_fields(metadata, 'test')

        # Asserts
        expected = {'a_field': 'some data', 'b_field': 'other data'}
        assert result == expected

        metadata.get_table_meta.assert_called_once_with('test')

    def test_get_primary_key(self):
        """Test get primary key"""
        # Setup
        table_meta = {
            'primary_key': 'a_primary_key'
        }
        metadata = Mock(spec_set=Metadata)
        metadata.get_table_meta.return_value = table_meta

        # Run
        result = Metadata.get_primary_key(metadata, 'test')

        # Asserts
        assert result == 'a_primary_key'
        metadata.get_table_meta.assert_called_once_with('test')

    def test_get_foreign_keys(self):
        """Test get foreign key"""
        # Setup
        metadata = {
            'tables': {
                'parent': {
                    'fields': {
                        'parent_id': {
                            'type': 'id',
                        }
                    },
                    'primary_key': 'parent_id'
                },
                'child': {
                    'fields': {
                        'parent_id': {
                            'type': 'id',
                            'ref': {
                                'table': 'parent',
                                'field': 'id'
                            }
                        },
                        'parent_id_2': {
                            'type': 'id',
                            'ref': {
                                'table': 'parent',
                                'field': 'id'
                            }
                        },
                    }
                }
            }
        }
        metadata = Metadata(metadata)

        # Run
        result = Metadata.get_foreign_keys(metadata, 'parent', 'child')

        # Asserts
        assert result == ['parent_id', 'parent_id_2']

    def test_add_table_already_exist(self):
        """Try to add a new table that already exist"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']

        # Run
        err_msg = 'Table "a_table" already exists.'
        with pytest.raises(ValueError, match=err_msg):
            Metadata.add_table(metadata, 'a_table')

    def test_add_table_only_name(self):
        """Add table with only the name"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}

        # Run
        Metadata.add_table(metadata, 'x_table')

        # Asserts
        expected_table_meta = {
            'fields': {}
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_primary_key(self):
        """Add table with primary key"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}

        # Run
        Metadata.add_table(metadata, 'x_table', primary_key='id')

        # Asserts
        expected_table_meta = {
            'fields': {}
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.assert_called_once_with('x_table', 'id')
        metadata.add_relationship.call_count == 0

    def test_add_table_with_foreign_key(self):
        """Add table with foreign key"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}

        # Run
        Metadata.add_table(metadata, 'x_table', parent='users')

        # Asserts
        expected_table_meta = {
            'fields': {}
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.assert_called_once_with('users', 'x_table', None)

    def test_add_table_with_fields_metadata(self):
        """Add table with fields metadata"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}

        # Run
        fields_metadata = {
            'a_field': {'type': 'numerical', 'subtype': 'integer'}
        }

        Metadata.add_table(metadata, 'x_table', fields_metadata=fields_metadata)

        # Asserts
        expected_table_meta = {
            'fields': {
                'a_field': {'type': 'numerical', 'subtype': 'integer'}
            }
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_fields_no_data(self):
        """Add table with fields and no data"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}

        # Run
        fields = ['a_field', 'b_field']

        Metadata.add_table(metadata, 'x_table', fields=fields)

        # Asserts
        expected_table_meta = {
            'fields': {}
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

    def test_add_table_with_fields_data(self):
        """Add table with fields and data"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}
        metadata._get_field_details.return_value = {
            'a_field': {'type': 'numerical', 'subtype': 'integer'},
            'b_field': {'type': 'boolean'}
        }

        # Run
        fields = ['a_field', 'b_field']
        data = pd.DataFrame({'a_field': [0, 1], 'b_field': [True, False], 'c_field': ['a', 'b']})

        Metadata.add_table(metadata, 'x_table', fields=fields, data=data)

        # Asserts
        expected_table_meta = {
            'fields': {
                'a_field': {'type': 'numerical', 'subtype': 'integer'},
                'b_field': {'type': 'boolean'}
            }
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_no_fields_data(self):
        """Add table with data to analyze all"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}
        metadata._get_field_details.return_value = {
            'a_field': {'type': 'numerical', 'subtype': 'integer'},
            'b_field': {'type': 'boolean'},
            'c_field': {'type': 'categorical'}
        }

        # Run
        data = pd.DataFrame({'a_field': [0, 1], 'b_field': [True, False], 'c_field': ['a', 'b']})

        Metadata.add_table(metadata, 'x_table', data=data)

        # Asserts
        expected_table_meta = {
            'fields': {
                'a_field': {'type': 'numerical', 'subtype': 'integer'},
                'b_field': {'type': 'boolean'},
                'c_field': {'type': 'categorical'}
            }
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    @patch('sdv.metadata.dataset.pd.read_csv')
    def test_add_table_with_data_str(self, mock_read_csv):
        """Add table with data as str"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}
        mock_read_csv.return_value = pd.DataFrame({
            'a_field': [0, 1],
            'b_field': [True, False],
            'c_field': ['a', 'b']
        })
        metadata._get_field_details.return_value = {
            'a_field': {'type': 'numerical', 'subtype': 'integer'},
            'b_field': {'type': 'boolean'},
            'c_field': {'type': 'categorical'}
        }

        # Run
        Metadata.add_table(metadata, 'x_table', data='/path/to/file.csv')

        expected_table_meta = {
            'fields': {
                'a_field': {'type': 'numerical', 'subtype': 'integer'},
                'b_field': {'type': 'boolean'},
                'c_field': {'type': 'categorical'}
            },
            'path': '/path/to/file.csv'
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_table_with_constraints(self):
        """Test the ``Metadata.add_table`` method with constraints.

        Expect that when constraints are provided, the metadata for the
        specified table is created with the given constraints.

        Input:
        - Metadata object
        - Table name of the desired table to add
        - Metadata for the table's fields
        - Constraints for the given table
        Side Effects:
        - An entry is added to the metadata for the provided table, which contains
          the given fields and constraints.
        """
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata._metadata = {'tables': {}}

        # Run
        fields_metadata = {
            'a_field': {'type': 'numerical', 'subtype': 'integer'},
            'b_field': {'type': 'numerical', 'subtype': 'integer'}
        }
        constraints = [
            {
                'constraint_name': 'sdv.constraints.tabular.Inequality',
                'columns': [
                    'a_field',
                    'b_field',
                ],
            }
        ]

        Metadata.add_table(
            metadata,
            'x_table',
            fields_metadata=fields_metadata,
            constraints=constraints,
        )

        # Asserts
        expected_table_meta = {
            'fields': {
                'a_field': {'type': 'numerical', 'subtype': 'integer'},
                'b_field': {'type': 'numerical', 'subtype': 'integer'},
            },
            'constraints': [
                {
                    'constraint_name': 'sdv.constraints.tabular.Inequality',
                    'columns': [
                        'a_field',
                        'b_field',
                    ],
                },
            ]
        }

        assert metadata._metadata['tables']['x_table'] == expected_table_meta

        metadata.set_primary_key.call_count == 0
        metadata.add_relationship.call_count == 0

    def test_add_relationship_table_no_exist(self):
        """Add relationship table no exist"""
        # Run
        err_msg = 'Table "a_table" does not exist'
        with pytest.raises(ValueError, match=err_msg):
            Metadata.add_relationship(Metadata(), 'a_table', 'b_table')

    def test_add_relationship_parent_no_exist(self):
        """Add relationship table no exist"""
        # Setup
        metadata = Metadata()
        metadata._metadata['tables'] = {'a_table': {}}

        # Run
        err_msg = 'Table "b_table" does not exist'
        with pytest.raises(ValueError, match=err_msg):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_already_exist(self):
        """Add relationship already exist"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata.get_parents.return_value = {'b_table'}

        # Run
        mock = metadata.get_primary_key()
        err_msg = re.escape(f'Field "b_table.{mock}" already defines a relationship')
        with pytest.raises(ValueError, match=err_msg):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_add_relationship_parent_no_primary_key(self):
        """Add relationship parent no primary key"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = ['a_table', 'b_table']
        metadata.get_parents.return_value = set()
        metadata.get_children.return_value = set()
        metadata.get_primary_key.return_value = None

        # Run
        err_msg = 'Parent table "a_table" does not have a primary key'
        with pytest.raises(ValueError, match=err_msg):
            Metadata.add_relationship(metadata, 'a_table', 'b_table')

    def test_set_primary_key(self):
        """Set primary key table no exist"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = []
        metadata.get_fields.return_value = {'a_field': {'type': 'id', 'subtype': 'integer'}}
        metadata._metadata = {
            'tables': {
                'a_table': {
                    'fields': {'a_field': {'type': 'id', 'subtype': 'integer'}}
                }
            }
        }

        # Run
        Metadata.set_primary_key(metadata, 'a_table', 'a_field')

        # Asserts
        metadata._check_field.assert_called_once_with('a_table', 'a_field', exists=True)
        metadata.get_fields.assert_called_once_with('a_table')
        metadata._get_key_subtype.assert_called_once_with({'type': 'id', 'subtype': 'integer'})

    def test_add_field(self):
        """Add field table no exist"""
        # Setup
        metadata = Mock(spec_set=Metadata)
        metadata.get_tables.return_value = []
        metadata._metadata = {
            'tables': {
                'a_table': {'fields': {}}
            }
        }

        # Run
        Metadata.add_field(metadata, 'a_table', 'a_field', 'id', 'string', None)

        # Asserts
        expected_metadata = {
            'tables': {
                'a_table': {
                    'fields': {'a_field': {'type': 'id', 'subtype': 'string'}}
                }
            }
        }

        assert metadata._metadata == expected_metadata
        metadata._check_field.assert_called_once_with('a_table', 'a_field', exists=False)
