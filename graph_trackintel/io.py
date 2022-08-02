from psycopg2 import sql
import pickle
import zlib
from warnings import warn
from collections import defaultdict


def write_graphs_to_postgresql(
    graph_data, graph_table_name, psycopg_con, graph_schema_name="public", file_name="graph_data", drop_and_create=True
):
    """
    Stores `graph_data` as compressed binary string in a postgresql database. Stores a graph in the table
    `table_name` in the schema `schema_name`. `graph_data` is stored as a single row in `table_name`

    Parameters
    ----------
    graph_data: Dictionary of activity graphs to be stored
    graph_table_name: str
        Name of graph in database. Corresponds to the row name.
    psycopg_con: psycopg connection object
    graph_schema_name: str
        schema name for table
    file_name: str
        Name of row to store the graph (identifier)
    drop_and_create: Boolean
        If True, drop and recreate Table `graph_table_name`

    Returns
    -------

    """
    warn("write_graphs_to_postgresql is deprecated and follows an old data model", DeprecationWarning, stacklevel=2)

    pickle_string = zlib.compress(pickle.dumps(graph_data))

    cur = psycopg_con.cursor()
    if drop_and_create:
        cur.execute(
            sql.SQL("drop table if exists {}.{}").format(
                sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
            )
        )
        cur.execute(
            sql.SQL("create table {}.{} (name text, data bytea)").format(
                sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
            )
        )
    cur.execute(
        sql.SQL("insert into {}.{} values (%s, %s)").format(
            sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
        ),
        [file_name, pickle_string],
    )

    psycopg_con.commit()
    cur.close()

def write_activity_graphs_to_postgresql():
    pass

def create_activity_graph_standard_table(con, table_name, schema_name, drop_if_exists=True, create_schema=True):

    field_type_dict = {
        "user_id": "text",
        "start_date": "text",
        "duration": "text",
        "is_full_graph": "boolean",
        "data": "bytea",
    }

    create_table(
        psycopg_con=con,
        field_type_dict=field_type_dict,
        table_name=table_name,
        schema_name=schema_name,
        create_schema=create_schema,
        drop_if_exists=drop_if_exists
    )


def read_graphs_from_postgresql(
    graph_table_name, psycopg_con, graph_schema_name="public", file_name="graph_data", decompress=True
):
    """
    reads `graph_data` from postgresql database. Reads a single row named `graph_data` from `schema_name`.`table_name`

    Parameters
    ----------
    graph_data: Dictionary of activity graphs to be stored
    graph_table_name: str
        Name of graph in database. Corresponds to the row name.
    graph_schema_name: str
        schema name for table
    file_name: str
        Name of row to store the graph (identifier)
    decompress

    Returns
    -------

    """
    warn("write_graphs_to_postgresql is deprecated and follows an old data model", DeprecationWarning, stacklevel=2)
    # retrieve string
    cur = psycopg_con.cursor()
    cur.execute(
        sql.SQL("select data from {}.{} where name = %s").format(
            sql.Identifier(graph_schema_name), sql.Identifier(graph_table_name)
        ),
        (file_name,),
    )
    pickle_string2 = cur.fetchall()[0][0].tobytes()

    cur.close()
    if decompress:
        AG_dict2 = pickle.loads(zlib.decompress(pickle_string2))
    else:
        AG_dict2 = pickle.loads(pickle_string2)

    return AG_dict2


def create_table(
    psycopg_con, table_name, field_type_dict, schema_name="public", drop_if_exists=False, create_schema=False
):
    """Create a postgresql table based on a field name - field type mapping

    Parameters
    ----------
    psycopg_con:
        psycopg2 connection object
    table_name: str
        name of the table that will be created
    field_type_dict: dict
        A dictionary that maps field name and field type of the new table. All types musst be postgresql types given
        as strings. E.g.,
            field_type_dict = {'user_id': 'text', 'start_date': 'text', 'duration': 'text',
                                'is_full_graph': 'bool', 'data': 'bytea'}
    schema_name: str
        name of the schema in which the table will be created. Default is "public"
    drop_if_exists: bool
        if True, an existing table is dropped before creating it.
    create_schema: bool
        if True, the schema `schema_name` is created first.

    Returns
    -------

    """

    with psycopg_con.cursor() as cur:
        if create_schema:
            cur.execute(sql.SQL(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
            psycopg_con.commit()

        if drop_if_exists:
            cur.execute(sql.SQL(f"drop table if exists {schema_name}.{table_name}"))
            psycopg_con.commit()

        sql_string = sql.SQL(f"create table {schema_name}.{table_name} ( ")

        for ix, (field_name, field_type) in enumerate(field_type_dict.items()):

            if ix == len(field_type_dict) - 1:

                sql_string = sql_string + sql.SQL(f"{field_name} {field_type}")
            else:
                sql_string = sql_string + sql.SQL(f"{field_name} {field_type}, ")

        sql_string = sql_string + sql.SQL(" );")

        cur.execute(sql_string)

        psycopg_con.commit()


def write_data_to_table(psycopg_con, table_name, input_data, schema_name="public"):
    """write data to postgresql table

    Parameters
    ----------
    psycopg_con:
        psycopg2 connection object
    table_name: str
        name of the table that will be created
    input_data
        Data needs to be structured in one of three ways and needs to match an existing table. Tables can be created
        using the `graph-trackintel.io.create_table` function.
        The three supported data structures are:

            case 1: a single dictionary with field names as keys and values as values.
                E.g., input_data = {"user_id": "1", "start_date": "10-20-20",
                                "duration": "20 minutes", "is_full_graph": False}
            case 2: a single dictionary with field names as keys and a list of values for each key.
                E.g.,  input_data = {"user_id": ["1", "2", "3", "4"],
                          "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
                          "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
                          "is_full_graph": [True, False, True, False]}
            case 3: a list of dictionaries with field names as keys and single values as values.
                E.g., input_data = [{"user_id": "1", "start_date": "10-20-21", "duration": "21 minutes",
                                  "is_full_graph": False},
                                  {"user_id": "2", "start_date": "10-20-22",  "duration": "22 minutes",
                                   "is_full_graph": True},
                                  {"user_id": "3", "start_date": "10-20-23", "duration": "230 minutes",
                                  "is_full_graph": True},
                                  {"user_id": "4", "start_date": "10-20-24", "duration": "204 minutes",
                                    "is_full_graph": False}]

    schema_name: str
        name of the schema in which the table will be created. Default is "public"

    Returns
    -------

    """

    # possible inputs:
    # 1) dictionary with a single value per field
    # 2) a dictionary with a list of values per field
    # 3) a list of dictionaries with a single value per field

    if isinstance(input_data, dict):

        if all([isinstance(k, list) for k in input_data.values()]):
            case = 2
        else:
            case = 1
    else:
        case = 3

    # sql insert header
    sql_header = f"INSERT INTO {schema_name}.{table_name}("
    if case == 3:
        sql_header = sql_header + ", ".join(list(input_data[0].keys()))
    else:
        sql_header = sql_header + ", ".join(list(input_data.keys()))
    sql_header = sql_header + ") VALUES "

    if case == 3:
        input_data = _merge_dicts(input_data)
        case = 2

    # list of list with all values
    list_of_input_values = [v for v in input_data.values()]

    # list of tuples
    if case == 1:
        sql_placeholder = "(" + ", ".join(["%s"] * len(input_data.keys())) + ")"
        sql_data = tuple(list_of_input_values)
    if case == 2:
        sql_placeholder = ", ".join(["%s"] * len(input_data.keys()))
        sql_data = list(zip(*list_of_input_values))

    sql_string = sql_header + sql_placeholder
    with psycopg_con.cursor() as cur:
        cur.execute(sql_string, sql_data)


def _merge_dicts(dicts):
    """merge two dictionaries with overlapping keys

    Parameters
    ----------
    dicts: iterable of dictionaries

    Returns: dictionary
    -------

    https://stackoverflow.com/questions/5946236/how-to-merge-multiple-dicts-with-same-key-or-different-key
    """

    dd = defaultdict(list)

    for d in dicts:  # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)

    return dd
