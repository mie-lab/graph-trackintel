from psycopg2 import sql
import pickle
import zlib


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

    """

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
