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
        psycopg_con.commit()
        cur.execute(
            sql.SQL("GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO wnina;").format(
                sql.Identifier(graph_schema_name)
            )
        )
        cur.execute(sql.SQL("GRANT ALL ON SCHEMA {} TO wnina;").format(sql.Identifier(graph_schema_name)))
        psycopg_con.commit()

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
