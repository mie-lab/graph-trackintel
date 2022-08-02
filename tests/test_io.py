import pytest
import sqlalchemy
from graph_trackintel.io import create_table
import os
import psycopg2
from psycopg2 import sql


@pytest.fixture()
def conn_postgis():
    """
    Initiates a connection to a postGIS database that must already exist.

    Yields
    -------
    conn_string, con
    """

    dbname = "test_geopandas"
    user = os.environ.get("PGUSER")
    password = os.environ.get("PGPASSWORD")
    host = os.environ.get("PGHOST")
    port = os.environ.get("PGPORT")
    conn_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

    try:
        con = psycopg2.connect(conn_string)
        con.set_session(autocommit=True)
    except psycopg2.OperationalError:
        try:
            # psycopg2.connect may gives operational error due to
            # unsupported frontend protocol in conda environment.
            # https://stackoverflow.com/questions/61081102/psycopg2-connect-gives-operational-error-unsupported-frontend-protocol
            conn_string = conn_string + "?sslmode=disable"
            con = psycopg2.connect(conn_string)
        except psycopg2.OperationalError:
            pytest.skip("Cannot connect with postgresql database")

    yield conn_string, con
    con.close()


@pytest.fixture()
def clean_up_database(conn_postgis):

    yield
    conn_string, con = conn_postgis

    with con.cursor() as cur:
        cur.execute(
            "SELECT table_schema,table_name FROM information_schema.tables WHERE table_schema not in ('information_schema', 'pg_catalog')  ORDER BY table_schema,table_name"
        )
        rows = cur.fetchall()
        for row in rows:
            if row[1] in ["geography_columns", "geometry_columns", "nybb", "spatial_ref_sys"]:
                continue
            print("dropping table: ", row[1])

            cur.execute("drop table " + row[0] + "." + row[1] + " cascade")

        for row in rows:
            if row[0] == "public":
                continue
            cur.execute("drop schema " + row[0] + " cascade")

        cur.close()


# create a table dynamically based on a dictionary of name and type pairs


def get_table_schema(con, schema, table):
    """Get Schema of an SQL table (column names, datatypes)"""
    # https://stackoverflow.com/questions/20194806/how-to-get-a-list-column-names-and-datatypes-of-a-table-in-postgresql
    query = f"""
            SELECT
            pg_attribute.attname AS column_name,
            pg_catalog.format_type(pg_attribute.atttypid, pg_attribute.atttypmod) AS data_type
        FROM
            pg_catalog.pg_attribute
        INNER JOIN
            pg_catalog.pg_class ON pg_class.oid = pg_attribute.attrelid
        INNER JOIN
            pg_catalog.pg_namespace ON pg_namespace.oid = pg_class.relnamespace
        WHERE
            pg_attribute.attnum > 0
            AND NOT pg_attribute.attisdropped
            AND pg_namespace.nspname = '{schema}'
            AND pg_class.relname = '{table}'
        ORDER BY
            attnum ASC;"""
    print(query)
    cur = con.cursor()
    cur.execute(query)
    schema = cur.fetchall()
    column, datatype = map(list, zip(*schema))
    return column, datatype


class TestCreateTable:
    def test_create_table_drop_if_exists_flag_error(self, conn_postgis, clean_up_database):

        conn_string, con = conn_postgis
        field_type_dict = {"user_id": "text", "start_date": "text"}

        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name="test",
            schema_name="public",
            drop_if_exists=False,
        )
        with pytest.raises(psycopg2.errors.DuplicateTable):
            create_table(
                psycopg_con=con,
                field_type_dict=field_type_dict,
                table_name="test",
                schema_name="public",
                drop_if_exists=False,
            )

    def test_create_table_drop_if_exists_flag_noerror(self, conn_postgis, clean_up_database):
        conn_string, con = conn_postgis
        field_type_dict = {"user_id": "text", "start_date": "text"}

        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name="test",
            schema_name="public",
            drop_if_exists=False,
        )
        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name="test",
            schema_name="public",
            drop_if_exists=True,
        )

    def test_create_schema(self, conn_postgis, clean_up_database):
        conn_string, con = conn_postgis
        field_type_dict = {"user_id": "text", "start_date": "text"}

        with pytest.raises(psycopg2.errors.InvalidSchemaName):
            create_table(
                psycopg_con=con,
                field_type_dict=field_type_dict,
                table_name="test",
                schema_name="nonpublic",
                create_schema=False,
            )

        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name="test",
            schema_name="nonpublic",
            create_schema=True,
        )

    def test_create_table_for_time_binned_graphs(self, conn_postgis, clean_up_database):
        schema_name = "dataset_name"
        table_name = "graphs"
        conn_string, con = conn_postgis

        field_type_dict = {
            "user_id": "text",
            "start_date": "text",
            "duration": "text",
            "is_full_graph": "boolean",
            "data": "bytea",
            "test_int": "integer",
            "test_float": "double precision",
        }

        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name=table_name,
            schema_name=schema_name,
            create_schema=True,
        )

        column, datatype = get_table_schema(con, schema=schema_name, table=table_name)

        x = 1

        for ix, col in enumerate(column):
            datatype_from_dict = field_type_dict[col]

            assert datatype_from_dict == datatype[ix]

        # Columns: user_id, start_date, duration, full_graph[bool], data[binary]
