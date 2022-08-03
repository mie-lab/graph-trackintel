import pandas as pd
import pytest
import sqlalchemy
from graph_trackintel.io import (
    create_table,
    write_data_to_table,
    create_activity_graph_standard_table,
    insert_data_to_postgresql,
    read_data_from_postgresql,
    write_table_to_postgresql,
)
import os
import psycopg2
import networkx as nx


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
        delete_all_tables_and_schemas(con)
    except psycopg2.OperationalError:
        try:
            # psycopg2.connect may gives operational error due to
            # unsupported frontend protocol in conda environment.
            # https://stackoverflow.com/questions/61081102/psycopg2-connect-gives-operational-error-unsupported-frontend-protocol
            conn_string = conn_string + "?sslmode=disable"
            con = psycopg2.connect(conn_string)
            con.set_session(autocommit=True)
            delete_all_tables_and_schemas(con)
        except psycopg2.OperationalError:
            pytest.skip("Cannot connect with postgresql database")

    yield conn_string, con
    con.close()


def delete_all_tables_and_schemas(con):

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


@pytest.fixture()
def clean_up_database(conn_postgis):
    """drops all tables and schemas that were created"""

    yield

    conn_string, con = conn_postgis
    delete_all_tables_and_schemas(con)


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
    cur = con.cursor()
    cur.execute(query)
    schema = cur.fetchall()
    column, datatype = map(list, zip(*schema))
    return column, datatype


class TestCreateTable:
    def test_create_table_drop_if_exists_flag_error(self, conn_postgis, clean_up_database):
        """Check create table error if table exists"""
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
        """test if existing tables can be overwritten"""
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
        """test 'create_schema' option"""
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
        """Test the creation of a table with different datatypes"""
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

        for ix, col in enumerate(column):
            datatype_from_dict = field_type_dict[col]

            assert datatype_from_dict == datatype[ix]


class TestWriteToTable:
    def init_testing_table(self, con, schema_name="dataset_name", table_name="graphs"):
        """function to initialize a table for testing"""

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

    def test_write_to_table_case_1(self, conn_postgis, clean_up_database):
        """test input format (case 1)"""

        conn_string, con = conn_postgis
        schema_name = "dataset_name"
        table_name = "graphs"
        self.init_testing_table(con, schema_name="dataset_name", table_name="graphs")

        input_data = {"user_id": "1", "start_date": "10-20-20", "duration": "20 minutes", "is_full_graph": False}

        write_data_to_table(psycopg_con=con, table_name=table_name, input_data=input_data, schema_name=schema_name)

        df = pd.DataFrame(input_data, index=[0])

        df_from_sql = pd.read_sql(
            f"select user_id, start_date, duration, is_full_graph from {schema_name}" f".{table_name}", con=con
        )

        pd.testing.assert_frame_equal(df, df_from_sql)

    def test_write_to_table_case_2(self, conn_postgis, clean_up_database):
        """test input format (case 2)"""

        conn_string, con = conn_postgis
        schema_name = "dataset_name"
        table_name = "graphs"
        self.init_testing_table(con, schema_name="dataset_name", table_name="graphs")

        input_data = {
            "user_id": ["1", "2", "3", "4", "5"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30", "10-20-31"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes", "25 minutes"],
            "is_full_graph": [True, False, True, False, False],
        }

        write_data_to_table(psycopg_con=con, table_name=table_name, input_data=input_data, schema_name=schema_name)

        df = pd.DataFrame(input_data)

        df_from_sql = pd.read_sql(
            f"select user_id, start_date, duration, is_full_graph from {schema_name}" f".{table_name}", con=con
        )

        pd.testing.assert_frame_equal(df, df_from_sql)

    def test_write_to_table_case_3(self, conn_postgis, clean_up_database):
        """test input format (case 3)"""

        conn_string, con = conn_postgis
        schema_name = "dataset_name"
        table_name = "graphs"
        self.init_testing_table(con, schema_name="dataset_name", table_name="graphs")

        input_data = [
            {"user_id": "1", "start_date": "10-20-21", "duration": "21 minutes", "is_full_graph": False},
            {"user_id": "2", "start_date": "10-20-22", "duration": "22 minutes", "is_full_graph": True},
            {"user_id": "3", "start_date": "10-20-23", "duration": "230 minutes", "is_full_graph": True},
            {"user_id": "4", "start_date": "10-20-24", "duration": "204 minutes", "is_full_graph": False},
        ]

        write_data_to_table(psycopg_con=con, table_name=table_name, input_data=input_data, schema_name=schema_name)

        df = pd.DataFrame(input_data)

        df_from_sql = pd.read_sql(
            f"select user_id, start_date, duration, is_full_graph from {schema_name}" f".{table_name}", con=con
        )

        pd.testing.assert_frame_equal(df, df_from_sql)


class TestCreateActivityGraphStandardTable:
    def test_run_function(self, conn_postgis, clean_up_database):
        """test if table with correct datatypes is created"""
        conn_string, con = conn_postgis
        schema_name = "my_graph_dataset"
        table_name = "graphs"

        create_activity_graph_standard_table(con=con, table_name=table_name, schema_name=schema_name)

        field_type_dict = {
            "user_id": "text",
            "start_date": "text",
            "duration": "text",
            "is_full_graph": "boolean",
            "gap_threshold": "integer",
            "trips": "boolean",
            "min_daily_coverage": "double precision",
            "min_nb_good_days": "integer",
            "number_of_good_days": "integer",
            "data": "bytea",
        }

        column, datatype = get_table_schema(con, schema=schema_name, table=table_name)

        for ix, col in enumerate(column):
            datatype_from_dict = field_type_dict[col]

            assert datatype_from_dict == datatype[ix]


class TestWriteActivityGraphsToPostgresql:
    def test_case2(self, conn_postgis, clean_up_database):
        """test input format case 2"""
        conn_string, con = conn_postgis
        schema_name = "my_graph_dataset"
        table_name = "graphs"

        create_activity_graph_standard_table(con=con, table_name=table_name, schema_name=schema_name)

        graphs = [nx.erdos_renyi_graph(10, 0.2, seed=None, directed=False) for i in range(4)]
        input_data = {
            "user_id": ["1", "2", "3", "4"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
            "is_full_graph": [True, False, True, False],
            "data": graphs,
        }

        insert_data_to_postgresql(
            con=con, table_name=table_name, schema_name=schema_name, input_data=input_data, data_field_name="data"
        )

    def test_case2_multiple_data_fields(self, conn_postgis, clean_up_database):
        """test writing table with multiple data fields that need encoding"""
        conn_string, con = conn_postgis
        schema_name = "my_graph_dataset"
        table_name = "graphs"

        field_type_dict = {
            "user_id": "text",
            "start_date": "text",
            "duration": "text",
            "is_full_graph": "boolean",
            "data": "bytea",
            "data2": "bytea",
        }

        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name=table_name,
            schema_name=schema_name,
            create_schema=True,
            drop_if_exists=False,
        )

        graphs = [nx.erdos_renyi_graph(10, 0.2, seed=None, directed=False) for i in range(4)]
        graphs2 = [nx.erdos_renyi_graph(10, 0.2, seed=None, directed=False) for i in range(4)]
        input_data = {
            "user_id": ["1", "2", "3", "4"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
            "is_full_graph": [True, False, True, False],
            "data": graphs,
            "data2": graphs2,
        }

        insert_data_to_postgresql(
            con=con,
            table_name=table_name,
            schema_name=schema_name,
            input_data=input_data,
            data_field_name=["data", "data2"],
        )


class TestReadDataFromPostgresql:
    def test_write_read_compare(self, conn_postgis, clean_up_database):
        """test consistency after writing to and reading from postgresql"""

        conn_string, con = conn_postgis
        engine = sqlalchemy.create_engine(conn_string)

        schema_name = "my_graph_dataset"
        table_name = "graphs"

        create_activity_graph_standard_table(con=con, table_name=table_name, schema_name=schema_name)

        graphs = [nx.erdos_renyi_graph(10 * (i + 1), 0.2, seed=None, directed=False) for i in range(4)]
        input_data = {
            "user_id": ["1", "2", "3", "4"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
            "is_full_graph": [True, False, True, False],
            "data": graphs,
        }

        insert_data_to_postgresql(
            con=con, table_name=table_name, schema_name=schema_name, input_data=input_data, data_field_name="data"
        )

        query = f"select * from {schema_name}.{table_name}"
        df_db = read_data_from_postgresql(sql=query, engine=engine, data_field_name="data")

        df_input = pd.DataFrame(input_data)

        for ix in range(len(graphs)):
            g1 = df_db["data"].iloc[ix]
            g2 = df_input["data"].iloc[ix]
            assert len(g1) == len(g2)
            assert len(g1.edges()) == len(g2.edges())


class TestWriteTableToPostgresql:
    def test_write_read_compare_pd(self, conn_postgis, clean_up_database):
        """test consistency after writing to and reading from postgresql"""
        conn_string, con = conn_postgis
        engine = sqlalchemy.create_engine(conn_string)

        schema_name = "my_graph_dataset"
        table_name = "graphs"

        create_activity_graph_standard_table(con=con, table_name=table_name, schema_name=schema_name)

        graphs = [nx.erdos_renyi_graph(10 * (i + 1), 0.2, seed=None, directed=False) for i in range(4)]
        input_data = {
            "user_id": ["1", "2", "3", "4"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
            "is_full_graph": [True, False, True, False],
            "data": graphs,
        }
        df_input = pd.DataFrame(input_data)

        write_table_to_postgresql(
            df=df_input,
            table_name=table_name,
            engine=engine,
            data_field_name="data",
            encode_inplace=False,
            schema_name=schema_name,
            if_exists="append",
        )

        query = f"select * from {schema_name}.{table_name}"
        df_db = read_data_from_postgresql(sql=query, engine=engine, data_field_name="data")

        for ix in range(len(graphs)):
            g1 = df_db["data"].iloc[ix]
            g2 = df_input["data"].iloc[ix]
            assert len(g1) == len(g2)
            assert len(g1.edges()) == len(g2.edges())

    def test_write_chunks_read_compare(self, conn_postgis, clean_up_database):
        """test consistency if data is written in chunks"""

        conn_string, con = conn_postgis
        engine = sqlalchemy.create_engine(conn_string)

        schema_name = "my_graph_dataset"
        table_name = "graphs"

        create_activity_graph_standard_table(con=con, table_name=table_name, schema_name=schema_name)

        graphs = [nx.erdos_renyi_graph(10 * (i + 1), 0.2, seed=None, directed=False) for i in range(4)]
        input_data = {
            "user_id": ["1", "2", "3", "4"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
            "is_full_graph": [True, False, True, False],
            "data": graphs,
        }
        df_input = pd.DataFrame(input_data)

        write_table_to_postgresql(
            df=df_input[0:2],
            table_name=table_name,
            engine=engine,
            data_field_name="data",
            encode_inplace=False,
            schema_name=schema_name,
            if_exists="append",
        )

        write_table_to_postgresql(
            df=df_input[2:],
            table_name=table_name,
            engine=engine,
            data_field_name="data",
            encode_inplace=False,
            schema_name=schema_name,
            if_exists="append",
        )

        query = f"select * from {schema_name}.{table_name}"
        df_db = read_data_from_postgresql(sql=query, engine=engine, data_field_name="data")

        for ix in range(len(graphs)):
            g1 = df_db["data"].iloc[ix]
            g2 = df_input["data"].iloc[ix]
            assert len(g1) == len(g2)
            assert len(g1.edges()) == len(g2.edges())

    def test_write_multi_data_read_compare(self, conn_postgis, clean_up_database):
        """test writing multiple data firelds that need encoding"""
        conn_string, con = conn_postgis
        engine = sqlalchemy.create_engine(conn_string)

        schema_name = "my_graph_dataset"
        table_name = "graphs"

        field_type_dict = {
            "user_id": "text",
            "start_date": "text",
            "duration": "text",
            "is_full_graph": "boolean",
            "data": "bytea",
            "data2": "bytea",
        }

        create_table(
            psycopg_con=con,
            field_type_dict=field_type_dict,
            table_name=table_name,
            schema_name=schema_name,
            create_schema=True,
            drop_if_exists=False,
        )

        graphs = [nx.erdos_renyi_graph(10 * (i + 1), 0.2, seed=None, directed=False) for i in range(4)]
        graphs2 = [nx.erdos_renyi_graph(10 * (i + 1), 0.2, seed=None, directed=False) for i in range(4)]
        input_data = {
            "user_id": ["1", "2", "3", "4"],
            "start_date": ["10-20-20", "10-20-21", "10-20-22", "10-20-30"],
            "duration": ["20 minutes", "22 minutes", "23 minutes", "24 minutes"],
            "is_full_graph": [True, False, True, False],
            "data": graphs,
            "data2": graphs2,
        }
        df_input = pd.DataFrame(input_data)

        write_table_to_postgresql(
            df=df_input,
            table_name=table_name,
            engine=engine,
            data_field_name=["data", "data2"],
            encode_inplace=False,
            schema_name=schema_name,
            if_exists="append",
        )

        query = f"select * from {schema_name}.{table_name}"
        df_db = read_data_from_postgresql(sql=query, engine=engine, data_field_name=["data", "data2"])

        for ix in range(len(graphs)):
            g1 = df_db["data"].iloc[ix]
            g2 = df_input["data"].iloc[ix]
            assert len(g1) == len(g2)
            assert len(g1.edges()) == len(g2.edges())

        for ix in range(len(graphs)):
            g1 = df_db["data2"].iloc[ix]
            g2 = df_input["data2"].iloc[ix]
            assert len(g1) == len(g2)
            assert len(g1.edges()) == len(g2.edges())
