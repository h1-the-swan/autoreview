import os
from pathlib import Path
# from dotenv import load_dotenv, find_dotenv, DotEnv
from dotenv import load_dotenv, find_dotenv, dotenv_values
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
# dotenv_vals = load_dotenv(dotenv_path)

REQUIRED_VARS = [
]

OPTIONAL_VARS = [
    'MYSQL_HOST',
    'MYSQL_USERNAME',
    'MYSQL_PASSWORD',
    'MYSQL_DEFAULT_DB',
    'MYSQL_ENCODING', # will default to "utf8mb4"
    'MYSQL_PROTOCOL', # will default to "mysql+pymysql"
]

class Config(object):

    def __init__(self, spark_mem=None, path_to_paper_data=None, path_to_citation_data=None):
        self.PROJECT_DIR = os.environ.get('PROJECT_DIR') or Path(__file__).resolve().parents[1]

        for var in REQUIRED_VARS:
            setattr(self, var, os.environ[var])

        for var in OPTIONAL_VARS:
            setattr(self, var, os.environ.get(var, default=None))
            
        self.PATH_TO_PAPER_DATA = path_to_paper_data or os.environ.get('DEFAULT_PAPER_DATA', "")
        self.PATH_TO_CITATION_DATA = path_to_citation_data or os.environ.get('DEFAULT_PAPER_CITATION_DATA', "")

        self._mysql_db = None
        self._spark = None
        self.spark_mem = spark_mem or os.environ.get('SPARK_MEM') or '80g'

    def get(self, x, default=None):
        return getattr(self, x, default)

    @property
    def mysql_db(self):
        """database object used for connecting to MySQL"""
        # if not hasattr(self, '_mysql') or self._mysql_db is None:
        if self.get('_mysql_db') is None:
            db_name = self.MYSQL_DEFAULT_DB or None
            self._mysql_db = self._get_mysql_connection(db_name=db_name)
        return self._mysql_db

    @mysql_db.deleter
    def mysql_db(self):
        del self._mysql_db

    @property
    def spark(self):
        if self.get('_spark') is None:
            self._spark = self.load_spark_session(mem=self.spark_mem)
        return self._spark

    @spark.deleter
    def spark(self):
        if self.get('_spark') is not None:
            self._spark.stop()
        del self._spark

    def restart_spark(self):
        del self.spark
        return self.spark




    def _get_mysql_connection(self, db_name=None):
        """Use SQLAlchemy to get a connection to MySQL

        :db_name: optional name of database to connect to
        :returns: db object

        """
        mysql_settings = {
            'db_name': db_name,
            'host_name': self.MYSQL_HOST or '127.0.0.1',
            'user_name': self.MYSQL_USERNAME,
            'password': self.MYSQL_PASSWORD,
            'encoding': self.MYSQL_ENCODING or 'utf8mb4'
        }
        c = MySQLConnect(**mysql_settings)
        return c

    def reset_mysql_db(self, db_name=None):
        """Reset the MySQL DB Connection

        :db_name: optional name of database to connect to
        :returns: db object

        """
        self._mysql_db = self._get_mysql_connection(db_name=db_name)

    def load_spark_session(self, appName="sparkApp", mem='80g', showConsoleProgress=False, additional_conf=[], logLevel=None):
        # import findspark
        # findspark.init()

        import pyspark
        conf = pyspark.SparkConf().setAll([
            ('spark.executor.memory', mem), 
            ('spark.driver.memory', mem),
            ('spark.ui.showConsoleProgress', showConsoleProgress),
            ('spark.driver.maxResultSize', '0'),
            ('spark.reducer.maxSizeInFlight', '5g'),
            ("spark.sql.execution.arrow.enabled", "false"),
            ("spark.driver.extraJavaOptions", "-Duser.timezone=UTC"),  # https://stackoverflow.com/a/48767250
            ("spark.executor.extraJavaOptions", "-Duser.timezone=UTC"),
        ])
        for k,v in additional_conf:
            conf.set(k, v)

        sc = pyspark.SparkContext(appName=appName, conf=conf)
        if logLevel:
            sc.setLogLevel(logLevel)
        spark = pyspark.sql.SparkSession(sc)
        return spark

    def teardown(self):
        del self.spark
        del self.mysql_db
        

class MySQLConnect(object):

    """Use SQLAlchemy to connect to MySQL"""

    def __init__(self,
                 protocol = 'mysql+pymysql',
                 db_name = None,
                 host_name = None,
                 user_name = None,
                 password = None,
                 encoding = None,
                 module_path = None,
                 test = False):

        from sqlalchemy import create_engine, MetaData, Table
        self.create_engine = create_engine
        self.MetaData = MetaData
        self.Table = Table

        self.engine = None
        self.metadata = None
        self.tables = None

        self.db_name = db_name
        self.protocol = protocol.lower()
        self.host_name = host_name
        self.user_name = user_name
        self.password = password
        self.encoding = encoding
        self.module_path = module_path
        self.test = test

        if self.host_name:
            self.get_engine()

    def get_connection_string(self):
        con = self.protocol + "://"
        con = con + self.user_name
        if self.password:
            con = con + ":" + self.password
        con = con + "@" + self.host_name
        if self.db_name:
            con = con + "/" + self.db_name
        else:
            con = con + "/"
        con = con + "?local_infile=1"
        if self.encoding:
            con = con + "&charset=" + self.encoding
        return con

    def get_engine(self, pool_recycle=25200):
        """Get sqlalchemy engine
        :returns: sqlalchemy engine object

        """
        if not self.engine:
            self.engine = self.create_engine(self.get_connection_string(), pool_recycle=pool_recycle)
        if self.db_name:
            self.metadata = self.MetaData(self.engine)
            self.metadata.reflect()
            self.tables = self.get_all_tables()
        return self.engine

    def load_table(self, table_name=None):
        """Load a database table by name

        :table_name: database table name
        :returns: sqlalchemy Table object

        """
        return self.Table(table_name, self.metadata, autoload=True)

    def get_all_tables(self):
        """
        Returns a dictionary of {tablename: Table object}
        for the tables in the database
        """
        return {tablename: self.load_table(tablename) for tablename in self.engine.table_names()}

    # def read_sql(self, sq):
    #     """
    #     Use pandas.read_sql() to execute a SQL query to get a dataframe
    #     See pandas documentation
    #     """
    #     try:
    #         import pandas as pd
    #     except ImportError:
    #         raise ImportError("You need to install Pandas for this")
    #     return pd.read_sql(sq, self.engine)


    def __repr__(self):
        result = "DBConfiguration():\n\t"
        result = result + "PROTOCOL='%s'\n\t"%(self.protocol)
        result = result + "DATABASE_NAME='%s'\n\t"%(self.db_name)
        result = result + "DATABASE_HOST='%s'\n\t"%(self.host_name)
        result = result + "DATABASE_USER='%s'\n\t"%(self.user_name)
        if( self.password ):
            result = result + "PASSWORD <is_set>\n\t"
        else:
            result = result + "PASSWORD <is_NOT_set>\n\t"
        result = result + "MODULE_PATH='%s'"%(self.module_path)
        return result

    

