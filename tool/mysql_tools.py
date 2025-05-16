import mysql.connector
from mysql.connector import Error


class MySQLTools:
    def __init__(self, host, user, password, database='csi'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.create_schema()
        self.create_connection()
        self.create_table()

    def create_connection(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                print(f'Connected to MySQL database: {self.database}')
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")

    def create_schema(self):
        try:
            # 先不指定数据库连接
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            cursor = connection.cursor()
            cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {self.database}")
            connection.commit()
            print(f"Schema {self.database} created or already exists.")
            connection.close()
        except Error as e:
            print(f"Error while creating schema: {e}")

    def create_table(self):
        try:
            cursor = self.connection.cursor()
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {self.database}.csi_model_saved_table (
                uuid VARCHAR(36) PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                model_saved_path VARCHAR(255) NOT NULL,
                model_insert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_query)
            self.connection.commit()
            print("Table csi_model_saved_table created or already exists.")
        except Error as e:
            print(f"Error while creating table: {e}")

    def insert_record(self, uuid, model_name, model_saved_path):
        try:
            cursor = self.connection.cursor()
            insert_query = f"""
            INSERT INTO {self.database}.csi_model_saved_table (uuid, model_name, model_saved_path)
            VALUES (%s, %s, %s)
            """
            cursor.execute(insert_query, (uuid, model_name, model_saved_path))
            self.connection.commit()
            print("Record inserted successfully.")
        except Error as e:
            print(f"Error while inserting record: {e}")

    def delete_record(self, uuid):
        try:
            cursor = self.connection.cursor()
            delete_query = f"DELETE FROM {self.database}.csi_model_saved_table WHERE uuid = %s"
            cursor.execute(delete_query, (uuid,))
            self.connection.commit()
            print("Record deleted successfully.")
        except Error as e:
            print(f"Error while deleting record: {e}")

    def update_record(self, uuid, model_name=None, model_saved_path=None):
        try:
            update_query = f"UPDATE {self.database}.csi_model_saved_table SET "
            values = []
            if model_name:
                update_query += "model_name = %s, "
                values.append(model_name)
            if model_saved_path:
                update_query += "model_saved_path = %s, "
                values.append(model_saved_path)
            update_query = update_query.rstrip(', ')
            update_query += " WHERE uuid = %s"
            values.append(uuid)
            cursor = self.connection.cursor()
            cursor.execute(update_query, tuple(values))
            self.connection.commit()
            print("Record updated successfully.")
        except Error as e:
            print(f"Error while updating record: {e}")

    def select_record(self, uuid=None):
        try:
            cursor = self.connection.cursor(dictionary=True)
            if uuid:
                select_query = f"SELECT * FROM {self.database}.csi_model_saved_table WHERE uuid = %s"
                cursor.execute(select_query, (uuid,))
            else:
                select_query = f"SELECT * FROM {self.database}.csi_model_saved_table"
                cursor.execute(select_query)
            records = cursor.fetchall()
            return records
        except Error as e:
            print(f"Error while selecting record: {e}")
            return []

    def close_connection(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection is closed.")


if __name__ == "__main__":
    # 使用示例
    tool = MySQLTools(host='localhost', user='your_username', password='your_password')
    # 插入记录
    tool.insert_record('123e4567-e89b-12d3-a456-426614174000', 'test_model', '/path/to/model')
    # 查询所有记录
    all_records = tool.select_record()
    print(all_records)
    # 查询指定记录
    specific_record = tool.select_record('123e4567-e89b-12d3-a456-426614174000')
    print(specific_record)
    # 更新记录
    tool.update_record('123e4567-e89b-12d3-a456-426614174000', model_name='new_model_name')
    # 删除记录
    tool.delete_record('123e4567-e89b-12d3-a456-426614174000')
    tool.close_connection()

