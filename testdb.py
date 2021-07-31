import sqlite3


# conn = sqlite3.connect('directory/test.db')
# c = conn.cursor()
# print ("Opened database successfully")
# cursor = c.execute("SELECT ID, Name  from person")
# print(cursor)
#
# for row in cursor:
#
#    print ("ID = ", row[0])
#    print ("NAME = ", row[1])



def get_sql_conn():
    """
    获取数据库连接
    """
    conn= sqlite3.connect('directory/test.db')
    cursor = conn.cursor()
    return conn,cursor

def get_index_dict(cursor):

    """
    获取数据库对应表中的字段名
    """
    index_dict=dict()
    index=0
    for desc in cursor.description:
        index_dict[desc[0]]=index
        index=index+1
    return index_dict

def get_dict_data_sql(cursor,sql):
    """
   运行sql语句，获取结果，并根据表中字段名，转化成dict格式（默认是tuple格式）
    """
    cursor.execute(sql)
    cursor.execute("SELECT ID, Name  from person")

    data=cursor.fetchall()
    #index_dict=get_index_dict(cursor)
    res=[]
    resi={}

    for datai in data:
        print(datai[0],datai[1])

        #for indexi in index_dict:
        #   resi[indexi]=datai[index_dict[indexi]]
        resi[datai[0]]=datai[1]

    #print(list(resi.keys())[list(resi.values()).index("范治国2")])

    return resi

def dbmain(sql):
    try:
        con,cursor = get_sql_conn()
        #sql = "SELECT ID, Name  from person"
        result=get_dict_data_sql(cursor, sql)
        #print(result)
        con.commit()
        print ("Records created successfully")
        con.close()
        return result
    except:
       pass
if __name__ == '__main__':
    # sql="INSERT INTO person (ID,Name) \
    #      VALUES( '3', 'Paul')"
    # sql="SELECT ID, Name  from person"
    sql = "DELETE from person  where id != 0"
    # sql = "DELETE from person  where id = 12"
    r=dbmain(sql)
    print(r)

